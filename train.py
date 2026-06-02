import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from components.datasets import AudioTextDataset, DatasetSpec, collate_batch, _pad_x
from components.loss import RetrievalLoss
from components.models import AudioSparseModel
from components.splade import SPLADE


def mean_stats(items):
    out = {}
    if not items:
        return out
    for stats in items:
        for k, v in stats.items():
            out[k] = out.get(k, 0.0) + float(v)
    return {k: v / len(items) for k, v in out.items()}


def amp_autocast(enabled):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def amp_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def parse_data_spec(text, default_teacher_field=""):
    parts = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            parts.setdefault("path", item)
            continue
        k, v = item.split("=", 1)
        parts[k.strip()] = v.strip()
    task = parts.get("task", parts.get("kind", "caption")).lower()
    if task == "cap":
        task = "caption"
    if task not in {"asr", "content", "caption", "sqa", "logic"}:
        raise ValueError(f"unknown task in data spec: {task}")
    path = parts.get("path")
    if not path:
        raise ValueError(f"data spec requires path=: {text}")
    if task in {"asr", "content"}:
        text_field = parts.get("text", parts.get("text_field", "transcript"))
        audio_field = parts.get("audio", parts.get("audio_field", "audio_codes"))
    elif task == "caption":
        text_field = parts.get("text", parts.get("text_field", "caption"))
        audio_field = parts.get("audio", parts.get("audio_field", "audio_codes"))
    else:
        text_field = ""
        audio_field = parts.get(
            "audio", parts.get("audio_field", "context_audio_codes")
        )
    return DatasetSpec(
        name=parts.get("name", f"{task}_{Path(path).stem}"),
        task=task,
        path=path,
        text_field=text_field,
        audio_field=audio_field,
        question_field=parts.get("question", parts.get("question_field", "question")),
        context_field=parts.get("context", parts.get("context_field", "context")),
        group_field=parts.get("group", parts.get("group_field", "")),
        teacher_field=parts.get(
            "teacher", parts.get("teacher_field", default_teacher_field)
        ),
        start=float(parts.get("start", 0.0)),
        end=float(parts.get("end", 1.0)),
    )


def make_loaders(datasets, batch_size, shuffle, num_workers):
    kw = {}
    if num_workers:
        kw["prefetch_factor"] = 1
        kw["persistent_workers"] = False
    return [
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=True,
            **kw,
        )
        for ds in datasets
    ]


def make_order(loaders, train, max_steps=0):
    order = []
    for i, loader in enumerate(loaders):
        order.extend([i] * len(loader))
    if train:
        random.shuffle(order)
    if max_steps and max_steps > 0:
        if train and max_steps > len(order):
            order.extend(
                random.choices(list(range(len(loaders))), k=max_steps - len(order))
            )
        order = order[:max_steps]
    return order


def load_checkpoint(model, path, device, partial=False):
    state = torch.load(path, map_location=device)
    if not partial:
        model.load_state_dict(state)
        return {"loaded": len(state), "skipped": 0, "partial": False}
    current = model.state_dict()
    loaded, skipped = {}, []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            loaded[key] = value
        else:
            skipped.append(key)
    current.update(loaded)
    model.load_state_dict(current)
    return {"loaded": len(loaded), "skipped": len(skipped), "partial": True}


def first_sparse_rows(y, n):
    y = y.coalesce()
    n = min(int(n), int(y.size(0)))
    idx = y.indices()
    vals = y.values()
    keep = idx[0] < n
    return torch.sparse_coo_tensor(idx[:, keep], vals[keep], (n, y.size(1))).coalesce()


def scale_dense_for_score_p(x, score_p, eps=1e-8):
    score_p = float(score_p)
    if score_p <= 0:
        return x
    norm = x.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return x / norm.pow(score_p)


def scale_sparse_rows_for_score_p(y, score_p, device, eps=1e-8):
    y = y.coalesce().to(device)
    score_p = float(score_p)
    if score_p <= 0:
        return y
    idx = y.indices()
    vals = y.values().float()
    row = idx[0]
    row_norm = torch.zeros((y.size(0),), device=device, dtype=vals.dtype)
    row_norm.index_add_(0, row, vals.square())
    vals = vals / row_norm.sqrt().clamp_min(eps)[row].pow(score_p)
    return torch.sparse_coo_tensor(idx, vals, y.shape, device=device).coalesce()


class AudioOnlySubset(Dataset):
    def __init__(self, ds, n):
        self.ds = ds
        self.n = min(int(n), len(ds))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x, _ = self.ds.rows[idx]
        x, attn = _pad_x(x, self.ds.seq_len)
        return x, attn, self.ds.typ


def collate_audio_only(batch):
    x, attn, typ = zip(*batch)
    return torch.stack(x, 0), torch.stack(attn, 0), int(typ[0])


def sparse_rows_to_dense(y_cpu, start, end, device, dtype=torch.float32, eps=1e-8):
    out = torch.zeros((end - start, y_cpu.size(1)), device=device, dtype=dtype)
    for local, row in enumerate(range(start, end)):
        y = y_cpu[row].coalesce()
        idx = y.indices()[-1]
        if idx.numel() == 0:
            continue
        out[local, idx.to(device)] = y.values().to(device=device, dtype=dtype)
    return out / out.norm(p=2, dim=1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def refresh_hard_negatives(
    model,
    datasets,
    device,
    max_items=0,
    topk=8,
    query_batch=128,
    score_p=1.0,
    filter_same_group=True,
    mix_teacher=False,
    teacher_sim_max=0.0,
):
    stats = []
    was_training = model.training
    model.eval()
    use_amp = str(device).startswith("cuda")
    for ds in datasets:
        n = len(ds)
        m = min(n, int(max_items) if int(max_items) > 0 else n)
        width = min(int(topk), int(ds.hard_ids.size(1)) if ds.hard_ids.ndim == 2 else 0)
        if m <= 1 or width <= 0:
            stats.append({"dataset": ds.name, "items": m, "topk": 0, "updated": 0})
            continue
        loader = DataLoader(
            AudioOnlySubset(ds, m),
            batch_size=max(1, int(query_batch)),
            shuffle=False,
            collate_fn=collate_audio_only,
            num_workers=0,
            pin_memory=use_amp,
        )
        chunks = []
        for x, attn, typ in loader:
            x = x.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            with amp_autocast(use_amp):
                pred = model(x, attn, typ)
            chunks.append(scale_dense_for_score_p(pred.float(), score_p).cpu())
        pred_all = torch.cat(chunks, dim=0).to(device)
        y_first = first_sparse_rows(ds.y, m)
        y_pool = scale_sparse_rows_for_score_p(y_first, score_p, device)
        y_pool_teacher = (
            scale_sparse_rows_for_score_p(y_first, 1.0, device)
            if teacher_sim_max > 0
            else None
        )
        group_tensor = (
            torch.tensor(ds.group_ids[:m], dtype=torch.long, device=device)
            if filter_same_group
            else None
        )
        new_ids = torch.empty((m, width), dtype=torch.long)
        false_filtered = 0
        for start in range(0, m, max(1, int(query_batch))):
            end = min(m, start + max(1, int(query_batch)))
            scores = torch.sparse.mm(y_pool, pred_all[start:end].t()).t()
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            scores[rows, cols] = -float("inf")
            if group_tensor is not None:
                same_group = group_tensor.unsqueeze(0).eq(
                    group_tensor[start:end].unsqueeze(1)
                )
                scores.masked_fill_(same_group, -float("inf"))
            if y_pool_teacher is not None:
                pre_k = min(scores.size(1), max(width * 8, width + 32))
                pre_scores, pre_idx = scores.topk(pre_k, dim=1)
                pos_dense = sparse_rows_to_dense(
                    ds.y, start, end, device, dtype=scores.dtype
                )
                flat_idx = pre_idx.reshape(-1)
                cand = y_pool_teacher.index_select(0, flat_idx).coalesce()
                cidx = cand.indices()
                cvals = cand.values().to(device=device, dtype=scores.dtype)
                owner = torch.arange(end - start, device=device).repeat_interleave(
                    pre_k
                )
                sim_flat = scores.new_zeros((flat_idx.numel(),))
                if cidx.numel():
                    src = cvals * pos_dense[owner[cidx[0]], cidx[1]]
                    sim_flat.index_add_(0, cidx[0], src)
                teacher_sim = sim_flat.view(end - start, pre_k)
                false_mask = teacher_sim >= float(teacher_sim_max)
                false_filtered += int(false_mask.sum().item())
                pre_scores = pre_scores.masked_fill(false_mask, -float("inf"))
                _, take = pre_scores.topk(width, dim=1)
                idx = pre_idx.gather(1, take)
            else:
                _, idx = scores.topk(width, dim=1)
            new_ids[start:end] = idx.cpu()
        assigned = new_ids
        if mix_teacher:
            teacher_ids = ds.hard_ids[:m, :width].clone()
            mixed = teacher_ids.clone()
            online_cols = (width + 1) // 2
            teacher_cols = width // 2
            mixed[:, 0 : 2 * online_cols : 2] = new_ids[:, :online_cols]
            if teacher_cols > 0:
                mixed[:, 1 : 1 + 2 * teacher_cols : 2] = teacher_ids[:, :teacher_cols]
            assigned = mixed
        ds.hard_ids[:m, :width] = assigned
        stats.append(
            {
                "dataset": ds.name,
                "items": m,
                "topk": width,
                "updated": int(m * width),
                "score_p": float(score_p),
                "false_filtered": int(false_filtered),
            }
        )
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if was_training:
        model.train()
    return stats


def run_epoch(
    model,
    loaders,
    loss_fn,
    optimizer,
    scaler,
    device,
    train,
    max_steps=0,
    y_on_device=True,
    score_p=1.0,
):
    if not loaders:
        return {}
    stats_all = []
    order = make_order(loaders, train=train, max_steps=max_steps)
    iters = [None for _ in loaders]
    y_tables = [None for _ in loaders]
    use_amp = str(device).startswith("cuda")
    model.train(train)
    loss_fn.train(train)
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for li in tqdm(order, leave=False):
            loader = loaders[li]
            ds = loader.dataset
            if iters[li] is None:
                iters[li] = iter(loader)
            try:
                batch = next(iters[li])
            except StopIteration:
                iters[li] = iter(loader)
                batch = next(iters[li])
            if y_tables[li] is None:
                y_tables[li] = ds.y.to(device) if y_on_device else ds.y
            x, attn, y_list, hard_ids, idxs, typ, group_ids = batch
            x = x.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            group_ids = group_ids.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                pred = model(x, attn, typ)
                loss, stats = loss_fn(
                    pred,
                    y_tables[li],
                    y_list,
                    hard_ids,
                    idxs,
                    typ=typ,
                    group_ids=group_ids,
                    group_all=ds.group_ids,
                    score_p=score_p,
                )
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            stats_all.append(stats)
    return mean_stats(stats_all)


def print_stats(epoch, train_stats, val_stats, log_file):
    def fmt(prefix, stats):
        if not stats:
            return f"{prefix} skipped"
        keys = [
            "loss",
            "main",
            "audio_to_text",
            "text_to_audio",
            "distribution",
            "topk_kl",
            "topk_mse",
            "rank_loss",
            "rank_kth_loss",
            "margin_mse",
            "reg",
            "flops",
            "nnz_reg",
            "hard_k_eff",
            "filtered_hard",
            "acc",
            "acc_strict",
            "pos",
            "neg_in",
            "neg_hard",
        ]
        return prefix + " " + " ".join(f"{k}={stats.get(k, 0.0):.4f}" for k in keys)

    lines = [f"Epoch {epoch}", fmt("train", train_stats), fmt("val  ", val_stats)]
    print("\n".join(lines), flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_datasets(args, specs, splade, hard_topk, max_items):
    return [
        AudioTextDataset(
            spec,
            seq_len=args.max_len,
            splade=splade,
            splade_batch_size=args.splade_batch_size,
            teacher_vocab_size=args.teacher_vocab_size,
            hard_topk=hard_topk,
            hard_batch_size=args.hard_batch_size,
            hard_teacher_sim_max=args.teacher_neg_sim_max,
            max_items=max_items,
        )
        for spec in specs
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", action="append", default=[])
    ap.add_argument("--val_data", action="append", default=[])
    ap.add_argument("--teacher_field", default="")
    ap.add_argument("--teacher_vocab_size", type=int, default=0)
    ap.add_argument("--splade_model", default="models/splade-v3")
    ap.add_argument("--splade_batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--score_p_train", type=float, default=1.0)
    ap.add_argument("--score_p_eval", type=float, default=0.0)
    ap.add_argument("--eval_p_sweep", default="")
    ap.add_argument("--score_b", type=float, default=0.0)
    ap.add_argument("--score_b_start", type=float, default=-1.0)
    ap.add_argument("--score_b_final", type=float, default=-1.0)
    ap.add_argument("--score_b_steps", type=int, default=0)
    ap.add_argument("--bidirectional_weight", type=float, default=1.0)
    ap.add_argument("--distribution_weight", type=float, default=0.05)
    ap.add_argument("--teacher_tau", type=float, default=0.1)
    ap.add_argument("--topk_levels", default="16,32,64,128")
    ap.add_argument("--topk_mse_weight", type=float, default=0)
    ap.add_argument("--topk_kl_weight", type=float, default=0)
    ap.add_argument("--topk_kl_tau", type=float, default=1.0)
    ap.add_argument("--topk_kl_teacher_tau", type=float, default=1.0)
    ap.add_argument("--rank_margin_weight", type=float, default=0.05)
    ap.add_argument("--rank_margin", type=float, default=0.1)
    ap.add_argument("--rank_kth_weight", type=float, default=0.05)
    ap.add_argument("--rank_kth", type=int, default=8)
    ap.add_argument("--rank_kth_margin", type=float, default=0.05)
    ap.add_argument("--margin_mse_weight", type=float, default=0.05)
    ap.add_argument("--hard_k", type=int, default=8)
    ap.add_argument("--hard_k_start", type=int, default=0)
    ap.add_argument("--hard_k_final", type=int, default=8)
    ap.add_argument("--hard_k_steps", type=int, default=1000)
    ap.add_argument("--hard_rank_start", type=int, default=0)
    ap.add_argument("--hard_rank_end", type=int, default=0)
    ap.add_argument("--hard_batch_size", type=int, default=128)
    ap.add_argument("--random_k", type=int, default=0)
    ap.add_argument("--filter_same_group", type=int, default=1)
    ap.add_argument("--filter_term_overlap", type=int, default=0)
    ap.add_argument("--teacher_neg_sim_max", type=float, default=0.9)
    ap.add_argument("--multi_positive", type=int, default=1)
    ap.add_argument("--refresh_hard_at_start", type=int, default=0)
    ap.add_argument("--refresh_hard_every_epochs", type=int, default=5)
    ap.add_argument("--refresh_hard_items", type=int, default=0)
    ap.add_argument("--refresh_hard_topk", type=int, default=0)
    ap.add_argument("--refresh_hard_query_batch", type=int, default=128)
    ap.add_argument("--refresh_hard_score_p", type=float, default=-1.0)
    ap.add_argument("--refresh_hard_mix_teacher", type=int, default=0)
    ap.add_argument("--l1_weight", type=float, default=1e-4)
    ap.add_argument("--flops_weight", type=float, default=1e-4)
    ap.add_argument("--df_flops_weight", type=float, default=0.0)
    ap.add_argument("--nnz_target", type=float, default=0.0)
    ap.add_argument("--nnz_weight", type=float, default=0.0)
    ap.add_argument("--nnz_eps", type=float, default=1e-4)
    ap.add_argument("--nnz_gamma", type=float, default=1e-2)
    ap.add_argument("--asr_weight", type=float, default=1.0)
    ap.add_argument("--caption_weight", type=float, default=1.5)
    ap.add_argument("--sqa_weight", type=float, default=1.0)
    ap.add_argument("--asr_l1_weight", type=float, default=-1.0)
    ap.add_argument("--caption_l1_weight", type=float, default=-1.0)
    ap.add_argument("--sqa_l1_weight", type=float, default=-1.0)
    ap.add_argument("--codebook_vocab_size", type=int, default=2048)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--nhead", type=int, default=12)
    ap.add_argument("--num_layers", type=int, default=16)
    ap.add_argument("--dim_ff", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--head_pool", default="max", choices=["max", "meanmax"])
    ap.add_argument("--head_norm", default="none", choices=["none", "layernorm"])
    ap.add_argument("--asr_sem_scale", type=float, default=1.0)
    ap.add_argument("--asr_env_scale", type=float, default=1.0)
    ap.add_argument("--caption_sem_scale", type=float, default=1.0)
    ap.add_argument("--caption_env_scale", type=float, default=1.0)
    ap.add_argument("--sqa_sem_scale", type=float, default=1.0)
    ap.add_argument("--sqa_env_scale", type=float, default=1.0)
    ap.add_argument("--output_dir", default="runs")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--partial_ckpt", type=int, default=0)
    ap.add_argument("--train_steps", type=int, default=0)
    ap.add_argument("--val_steps", type=int, default=0)
    ap.add_argument("--max_train_items", type=int, default=0)
    ap.add_argument("--max_val_items", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--y_on_device", type=int, default=1)
    ap.add_argument("--torch_threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    if not args.train_data:
        raise ValueError("provide at least one --train_data spec")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    train_specs = [parse_data_spec(x, args.teacher_field) for x in args.train_data]
    val_specs = [parse_data_spec(x, args.teacher_field) for x in args.val_data]
    needs_splade = any(not spec.teacher_field for spec in train_specs + val_specs)
    splade = SPLADE(args.splade_model, device=args.device) if needs_splade else None
    hard_width = max(
        args.hard_k, args.hard_k_final, args.refresh_hard_topk, args.rank_kth
    )
    train_datasets = build_datasets(
        args, train_specs, splade, hard_width, args.max_train_items
    )
    val_datasets = (
        build_datasets(args, val_specs, splade, hard_width, args.max_val_items)
        if val_specs
        else []
    )
    if not train_datasets:
        raise ValueError("no training datasets loaded")

    train_loaders = make_loaders(
        train_datasets, args.batch_size, True, args.num_workers
    )
    val_loaders = (
        make_loaders(val_datasets, args.batch_size, False, args.num_workers)
        if val_datasets
        else []
    )
    vocab_size = max(ds.vocab_size for ds in train_datasets + val_datasets)
    codebook_vocab_size = max(
        args.codebook_vocab_size,
        max(ds.code_max for ds in train_datasets + val_datasets) + 1,
    )

    model = AudioSparseModel(
        codebook_vocab_size,
        args.d_model,
        args.nhead,
        args.num_layers,
        args.dim_ff,
        args.dropout,
        args.max_len,
        vocab_size,
        head_pool=args.head_pool,
        head_norm=args.head_norm,
    ).to(args.device)
    model.set_codebook_scales(
        asr_sem=args.asr_sem_scale,
        asr_env=args.asr_env_scale,
        caption_sem=args.caption_sem_scale,
        caption_env=args.caption_env_scale,
        sqa_sem=args.sqa_sem_scale,
        sqa_env=args.sqa_env_scale,
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{timestamp}_{args.name}" if args.name else timestamp
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.ckpt:
        ckpt_info = load_checkpoint(
            model, args.ckpt, args.device, partial=bool(args.partial_ckpt)
        )
        with open(save_dir / "ckpt_info.json", "w", encoding="utf-8") as f:
            json.dump(ckpt_info, f, indent=2, ensure_ascii=False)
        print(f"checkpoint loaded: {ckpt_info}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = amp_scaler(str(args.device).startswith("cuda"))
    loss_fn = RetrievalLoss(
        temperature=args.temperature,
        bidirectional_weight=args.bidirectional_weight,
        distribution_weight=args.distribution_weight,
        teacher_tau=args.teacher_tau,
        topk_levels=args.topk_levels,
        topk_mse_weight=args.topk_mse_weight,
        topk_kl_weight=args.topk_kl_weight,
        topk_kl_tau=args.topk_kl_tau,
        topk_kl_teacher_tau=args.topk_kl_teacher_tau,
        rank_margin_weight=args.rank_margin_weight,
        rank_margin=args.rank_margin,
        rank_kth_weight=args.rank_kth_weight,
        rank_kth=args.rank_kth,
        rank_kth_margin=args.rank_kth_margin,
        margin_mse_weight=args.margin_mse_weight,
        hard_k=args.hard_k,
        random_k=args.random_k,
        hard_k_start=args.hard_k_start,
        hard_k_final=args.hard_k_final,
        hard_k_steps=args.hard_k_steps,
        hard_rank_start=args.hard_rank_start,
        hard_rank_end=args.hard_rank_end,
        filter_same_group=bool(args.filter_same_group),
        filter_term_overlap=bool(args.filter_term_overlap),
        teacher_neg_sim_max=args.teacher_neg_sim_max,
        multi_positive=bool(args.multi_positive),
        l1_weight=args.l1_weight,
        flops_weight=args.flops_weight,
        df_flops_weight=args.df_flops_weight,
        nnz_target=args.nnz_target,
        nnz_weight=args.nnz_weight,
        nnz_eps=args.nnz_eps,
        nnz_gamma=args.nnz_gamma,
        asr_weight=args.asr_weight,
        caption_weight=args.caption_weight,
        sqa_weight=args.sqa_weight,
        asr_l1_weight=args.asr_l1_weight,
        caption_l1_weight=args.caption_l1_weight,
        sqa_l1_weight=args.sqa_l1_weight,
        score_p=args.score_p_train,
        score_b=args.score_b,
        score_b_start=args.score_b_start,
        score_b_final=args.score_b_final,
        score_b_steps=args.score_b_steps,
        seed=args.seed,
    )

    log_file = save_dir / "log.txt"
    for ep in range(1, args.epochs + 1):
        do_refresh = bool(args.refresh_hard_at_start and ep == 1) or (
            args.refresh_hard_every_epochs > 0
            and ep > 1
            and (ep - 1) % args.refresh_hard_every_epochs == 0
        )
        if do_refresh:
            refresh_topk = (
                args.refresh_hard_topk
                if args.refresh_hard_topk > 0
                else max(args.hard_k_final, args.hard_k, 1)
            )
            refresh_score_p = (
                args.refresh_hard_score_p
                if args.refresh_hard_score_p >= 0
                else args.score_p_train
            )
            refresh_stats = refresh_hard_negatives(
                model,
                train_datasets,
                args.device,
                max_items=args.refresh_hard_items,
                topk=refresh_topk,
                query_batch=args.refresh_hard_query_batch,
                score_p=refresh_score_p,
                filter_same_group=bool(args.filter_same_group),
                mix_teacher=bool(args.refresh_hard_mix_teacher),
                teacher_sim_max=args.teacher_neg_sim_max,
            )
            with open(
                save_dir / f"refresh_hard_ep{ep:02d}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(refresh_stats, f, indent=2, ensure_ascii=False)
            train_loaders = make_loaders(
                train_datasets, args.batch_size, True, args.num_workers
            )

        train_stats = run_epoch(
            model,
            train_loaders,
            loss_fn,
            optimizer,
            scaler,
            args.device,
            True,
            max_steps=args.train_steps,
            y_on_device=bool(args.y_on_device),
            score_p=args.score_p_train,
        )
        val_stats = run_epoch(
            model,
            val_loaders,
            loss_fn,
            optimizer,
            scaler,
            args.device,
            False,
            max_steps=args.val_steps,
            y_on_device=bool(args.y_on_device),
            score_p=args.score_p_eval,
        )
        print_stats(ep, train_stats, val_stats, log_file)
        if args.eval_p_sweep and val_loaders:
            for p in parse_float_list(args.eval_p_sweep):
                sweep_stats = run_epoch(
                    model,
                    val_loaders,
                    loss_fn,
                    optimizer,
                    scaler,
                    args.device,
                    False,
                    max_steps=args.val_steps,
                    y_on_device=bool(args.y_on_device),
                    score_p=p,
                )
                line = f"eval_p_sweep epoch={ep} p={p:.4f} loss={sweep_stats.get('loss', 0.0):.4f} acc={sweep_stats.get('acc', 0.0):.4f} acc_strict={sweep_stats.get('acc_strict', 0.0):.4f}"
                print(line, flush=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        torch.save(model.state_dict(), save_dir / f"ep{ep:02d}.pt")


if __name__ == "__main__":
    main()
