import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


TASK_TO_TYPE = {"asr": 1, "content": 1, "caption": 2, "cap": 2, "sqa": 3, "logic": 3}


@dataclass
class DatasetSpec:
    name: str
    task: str
    path: str
    text_field: str = ""
    audio_field: str = ""
    question_field: str = "question"
    context_field: str = "context"
    group_field: str = ""
    teacher_field: str = ""
    start: float = 0.0
    end: float = 1.0


def collate_batch(batch):
    x, attn, y, hard_ids, idx, typ, group_id = zip(*batch)
    return (
        torch.stack(x, 0),
        torch.stack(attn, 0),
        list(y),
        torch.stack(hard_ids, 0),
        torch.tensor(idx, dtype=torch.long),
        int(typ[0]),
        torch.tensor(group_id, dtype=torch.long),
    )


def _audio_group_id(x):
    payload = json.dumps(x, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _group_id(value, audio):
    if value is None:
        return _audio_group_id(audio)
    text = str(value)
    if text.isdigit():
        return int(text)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _pad_x(x, seq_len):
    x = torch.as_tensor(x, dtype=torch.long)
    if x.ndim != 2:
        raise ValueError(f"audio_codes must have shape [codebook, time], got {tuple(x.shape)}")
    rows = min(x.shape[0], 8)
    cols = min(x.shape[1], seq_len)
    out = torch.zeros((8, seq_len), dtype=torch.long)
    out[:rows, :cols] = x[:rows, :cols]
    attn = torch.zeros((seq_len,), dtype=torch.long)
    attn[:cols] = 1
    return out, attn


def _max_code(x):
    top = 0
    for row in x[:8]:
        if row:
            top = max(top, max(row))
    return int(top)


def _parse_sparse(raw):
    if raw is None:
        return None
    if isinstance(raw, dict) and "indices" in raw and "values" in raw:
        cols = [int(x) for x in raw["indices"]]
        vals = [float(x) for x in raw["values"]]
        return cols, vals
    if isinstance(raw, dict):
        cols = [int(k) for k in raw.keys()]
        vals = [float(v) for v in raw.values()]
        return cols, vals
    if isinstance(raw, list):
        cols, vals = [], []
        for item in raw:
            if isinstance(item, dict):
                cols.append(int(item.get("index", item.get("id"))))
                vals.append(float(item.get("value", item.get("score"))))
            else:
                cols.append(int(item[0]))
                vals.append(float(item[1]))
        return cols, vals
    raise ValueError(f"unsupported sparse teacher format: {type(raw)!r}")


def _sparse_table(items, vocab_size=0):
    rows, cols, vals = [], [], []
    max_col = -1
    for r, item in enumerate(items):
        if item is None:
            continue
        c, v = item
        if not c:
            continue
        c = torch.tensor(c, dtype=torch.long)
        v = torch.tensor(v, dtype=torch.float32)
        rows.append(torch.full_like(c, r))
        cols.append(c)
        vals.append(v)
        max_col = max(max_col, int(c.max().item()))
    width = max(int(vocab_size), max_col + 1)
    if not rows:
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty((0,)), (len(items), width)).coalesce()
    idx = torch.stack([torch.cat(rows), torch.cat(cols)])
    val = torch.cat(vals)
    return torch.sparse_coo_tensor(idx, val, (len(items), width)).coalesce()


def _row_l2_normalize_sparse(y, eps=1e-12):
    y = y.coalesce()
    idx = y.indices()
    vals = y.values().float()
    if idx.numel() == 0:
        return y
    norms = torch.zeros((y.size(0),), dtype=vals.dtype)
    norms.index_add_(0, idx[0], vals.square())
    vals = vals / norms.sqrt().clamp_min(eps)[idx[0]]
    return torch.sparse_coo_tensor(idx, vals, y.shape).coalesce()


def compute_teacher_hard_ids(y, group_ids, topk, batch_size=128, teacher_sim_max=0.0):
    n = int(y.size(0))
    topk = max(0, int(topk))
    out = torch.full((n, topk), -1, dtype=torch.long)
    if n <= 1 or topk <= 0:
        return out
    y_norm = _row_l2_normalize_sparse(y).cpu().coalesce()
    groups = torch.tensor(group_ids, dtype=torch.long)
    k_eff = min(topk, n - 1)
    for start in range(0, n, max(1, int(batch_size))):
        end = min(n, start + max(1, int(batch_size)))
        q = y_norm.index_select(0, torch.arange(start, end)).to_dense()
        scores = torch.sparse.mm(y_norm, q.t()).t()
        rows = torch.arange(end - start)
        cols = torch.arange(start, end)
        scores[rows, cols] = -float("inf")
        same_group = groups.unsqueeze(0).eq(groups[start:end].unsqueeze(1))
        scores.masked_fill_(same_group, -float("inf"))
        if teacher_sim_max > 0:
            scores.masked_fill_(scores >= float(teacher_sim_max), -float("inf"))
        _, idx = scores.topk(k_eff, dim=1)
        out[start:end, :k_eff] = idx
    return out


def _read_jsonl(spec, max_items=0):
    path = Path(spec.path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [line for line in f if line.strip()]
    n = len(raw_lines)
    s = max(0, min(n, int(n * float(spec.start))))
    e = max(0, min(n, int(n * float(spec.end))))
    for line in raw_lines[s:e]:
        obj = json.loads(line)
        audio = obj.get(spec.audio_field)
        if spec.task in {"sqa", "logic"}:
            q = obj.get(spec.question_field)
            c = obj.get(spec.context_field)
            text = f"{str(q).strip()} {str(c).strip()}" if q and c else None
        else:
            text = obj.get(spec.text_field)
        if not audio or not text:
            continue
        group = _group_id(obj.get(spec.group_field) if spec.group_field else None, audio)
        teacher = _parse_sparse(obj.get(spec.teacher_field)) if spec.teacher_field else None
        rows.append((audio[:8], str(text), group, teacher))
        if max_items and len(rows) >= int(max_items):
            break
    return rows


class AudioTextDataset(Dataset):
    def __init__(
        self,
        spec,
        seq_len,
        splade=None,
        splade_batch_size=128,
        teacher_vocab_size=0,
        hard_topk=0,
        hard_batch_size=128,
        hard_teacher_sim_max=0.0,
        max_items=0,
    ):
        self.spec = spec
        self.name = spec.name
        self.typ = TASK_TO_TYPE[spec.task]
        self.seq_len = int(seq_len)
        rows = _read_jsonl(spec, max_items=max_items)
        self.rows = [(x, text) for x, text, _, _ in rows]
        self.group_ids = [int(g) for _, _, g, _ in rows]
        teacher_items = [teacher for _, _, _, teacher in rows]
        if teacher_items and all(item is not None for item in teacher_items):
            self.y = _sparse_table(teacher_items, teacher_vocab_size).cpu()
        else:
            if splade is None:
                raise ValueError("SPLADE is required when teacher_field is absent")
            texts = [text for _, text in self.rows]
            self.y = splade.encode(texts, batch_size=splade_batch_size, convert_to_torch_sparse=True).cpu().coalesce()
        self.vocab_size = int(self.y.size(1))
        self.hard_ids = compute_teacher_hard_ids(
            self.y,
            self.group_ids,
            topk=hard_topk,
            batch_size=hard_batch_size,
            teacher_sim_max=hard_teacher_sim_max,
        )
        y = self.y.coalesce()
        y_idx = y.indices()
        y_row = y_idx[0].long()
        y_col = y_idx[1].long()
        y_val = y.values().float()
        counts = torch.bincount(y_row, minlength=int(y.size(0))).long() if y_row.numel() else torch.zeros((int(y.size(0)),), dtype=torch.long)
        offsets = torch.zeros((int(y.size(0)) + 1,), dtype=torch.long)
        offsets[1:] = counts.cumsum(0)
        self._y_cols = y_col.contiguous()
        self._y_vals = y_val.contiguous()
        self._y_offsets = offsets
        self.code_max = max((_max_code(x) for x, _ in self.rows), default=0)

    def __len__(self):
        return len(self.rows)

    def _row_y(self, idx):
        start = int(self._y_offsets[idx])
        end = int(self._y_offsets[idx + 1])
        cols = self._y_cols[start:end].unsqueeze(0)
        vals = self._y_vals[start:end]
        return torch.sparse_coo_tensor(cols, vals, (self.vocab_size,)).coalesce()

    def __getitem__(self, idx):
        x, _ = self.rows[idx]
        x, attn = _pad_x(x, self.seq_len)
        return x, attn, self._row_y(idx), self.hard_ids[idx], idx, self.typ, self.group_ids[idx]
