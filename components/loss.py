import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cols(y):
    idx = y.coalesce().indices()
    return idx[-1].tolist() if idx.numel() else []


def _overlap_cols(a, b):
    a = sorted(a)
    b = sorted(b)
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            return True
        if a[i] < b[j]:
            i += 1
        else:
            j += 1
    return False


def _pick_hard(pool, banned, total_n, need, rng, pos_cols=None, y_all=None, group_all=None, group_id=None, filter_term_overlap=False):
    if need <= 0:
        return []
    out = []
    used = set(int(x) for x in banned)

    def ok(j):
        if j < 0 or j >= total_n or j in used:
            return False
        if group_all is not None and group_id is not None and int(group_all[j]) == int(group_id):
            return False
        if filter_term_overlap and pos_cols is not None and y_all is not None and _overlap_cols(pos_cols, _cols(y_all[j])):
            return False
        return True

    for j in pool:
        j = int(j)
        if ok(j):
            out.append(j)
            used.add(j)
            if len(out) == need:
                return out

    tries = 0
    while len(out) < need and tries < max(total_n * 4, 1024):
        tries += 1
        j = rng.randrange(total_n)
        if ok(j):
            out.append(j)
            used.add(j)

    for j in range(total_n):
        if len(out) == need:
            break
        if ok(j):
            out.append(j)
            used.add(j)
    return out


def _masked_mean(x, mask):
    if x.numel() == 0:
        return x.new_zeros((x.shape[0],))
    den = mask.sum(dim=1).clamp_min(1).to(x.dtype)
    out = (x * mask.to(x.dtype)).sum(dim=1) / den
    return out.masked_fill(~mask.any(dim=1), 0.0)


def _parse_levels(levels):
    if isinstance(levels, str):
        vals = [int(x.strip()) for x in levels.split(",") if x.strip()]
    else:
        vals = [int(x) for x in levels]
    return sorted({x for x in vals if x > 0})


class RetrievalLoss(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        bidirectional_weight=0.2,
        distribution_weight=0.05,
        teacher_tau=0.1,
        topk_levels="16,32,64,128",
        topk_mse_weight=0.05,
        topk_kl_weight=0.05,
        topk_kl_tau=1.0,
        topk_kl_teacher_tau=1.0,
        rank_margin_weight=0.05,
        rank_margin=0.1,
        rank_kth_weight=0.05,
        rank_kth=8,
        rank_kth_margin=0.05,
        margin_mse_weight=0.05,
        hard_k=8,
        random_k=0,
        hard_k_start=0,
        hard_k_final=8,
        hard_k_steps=1000,
        hard_rank_start=0,
        hard_rank_end=0,
        filter_same_group=True,
        filter_term_overlap=False,
        teacher_neg_sim_max=0.9,
        multi_positive=True,
        l1_weight=1e-4,
        flops_weight=1e-4,
        df_flops_weight=0.0,
        nnz_target=0.0,
        nnz_weight=0.0,
        nnz_eps=1e-4,
        nnz_gamma=1e-2,
        asr_weight=1.0,
        caption_weight=1.5,
        sqa_weight=1.0,
        asr_l1_weight=-1.0,
        caption_l1_weight=-1.0,
        sqa_l1_weight=-1.0,
        score_p=1.0,
        score_b=0.0,
        score_b_start=-1.0,
        score_b_final=-1.0,
        score_b_steps=0,
        score_b_momentum=0.99,
        seed=0,
        eps=1e-12,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.bidirectional_weight = float(bidirectional_weight)
        self.distribution_weight = float(distribution_weight)
        self.teacher_tau = float(teacher_tau)
        self.topk_levels = _parse_levels(topk_levels)
        self.topk_mse_weight = float(topk_mse_weight)
        self.topk_kl_weight = float(topk_kl_weight)
        self.topk_kl_tau = float(topk_kl_tau)
        self.topk_kl_teacher_tau = float(topk_kl_teacher_tau)
        self.rank_margin_weight = float(rank_margin_weight)
        self.rank_margin = float(rank_margin)
        self.rank_kth_weight = float(rank_kth_weight)
        self.rank_kth = int(rank_kth)
        self.rank_kth_margin = float(rank_kth_margin)
        self.margin_mse_weight = float(margin_mse_weight)
        self.hard_k = int(hard_k)
        self.random_k = int(random_k)
        self.hard_k_start = int(hard_k_start)
        self.hard_k_final = int(hard_k_final)
        self.hard_k_steps = int(hard_k_steps)
        self.hard_rank_start = int(hard_rank_start)
        self.hard_rank_end = int(hard_rank_end)
        self.filter_same_group = bool(filter_same_group)
        self.filter_term_overlap = bool(filter_term_overlap)
        self.teacher_neg_sim_max = float(teacher_neg_sim_max)
        self.multi_positive = bool(multi_positive)
        self.l1_weight = float(l1_weight)
        self.flops_weight = float(flops_weight)
        self.df_flops_weight = float(df_flops_weight)
        self.nnz_target = float(nnz_target)
        self.nnz_weight = float(nnz_weight)
        self.nnz_eps = float(nnz_eps)
        self.nnz_gamma = float(nnz_gamma)
        self.asr_weight = float(asr_weight)
        self.caption_weight = float(caption_weight)
        self.sqa_weight = float(sqa_weight)
        self.asr_l1_weight = float(asr_l1_weight)
        self.caption_l1_weight = float(caption_l1_weight)
        self.sqa_l1_weight = float(sqa_l1_weight)
        self.score_p = float(score_p)
        self.score_b = float(score_b)
        self.score_b_start = float(score_b_start)
        self.score_b_final = float(score_b_final)
        self.score_b_steps = int(score_b_steps)
        self.score_b_momentum = float(score_b_momentum)
        self.eps = float(eps)
        self.rng = random.Random(seed)
        self.register_buffer("hard_k_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("score_b_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("running_pred_norm", torch.tensor(1.0, dtype=torch.float32))
        self._df_weight_cache = {}
        self._y_norm_cache = {}

    def effective_hard_k(self):
        if self.hard_k_start >= 0 and self.hard_k_final >= 0 and self.hard_k_steps > 0:
            step = min(float(self.hard_k_step.item()), float(self.hard_k_steps))
            mix = step / float(max(self.hard_k_steps, 1))
            return max(0, int(round(self.hard_k_start + (self.hard_k_final - self.hard_k_start) * mix)))
        return max(0, self.hard_k)

    def effective_score_b(self):
        if self.score_b_start >= 0 and self.score_b_final >= 0 and self.score_b_steps > 0:
            step = min(float(self.score_b_step.item()), float(self.score_b_steps))
            mix = step / float(max(self.score_b_steps, 1))
            return self.score_b_start + (self.score_b_final - self.score_b_start) * mix
        return max(self.score_b, 0.0)

    def task_weight(self, typ):
        if typ == 1:
            return self.asr_weight
        if typ == 2:
            return self.caption_weight
        if typ == 3:
            return self.sqa_weight
        return 1.0

    def task_l1_weight(self, typ):
        if typ == 1 and self.asr_l1_weight >= 0:
            return self.asr_l1_weight
        if typ == 2 and self.caption_l1_weight >= 0:
            return self.caption_l1_weight
        if typ == 3 and self.sqa_l1_weight >= 0:
            return self.sqa_l1_weight
        return self.l1_weight

    def normalize_dense(self, x, score_p):
        if score_p <= 0:
            return x
        norm = x.norm(p=2, dim=1).clamp_min(self.eps).pow(score_p)
        return x / norm.unsqueeze(1)

    def length_penalize_dense(self, x, raw, score_b):
        if score_b <= 0:
            return x
        norms = raw.norm(p=2, dim=1).clamp_min(self.eps)
        if self.training:
            batch_avg = norms.detach().mean().to(self.running_pred_norm.device, self.running_pred_norm.dtype)
            self.running_pred_norm.mul_(self.score_b_momentum).add_(batch_avg * (1.0 - self.score_b_momentum))
        avg = self.running_pred_norm.to(device=x.device, dtype=x.dtype).clamp_min(self.eps)
        penalty = (1.0 - score_b) + score_b * norms.to(x.dtype) / avg
        return x / penalty.clamp_min(self.eps).unsqueeze(1)

    def y_norm_mean(self, y_all, device, dtype):
        key = (id(y_all), str(y_all.device), y_all.dtype, int(y_all.size(0)), int(y_all.size(1)))
        cached = self._y_norm_cache.get(key)
        if cached is None:
            y = y_all.coalesce()
            idx = y.indices()
            vals = y.values().float()
            norms = torch.zeros((y.size(0),), device=y.device, dtype=torch.float32)
            if idx.numel():
                norms.index_add_(0, idx[0], vals.square())
            cached = norms.sqrt().mean().clamp_min(self.eps).detach()
            if len(self._y_norm_cache) > 32:
                self._y_norm_cache.clear()
            self._y_norm_cache[key] = cached
        return cached.to(device=device, dtype=dtype)

    def df_weights(self, y_all, dim, device, dtype):
        key = (id(y_all), dim, str(device))
        cached = self._df_weight_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        y = y_all.coalesce()
        idx = y.indices()
        weights = torch.ones((dim,), device=device, dtype=dtype)
        if idx.numel():
            cols = idx[-1].to(device)
            df = torch.zeros((dim,), device=device, dtype=dtype)
            df.index_add_(0, cols, torch.ones_like(cols, dtype=dtype))
            weights = weights + df / df.max().clamp_min(1.0)
        if len(self._df_weight_cache) > 32:
            self._df_weight_cache.clear()
        self._df_weight_cache[key] = weights.detach()
        return weights

    def sparse_regularization(self, pred, pred_score, y_all, typ):
        reg = pred.new_zeros(())
        l1_w = self.task_l1_weight(typ)
        if l1_w > 0:
            reg = reg + pred_score.abs().sum(dim=1).mean() * l1_w
        flops = pred.new_zeros(())
        if self.flops_weight > 0:
            flops = (pred.abs().mean(dim=0) ** 2).sum() * self.flops_weight
            reg = reg + flops
        df_flops = pred.new_zeros(())
        if self.df_flops_weight > 0:
            weights = self.df_weights(y_all, pred.size(1), pred.device, pred.dtype)
            df_flops = ((pred.abs().mean(dim=0) * weights) ** 2).sum() * self.df_flops_weight
            reg = reg + df_flops
        nnz_reg = pred.new_zeros(())
        if self.nnz_weight > 0 and self.nnz_target > 0:
            soft_nnz = torch.sigmoid((pred - self.nnz_eps) / self.nnz_gamma).sum(dim=1).mean()
            nnz_reg = (soft_nnz - self.nnz_target).square() * self.nnz_weight
            reg = reg + nnz_reg
        return reg, flops, df_flops, nnz_reg

    def topk_alignment(self, pred, y_list):
        mse_losses = []
        kl_losses = []
        tau_s = max(self.topk_kl_tau, self.eps)
        tau_t = max(self.topk_kl_teacher_tau, self.eps)
        if not self.topk_levels:
            z = pred.new_zeros(())
            return z, z
        for topk in self.topk_levels:
            cur_mse = []
            cur_kl = []
            for i, y in enumerate(y_list):
                y = y.coalesce()
                idx = y.indices()[-1]
                val = y.values().float()
                if idx.numel() == 0:
                    continue
                if idx.numel() > topk:
                    top = val.topk(topk)
                    idx = idx[top.indices]
                    val = top.values
                idx = idx.to(pred.device)
                val = val.to(pred.device, pred.dtype)
                if self.topk_mse_weight > 0:
                    cur_mse.append(F.mse_loss(pred[i, idx], val, reduction="mean"))
                if self.topk_kl_weight > 0:
                    teacher_prob = F.softmax(val / tau_t, dim=0).detach()
                    student_log = F.log_softmax(pred[i, idx] / tau_s, dim=0)
                    cur_kl.append(F.kl_div(student_log, teacher_prob, reduction="sum"))
            if cur_mse:
                mse_losses.append(torch.stack(cur_mse).mean())
            if cur_kl:
                kl_losses.append(torch.stack(cur_kl).mean())
        mse = torch.stack(mse_losses).mean() * self.topk_mse_weight if mse_losses else pred.new_zeros(())
        kl = torch.stack(kl_losses).mean() * self.topk_kl_weight if kl_losses else pred.new_zeros(())
        return mse, kl

    def batch_text_logits(self, pred_score, y_list, score_p=0.0, score_b=0.0):
        bsz = len(y_list)
        dim = pred_score.size(1)
        rows, cols, vals = [], [], []
        for i, y in enumerate(y_list):
            y = y.coalesce()
            idx = y.indices()[-1]
            if idx.numel() == 0:
                continue
            rows.append(torch.full((idx.numel(),), i, dtype=torch.long))
            cols.append(idx.cpu())
            vals.append(y.values().float().cpu())
        if not vals:
            return pred_score.new_zeros((bsz, bsz))
        row = torch.cat(rows).to(pred_score.device)
        col = torch.cat(cols).to(pred_score.device)
        val = torch.cat(vals).to(pred_score.device, pred_score.dtype)
        if score_p > 0 or score_b > 0:
            row_norm = pred_score.new_zeros((bsz,))
            row_norm.index_add_(0, row, val.square())
            row_norm = row_norm.sqrt().clamp_min(self.eps)
            if score_p > 0:
                val = val / row_norm.pow(score_p)[row]
            if score_b > 0:
                avg_norm = row_norm.mean().clamp_min(self.eps)
                penalty = (1.0 - score_b) + score_b * row_norm / avg_norm
                val = val / penalty.clamp_min(self.eps)[row]
        yb = torch.sparse_coo_tensor(torch.stack([row, col]), val, (bsz, dim), device=pred_score.device, dtype=pred_score.dtype).coalesce()
        return torch.sparse.mm(yb, pred_score.t())

    def batch_teacher_logits(self, y_list, dim, device, dtype):
        bsz = len(y_list)
        rows, cols, vals = [], [], []
        for i, y in enumerate(y_list):
            y = y.coalesce()
            idx = y.indices()[-1]
            if idx.numel() == 0:
                continue
            rows.append(torch.full((idx.numel(),), i, dtype=torch.long))
            cols.append(idx.cpu())
            vals.append(y.values().float().cpu())
        if not vals:
            return torch.zeros((bsz, bsz), device=device, dtype=dtype)
        row = torch.cat(rows).to(device)
        col = torch.cat(cols).to(device)
        val = torch.cat(vals).to(device, dtype)
        yb = torch.sparse_coo_tensor(torch.stack([row, col]), val, (bsz, dim), device=device, dtype=dtype).coalesce().to_dense()
        yb = F.normalize(yb, p=2, dim=1, eps=self.eps)
        return yb @ yb.t()

    def multi_pos_loss(self, logits, pos_mask, valid_mask):
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~valid_mask, neg_inf)
        pos_logits = logits.masked_fill(~pos_mask, neg_inf)
        return -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()

    def forward(self, pred, y_all, y_list, hard_ids, idxs, typ=None, group_ids=None, group_all=None, score_p=None):
        device = pred.device
        pred = pred.float()
        typ = int(typ) if typ is not None else 0
        score_p = self.score_p if score_p is None else float(score_p)
        score_b = self.effective_score_b()
        pred_score = self.length_penalize_dense(self.normalize_dense(pred, score_p), pred, score_b)
        if self.training and self.score_b_steps > 0:
            self.score_b_step.add_(1)
        hard_k_eff = self.effective_hard_k()
        if self.training and self.hard_k_steps > 0:
            self.hard_k_step.add_(1)

        batch_ids = [int(x) for x in idxs.tolist()]
        total_n = int(y_all.size(0))
        group_all_ref = group_all if self.filter_same_group else None
        batch_groups = group_ids.tolist() if group_ids is not None else None
        rows, pos_rows, hard_rows, flat_ids, row_owner = [], [], [], [], []

        for i, pos_id in enumerate(batch_ids):
            pos_cols = _cols(y_list[i])
            pool = hard_ids[i].tolist()
            if self.hard_rank_start > 0 or self.hard_rank_end > 0:
                s = max(self.hard_rank_start, 0)
                e = self.hard_rank_end
                pool = pool[s:e if e > 0 else None]
            group_id = batch_groups[i] if batch_groups is not None else None
            hard = _pick_hard(
                pool,
                batch_ids,
                total_n,
                hard_k_eff,
                self.rng,
                pos_cols=pos_cols,
                y_all=y_all,
                group_all=group_all_ref,
                group_id=group_id,
                filter_term_overlap=self.filter_term_overlap,
            )
            if self.random_k > 0:
                hard.extend(
                    _pick_hard(
                        [],
                        batch_ids + hard,
                        total_n,
                        self.random_k,
                        self.rng,
                        pos_cols=pos_cols,
                        y_all=y_all,
                        group_all=group_all_ref,
                        group_id=group_id,
                        filter_term_overlap=self.filter_term_overlap,
                    )
                )
            cand_ids = [pos_id]
            pos_mask_row = [True]
            hard_mask_row = [False]
            for bj, other in enumerate(batch_ids):
                if bj == i:
                    continue
                cand_ids.append(other)
                same_group_positive = self.multi_positive and batch_groups is not None and batch_groups[bj] == batch_groups[i]
                pos_mask_row.append(bool(same_group_positive))
                hard_mask_row.append(False)
            cand_ids.extend(hard)
            pos_mask_row.extend([False] * len(hard))
            hard_mask_row.extend([True] * len(hard))
            rows.append(cand_ids)
            pos_rows.append(pos_mask_row)
            hard_rows.append(hard_mask_row)
            flat_ids.extend(cand_ids)
            row_owner.extend([i] * len(cand_ids))

        bsz = len(rows)
        max_len = max(len(row) for row in rows)
        mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
        pos_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
        hard_filter_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
        score = pred.new_zeros((bsz, max_len))
        teacher_sim = pred.new_zeros((bsz, max_len))

        flat_idx = torch.tensor(flat_ids, dtype=torch.long, device=y_all.device)
        cand = y_all.index_select(0, flat_idx).coalesce().to(device)
        idx = cand.indices()
        row_idx = idx[0]
        col_idx = idx[1]
        val = cand.values().to(dtype=pred.dtype)
        owner = torch.tensor(row_owner, dtype=torch.long, device=device)
        cand_val = val
        if score_p > 0 or score_b > 0:
            cand_norm = pred_score.new_zeros((len(flat_ids),))
            cand_norm.index_add_(0, row_idx, val.square())
            cand_norm = cand_norm.sqrt().clamp_min(self.eps)
            if score_p > 0:
                cand_val = cand_val / cand_norm[row_idx].pow(score_p)
            if score_b > 0:
                y_avg = self.y_norm_mean(y_all, pred_score.device, pred_score.dtype).clamp_min(self.eps)
                cand_penalty = (1.0 - score_b) + score_b * cand_norm / y_avg
                cand_val = cand_val / cand_penalty.clamp_min(self.eps)[row_idx]
        src = cand_val * pred_score[owner[row_idx], col_idx]
        score_flat = pred.new_zeros((len(flat_ids),))
        score_flat.index_add_(0, row_idx, src)

        teacher_score_flat = None
        if self.teacher_neg_sim_max > 0 or self.margin_mse_weight > 0:
            cand_norm_teacher = pred_score.new_zeros((len(flat_ids),))
            cand_norm_teacher.index_add_(0, row_idx, val.square())
            cand_norm_teacher = cand_norm_teacher.sqrt().clamp_min(self.eps)
            pos_dense = pred_score.new_zeros((bsz, pred.size(1)))
            for pi, y in enumerate(y_list):
                y = y.coalesce()
                y_idx = y.indices()[-1]
                if y_idx.numel():
                    pos_dense[pi, y_idx.to(device)] = y.values().to(device, pred.dtype)
            pos_dense = F.normalize(pos_dense, p=2, dim=1, eps=self.eps)
            teacher_src = (val / cand_norm_teacher[row_idx]) * pos_dense[owner[row_idx], col_idx]
            teacher_score_flat = pred.new_zeros((len(flat_ids),))
            teacher_score_flat.index_add_(0, row_idx, teacher_src)

        off = 0
        for i, row in enumerate(rows):
            n = len(row)
            score[i, :n] = score_flat[off : off + n]
            if teacher_score_flat is not None:
                teacher_sim[i, :n] = teacher_score_flat[off : off + n]
            mask[i, :n] = True
            pos_mask[i, :n] = torch.tensor(pos_rows[i], dtype=torch.bool, device=device)
            hard_filter_mask[i, :n] = torch.tensor(hard_rows[i], dtype=torch.bool, device=device)
            off += n

        filtered_hard = pred.new_zeros(())
        if self.teacher_neg_sim_max > 0:
            drop_mask = hard_filter_mask & (teacher_sim >= self.teacher_neg_sim_max)
            if drop_mask.any():
                filtered_hard = drop_mask.sum(dim=1).to(score.dtype).mean()
                mask = mask & ~drop_mask

        logits = (score / self.temperature).masked_fill(~mask, torch.finfo(score.dtype).min)
        audio_to_text = self.multi_pos_loss(logits, pos_mask, mask) if self.multi_positive else F.cross_entropy(logits, torch.zeros((bsz,), dtype=torch.long, device=device))

        text_to_audio = pred.new_zeros(())
        if self.bidirectional_weight > 0:
            logits_t = self.batch_text_logits(pred_score, y_list, score_p=score_p, score_b=score_b) / self.temperature
            if self.multi_positive and batch_groups is not None:
                g = torch.tensor(batch_groups, dtype=torch.long, device=device)
                pos_t = g[:, None] == g[None, :]
                text_to_audio = self.multi_pos_loss(logits_t, pos_t, torch.ones_like(pos_t))
            else:
                text_to_audio = F.cross_entropy(logits_t, torch.arange(bsz, dtype=torch.long, device=device))

        distribution = pred.new_zeros(())
        if self.distribution_weight > 0:
            student = self.batch_text_logits(pred_score, y_list, score_p=score_p, score_b=score_b) / self.temperature
            teacher = self.batch_teacher_logits(y_list, pred.size(1), device, pred.dtype)
            teacher_prob = F.softmax(teacher / max(self.teacher_tau, self.eps), dim=1).detach()
            distribution = F.kl_div(F.log_softmax(student, dim=1), teacher_prob, reduction="batchmean")

        topk_mse, topk_kl = self.topk_alignment(pred, y_list)
        valid_neg = mask & ~pos_mask
        max_neg = score.masked_fill(~valid_neg, torch.finfo(score.dtype).min).max(dim=1).values
        rank_loss = torch.relu(self.rank_margin - score[:, 0] + max_neg).mean() if self.rank_margin_weight > 0 else pred.new_zeros(())

        rank_kth_loss = pred.new_zeros(())
        if self.rank_kth_weight > 0:
            kth = max(1, self.rank_kth)
            neg_for_kth = score.masked_fill(~valid_neg, torch.finfo(score.dtype).min)
            k_eff = min(kth, neg_for_kth.size(1))
            kth_neg = neg_for_kth.topk(k_eff, dim=1).values[:, -1]
            has_kth = valid_neg.sum(dim=1) >= k_eff
            if has_kth.any():
                rank_kth_loss = torch.relu(self.rank_kth_margin - score[:, 0] + kth_neg)[has_kth].mean()

        margin_mse = pred.new_zeros(())
        if self.margin_mse_weight > 0 and teacher_score_flat is not None:
            student_margin = score[:, :1] - score
            teacher_margin = (teacher_sim[:, :1] - teacher_sim).detach()
            sq = (student_margin - teacher_margin).square()
            margin_mse = (sq * valid_neg.to(sq.dtype)).sum() / valid_neg.sum().clamp_min(1).to(sq.dtype)

        reg, flops, df_flops, nnz_reg = self.sparse_regularization(pred, pred_score, y_all, typ)
        main = (
            audio_to_text
            + self.bidirectional_weight * text_to_audio
            + self.distribution_weight * distribution
            + topk_mse
            + topk_kl
            + self.rank_margin_weight * rank_loss
            + self.rank_kth_weight * rank_kth_loss
            + self.margin_mse_weight * margin_mse
        )
        loss = self.task_weight(typ) * main + reg

        n_in = max(len(batch_ids) - 1, 0)
        neg_in = score[:, 1 : 1 + n_in].mean(dim=1) if n_in else score.new_zeros((bsz,))
        hard_scores = score[:, 1 + n_in :]
        hard_mask = mask[:, 1 + n_in :]
        pred_col = logits.argmax(dim=1)
        acc = pos_mask.gather(1, pred_col[:, None]).squeeze(1).to(score.dtype).mean()
        max_valid_neg = score.masked_fill(~valid_neg, torch.finfo(score.dtype).min).max(dim=1).values
        acc_strict = (score[:, 0] > max_valid_neg).to(score.dtype).mean()
        stats = {
            "loss": float(loss.item()),
            "main": float(main.item()),
            "audio_to_text": float(audio_to_text.item()),
            "text_to_audio": float(text_to_audio.item()),
            "distribution": float(distribution.item()),
            "topk_mse": float(topk_mse.item()),
            "topk_kl": float(topk_kl.item()),
            "rank_loss": float(rank_loss.item()),
            "rank_kth_loss": float(rank_kth_loss.item()),
            "margin_mse": float(margin_mse.item()),
            "reg": float(reg.item()),
            "flops": float(flops.item()),
            "df_flops": float(df_flops.item()),
            "nnz_reg": float(nnz_reg.item()),
            "score_p": float(score_p),
            "score_b": float(score_b),
            "hard_k_eff": float(hard_k_eff),
            "acc": float(acc.item()),
            "acc_strict": float(acc_strict.item()),
            "pos": float(score[:, 0].mean().item()),
            "neg_in": float(neg_in.mean().item()),
            "neg_hard": float(_masked_mean(hard_scores, hard_mask).mean().item()),
            "used_hard": float(hard_mask.sum(dim=1).to(score.dtype).mean().item()),
            "filtered_hard": float(filtered_hard.item()),
        }
        return loss, stats
