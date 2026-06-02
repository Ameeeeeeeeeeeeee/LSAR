import torch
import torch.nn as nn


class SemanticEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.emb(x)


class EnvEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, num_codebooks=7):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(vocab_size, d_model) for _ in range(num_codebooks)])
        self.fuse = nn.Linear(d_model * num_codebooks, d_model)

    def forward(self, x):
        xs = [emb(x[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.stack(xs, dim=1).permute(0, 2, 1, 3)
        return self.fuse(x.flatten(-2))


class AudioTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_ff, dropout, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x, padding_mask):
        b, l = x.shape[:2]
        pos_ids = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        x = x + self.pos(pos_ids)
        return self.enc(x, src_key_padding_mask=padding_mask == 0)


def _masked_mean(h, padding_mask):
    if padding_mask is None:
        return h.mean(dim=1)
    m = padding_mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _sparse_token_max(proj, h, padding_mask):
    z = proj(h)
    a = torch.log1p(torch.relu(z))
    if padding_mask is not None:
        m = padding_mask.unsqueeze(-1).to(a.dtype)
        a = a * m + (-1e9) * (1 - m)
    return torch.clamp(torch.amax(a, dim=1), min=0.0)


class SparseProjectionHead(nn.Module):
    def __init__(self, vocab_size, d_model, pool="max", norm="none"):
        super().__init__()
        self.pool = str(pool)
        self.norm_kind = str(norm)
        if self.pool not in {"max", "meanmax"}:
            raise ValueError(f"bad head_pool: {pool}")
        if self.norm_kind not in {"none", "layernorm"}:
            raise ValueError(f"bad head_norm: {norm}")
        self.norm = nn.LayerNorm(d_model) if self.norm_kind == "layernorm" else nn.Identity()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.mean_proj = nn.Linear(d_model, vocab_size, bias=False) if self.pool == "meanmax" else None

    def forward(self, h, padding_mask):
        h = self.norm(h)
        out = _sparse_token_max(self.proj, h, padding_mask)
        if self.mean_proj is not None:
            out = out + torch.log1p(torch.relu(self.mean_proj(_masked_mean(h, padding_mask))))
        return torch.clamp(out, min=0.0)


class ASRHead(SparseProjectionHead):
    pass


class CaptionHead(SparseProjectionHead):
    pass


class SQAHead(nn.Module):
    def __init__(self, vocab_size, d_model, pool="max", norm="none"):
        super().__init__()
        self.pool = str(pool)
        self.norm_kind = str(norm)
        if self.pool not in {"max", "meanmax"}:
            raise ValueError(f"bad head_pool: {pool}")
        if self.norm_kind not in {"none", "layernorm"}:
            raise ValueError(f"bad head_norm: {norm}")
        self.norm = nn.LayerNorm(d_model) if self.norm_kind == "layernorm" else nn.Identity()
        layer = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model * 4, dropout=0.05, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=4)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.mean_proj = nn.Linear(d_model, vocab_size, bias=False) if self.pool == "meanmax" else None

    def forward(self, h, padding_mask):
        h = self.norm(h)
        h = self.enc(h, src_key_padding_mask=padding_mask == 0) if padding_mask is not None else self.enc(h)
        out = _sparse_token_max(self.proj, h, padding_mask)
        if self.mean_proj is not None:
            out = out + torch.log1p(torch.relu(self.mean_proj(_masked_mean(h, padding_mask))))
        return torch.clamp(out, min=0.0)
