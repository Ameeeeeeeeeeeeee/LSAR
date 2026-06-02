import torch
import torch.nn as nn

from components.layers import ASRHead, AudioTransformerEncoder, CaptionHead, EnvEmbedding, SQAHead, SemanticEmbedding


class AudioEmbedding(nn.Module):
    def __init__(self, codebook_vocab_size, d_model):
        super().__init__()
        self.semantic = SemanticEmbedding(codebook_vocab_size, d_model)
        self.env = EnvEmbedding(codebook_vocab_size, d_model, num_codebooks=7)
        self.fuse = nn.Linear(d_model * 2, d_model)

    def forward(self, x, semantic_scale=1.0, env_scale=1.0):
        x_sem = self.semantic(x[:, 0]) * float(semantic_scale)
        x_env = self.env(x[:, 1:8]) * float(env_scale)
        return self.fuse(torch.cat([x_sem, x_env], dim=-1))


class AudioSparseModel(nn.Module):
    def __init__(
        self,
        codebook_vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_ff,
        dropout,
        max_len,
        vocab_size,
        head_pool="max",
        head_norm="none",
    ):
        super().__init__()
        self.audio_emb = AudioEmbedding(codebook_vocab_size, d_model)
        self.encoder = AudioTransformerEncoder(d_model, nhead, num_layers, dim_ff, dropout, max_len)
        self.asr_head = ASRHead(vocab_size, d_model, pool=head_pool, norm=head_norm)
        self.caption_head = CaptionHead(vocab_size, d_model, pool=head_pool, norm=head_norm)
        self.sqa_head = SQAHead(vocab_size, d_model, pool=head_pool, norm=head_norm)
        self.codebook_scales = {1: (1.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 1.0)}

    def set_codebook_scales(self, asr_sem=1.0, asr_env=1.0, caption_sem=1.0, caption_env=1.0, sqa_sem=1.0, sqa_env=1.0):
        self.codebook_scales = {
            1: (float(asr_sem), float(asr_env)),
            2: (float(caption_sem), float(caption_env)),
            3: (float(sqa_sem), float(sqa_env)),
        }

    def forward(self, x, padding_mask, typ):
        typ = int(typ)
        sem_scale, env_scale = self.codebook_scales.get(typ, (1.0, 1.0))
        h = self.encoder(self.audio_emb(x, sem_scale, env_scale), padding_mask)
        if typ == 1:
            return self.asr_head(h, padding_mask)
        if typ == 2:
            return self.caption_head(h, padding_mask)
        if typ == 3:
            return self.sqa_head(h, padding_mask)
        raise ValueError(f"bad typ: {typ}")
