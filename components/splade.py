import torch


class SPLADE:
    def __init__(self, model_path="models/splade-v3", max_active_dims=None, device=None):
        from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder

        self.model = SparseEncoder(model_path, max_active_dims=max_active_dims) if max_active_dims else SparseEncoder(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer = self.model.tokenizer

    @torch.inference_mode()
    def encode(self, texts, batch_size=128, convert_to_torch_sparse=True):
        texts = [texts] if isinstance(texts, str) else texts
        emb = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True)
        if torch.is_tensor(emb):
            emb = emb.to(self.device)
        if convert_to_torch_sparse and not emb.is_sparse:
            emb = emb.to_sparse()
        return emb
