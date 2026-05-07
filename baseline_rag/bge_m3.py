import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from transformers import AutoModel, AutoTokenizer

from baseline_rag.runtime import preferred_torch_dtype, require_runtime_device

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


class BGEM3HybridModel:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = self.model_dir.name
        self.default_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
        self.default_max_length = int(os.getenv("EMBEDDING_MAX_LENGTH", "1024"))
        self.device = require_runtime_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        model_kwargs = {"local_files_only": True}
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = preferred_torch_dtype()
        self.model = AutoModel.from_pretrained(self.model_dir, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)
        self.model_dtype = next(self.model.parameters()).dtype

        hidden_size = int(self.model.config.hidden_size)
        self.vocab_size = int(self.model.config.vocab_size)

        sparse_state = torch.load(self.model_dir / "sparse_linear.pt", map_location="cpu")
        self.sparse_linear = torch.nn.Linear(hidden_size, 1)
        self.sparse_linear.load_state_dict(sparse_state)
        self.sparse_linear.eval()
        self.sparse_linear.to(device=self.device, dtype=self.model_dtype)

        colbert_state = torch.load(self.model_dir / "colbert_linear.pt", map_location="cpu")
        self.colbert_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.colbert_linear.load_state_dict(colbert_state)
        self.colbert_linear.eval()
        self.colbert_linear.to(device=self.device, dtype=self.model_dtype)

        self.unused_token_ids = {
            token_id
            for token_id in [
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            ]
            if token_id is not None
        }

    def _effective_max_length(self, max_length: int | None) -> int:
        requested = max_length or self.default_max_length
        return min(requested, int(self.tokenizer.model_max_length))

    @torch.inference_mode()
    def encode_dense(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> np.ndarray:
        items = list(texts)
        vectors: List[np.ndarray] = []
        effective_batch_size = batch_size or self.default_batch_size
        effective_max_length = self._effective_max_length(max_length)
        for start in range(0, len(items), effective_batch_size):
            batch = items[start : start + effective_batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=effective_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded)
            dense_vecs = F.normalize(output.last_hidden_state[:, 0], dim=-1)
            vectors.append(dense_vecs.float().cpu().numpy().astype("float32"))
        return np.vstack(vectors) if vectors else np.zeros((0, 0), dtype="float32")

    @torch.inference_mode()
    def encode_sparse(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> csr_matrix:
        items = list(texts)
        rows: List[int] = []
        cols: List[int] = []
        values: List[float] = []
        effective_batch_size = batch_size or self.default_batch_size
        effective_max_length = self._effective_max_length(max_length)

        global_row = 0
        for start in range(0, len(items), effective_batch_size):
            batch = items[start : start + effective_batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=effective_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded)
            token_weights = torch.relu(self.sparse_linear(output.last_hidden_state)).squeeze(-1)

            input_ids = encoded["input_ids"].cpu().numpy()
            attention_mask = encoded["attention_mask"].cpu().numpy()
            weights = token_weights.cpu().numpy()

            for local_row in range(input_ids.shape[0]):
                token_to_weight: dict[int, float] = {}
                for token_id, mask_value, weight in zip(input_ids[local_row], attention_mask[local_row], weights[local_row]):
                    if not mask_value:
                        continue
                    token_id = int(token_id)
                    if token_id in self.unused_token_ids:
                        continue
                    weight_value = float(weight)
                    previous = token_to_weight.get(token_id)
                    if previous is None or weight_value > previous:
                        token_to_weight[token_id] = weight_value
                for token_id, weight_value in token_to_weight.items():
                    rows.append(global_row)
                    cols.append(token_id)
                    values.append(weight_value)
                global_row += 1

        if not items:
            return csr_matrix((0, self.vocab_size), dtype=np.float32)
        matrix = csr_matrix((values, (rows, cols)), shape=(len(items), self.vocab_size), dtype=np.float32)
        return matrix

    @torch.inference_mode()
    def encode_colbert(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> List[np.ndarray]:
        items = list(texts)
        vectors: List[np.ndarray] = []
        effective_batch_size = batch_size or self.default_batch_size
        effective_max_length = self._effective_max_length(max_length)
        for start in range(0, len(items), effective_batch_size):
            batch = items[start : start + effective_batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=effective_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded)
            colbert_vecs = self.colbert_linear(output.last_hidden_state[:, 1:])
            mask = encoded["attention_mask"][:, 1:].unsqueeze(-1).to(dtype=colbert_vecs.dtype)
            colbert_vecs = F.normalize(colbert_vecs, dim=-1) * mask

            vecs_np = colbert_vecs.cpu().numpy().astype("float32")
            mask_np = encoded["attention_mask"][:, 1:].cpu().numpy()
            for sample_vecs, sample_mask in zip(vecs_np, mask_np):
                active_length = int(sample_mask.sum())
                vectors.append(sample_vecs[:active_length])
        return vectors

    def colbert_score(self, query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
        if query_vecs.size == 0 or doc_vecs.size == 0:
            return 0.0
        token_scores = query_vecs @ doc_vecs.T
        best_scores = token_scores.max(axis=1)
        return float(best_scores.mean())
