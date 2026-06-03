from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


# 현재 실행 환경에서 사용할 torch dtype을 결정합니다.
def preferred_torch_dtype() -> torch.dtype:
    dtype_name = os.getenv("FAIRDATA_TORCH_DTYPE", "float16").strip().lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    return torch.float16


# CUDA 사용 가능 여부에 따라 모델 실행 디바이스를 결정합니다.
def resolve_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BGEM3HybridModel:
    # BGE-M3 모델과 sparse, multi-vector 보조 헤드를 초기화합니다.
    def __init__(self, model_dir: Path) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BGE-M3 검색 경로를 사용하려면 transformers 패키지가 필요합니다."
            ) from exc

        self.model_dir = Path(model_dir)
        self.device = resolve_runtime_device()
        self.default_batch_size = int(os.getenv("FAIRDATA_EMBED_BATCH_SIZE", "8"))
        self.default_max_length = int(os.getenv("FAIRDATA_EMBED_MAX_LENGTH", "1024"))

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

    # 요청 길이와 토크나이저 한계를 함께 고려해 최대 길이를 결정합니다.
    def effective_max_length(self, max_length: int | None) -> int:
        requested = max_length or self.default_max_length
        return min(requested, int(self.tokenizer.model_max_length))

    # 공통 토크나이즈 단계를 수행해 모델 입력 텐서를 만듭니다.
    def tokenize_batch(self, texts: list[str], max_length: int | None) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.effective_max_length(max_length),
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    # dense retrieval용 문장 임베딩을 생성합니다.
    @torch.inference_mode()
    def encode_dense(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> np.ndarray:
        items = list(texts)
        vectors: list[np.ndarray] = []
        effective_batch_size = batch_size or self.default_batch_size
        for start in range(0, len(items), effective_batch_size):
            batch = items[start : start + effective_batch_size]
            encoded = self.tokenize_batch(batch, max_length)
            output = self.model(**encoded)
            dense_vectors = F.normalize(output.last_hidden_state[:, 0], dim=-1)
            vectors.append(dense_vectors.float().cpu().numpy().astype("float32"))
        if not vectors:
            return np.zeros((0, 0), dtype="float32")
        return np.vstack(vectors)

    # sparse retrieval용 토큰 가중치 행렬을 생성합니다.
    @torch.inference_mode()
    def encode_sparse(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
    ):
        from scipy.sparse import csr_matrix

        items = list(texts)
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        effective_batch_size = batch_size or self.default_batch_size

        global_row = 0
        for start in range(0, len(items), effective_batch_size):
            batch = items[start : start + effective_batch_size]
            encoded = self.tokenize_batch(batch, max_length)
            output = self.model(**encoded)
            token_weights = torch.relu(self.sparse_linear(output.last_hidden_state)).squeeze(-1)

            input_ids = encoded["input_ids"].cpu().numpy()
            attention_mask = encoded["attention_mask"].cpu().numpy()
            weights = token_weights.cpu().numpy()

            for local_row in range(input_ids.shape[0]):
                token_to_weight: dict[int, float] = {}
                for token_id, mask_value, weight in zip(
                    input_ids[local_row],
                    attention_mask[local_row],
                    weights[local_row],
                ):
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
        return csr_matrix((values, (rows, cols)), shape=(len(items), self.vocab_size), dtype=np.float32)

    # multi-vector retrieval용 토큰 단위 임베딩 시퀀스를 생성합니다.
    @torch.inference_mode()
    def encode_multivector(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[np.ndarray]:
        items = list(texts)
        vectors: list[np.ndarray] = []
        effective_batch_size = batch_size or self.default_batch_size
        for start in range(0, len(items), effective_batch_size):
            batch = items[start : start + effective_batch_size]
            encoded = self.tokenize_batch(batch, max_length)
            output = self.model(**encoded)
            multivectors = self.colbert_linear(output.last_hidden_state[:, 1:])
            mask = encoded["attention_mask"][:, 1:].unsqueeze(-1).to(dtype=multivectors.dtype)
            multivectors = F.normalize(multivectors, dim=-1) * mask

            vectors_np = multivectors.float().cpu().numpy().astype("float32")
            mask_np = encoded["attention_mask"][:, 1:].cpu().numpy()
            for sample_vectors, sample_mask in zip(vectors_np, mask_np):
                active_length = int(sample_mask.sum())
                vectors.append(sample_vectors[:active_length])
        return vectors

    # query와 document multi-vector 간 late interaction 점수를 계산합니다.
    def multivector_score(self, query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
        if query_vectors.size == 0 or doc_vectors.size == 0:
            return 0.0
        token_scores = query_vectors @ doc_vectors.T
        best_scores = token_scores.max(axis=1)
        return float(best_scores.mean())
