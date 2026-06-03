import json
import os
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from baseline_rag.runtime import preferred_torch_dtype, require_runtime_device


class LocalEmbeddingModel:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = self.model_dir.name
        self.pooling_config = self._load_pooling_config()
        self.normalize = (self.model_dir / "2_Normalize").exists()
        self.query_prefix, self.passage_prefix = self._resolve_prefixes()
        self.default_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        self.default_max_length = int(os.getenv("EMBEDDING_MAX_LENGTH", "384"))
        self.device = require_runtime_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        model_kwargs = {"local_files_only": True}
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = preferred_torch_dtype()
        self.model = AutoModel.from_pretrained(self.model_dir, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)

    def _load_pooling_config(self) -> dict:
        config_path = self.model_dir / "1_Pooling" / "config.json"
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _resolve_prefixes(self) -> tuple[str, str]:
        if self.model_name == "embedding_multilingual_e5_large":
            return "query: ", "passage: "
        return "", ""

    def _format_text(self, text: str, is_query: bool) -> str:
        prefix = self.query_prefix if is_query else self.passage_prefix
        return prefix + text if prefix else text

    def _pool(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        if self.pooling_config.get("pooling_mode_cls_token"):
            pooled = token_embeddings[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            masked = token_embeddings * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled

    @torch.inference_mode()
    def encode(
        self,
        texts: Iterable[str],
        *,
        is_query: bool,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> np.ndarray:
        items = [self._format_text(text, is_query=is_query) for text in texts]
        vectors: List[np.ndarray] = []
        effective_batch_size = batch_size or self.default_batch_size
        effective_max_length = min(max_length or self.default_max_length, int(self.tokenizer.model_max_length))
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
            pooled = self._pool(output, encoded["attention_mask"])
            vectors.append(pooled.cpu().numpy().astype("float32"))
        return np.vstack(vectors) if vectors else np.zeros((0, 0), dtype="float32")
