from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.retrieval.bgem3 import BGEM3HybridModel
from app.retrieval.interfaces import (
    DenseRetrievalBackend,
    MultiVectorRetrievalBackend,
    SparseRetrievalBackend,
)


class BGEM3Runtime:
    # BGE-M3 기반 세 검색 경로가 공유하는 공통 런타임입니다.
    def __init__(self, model_dir: Path) -> None:
        self.model = BGEM3HybridModel(model_dir)
        self.model_dir = self.model.model_dir


class BGEM3DenseBackend(DenseRetrievalBackend):
    # BGE-M3 런타임을 dense 경로 인터페이스로 노출합니다.
    def __init__(self, runtime: BGEM3Runtime) -> None:
        self.runtime = runtime
        self.model_dir = runtime.model_dir

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.runtime.model.encode_dense(texts)

    def encode_query(self, question: str) -> list[float]:
        query_matrix = self.runtime.model.encode_dense([question])
        if len(query_matrix) == 0:
            return []
        return query_matrix[0].astype("float32").tolist()


class BGEM3SparseBackend(SparseRetrievalBackend):
    # BGE-M3 런타임을 sparse 경로 인터페이스로 노출합니다.
    def __init__(self, runtime: BGEM3Runtime) -> None:
        self.runtime = runtime
        self.model_dir = runtime.model_dir

    def encode_documents(self, texts: list[str]) -> Any:
        return self.runtime.model.encode_sparse(texts)

    def encode_query(self, question: str) -> Any:
        return self.runtime.model.encode_sparse([question])


class BGEM3MultiVectorBackend(MultiVectorRetrievalBackend):
    # BGE-M3 런타임을 multivector 경로 인터페이스로 노출합니다.
    def __init__(self, runtime: BGEM3Runtime) -> None:
        self.runtime = runtime
        self.model_dir = runtime.model_dir

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self.runtime.model.encode_multivector(texts)

    def encode_query(self, question: str) -> np.ndarray:
        vectors = self.runtime.model.encode_multivector([question])
        if not vectors:
            return np.zeros((0, 0), dtype="float32")
        return vectors[0]

    def score(self, query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
        return float(self.runtime.model.multivector_score(query_vectors, doc_vectors))


def _get_bgem3_runtime(runtime_cache: dict[tuple[str, Path], Any], model_dir: Path) -> BGEM3Runtime:
    cache_key = ("bgem3", model_dir.resolve())
    runtime = runtime_cache.get(cache_key)
    if runtime is None:
        runtime = BGEM3Runtime(model_dir)
        runtime_cache[cache_key] = runtime
    return runtime


def build_dense_backend(
    backend_name: str,
    model_dir: Path,
    runtime_cache: dict[tuple[str, Path], Any],
) -> DenseRetrievalBackend:
    if backend_name == "bgem3":
        return BGEM3DenseBackend(_get_bgem3_runtime(runtime_cache, model_dir))
    raise ValueError(f"Unsupported dense backend: {backend_name}")


def build_sparse_backend(
    backend_name: str,
    model_dir: Path,
    runtime_cache: dict[tuple[str, Path], Any],
) -> SparseRetrievalBackend:
    if backend_name == "bgem3":
        return BGEM3SparseBackend(_get_bgem3_runtime(runtime_cache, model_dir))
    raise ValueError(f"Unsupported sparse backend: {backend_name}")


def build_multivector_backend(
    backend_name: str,
    model_dir: Path,
    runtime_cache: dict[tuple[str, Path], Any],
) -> MultiVectorRetrievalBackend:
    if backend_name == "bgem3":
        return BGEM3MultiVectorBackend(_get_bgem3_runtime(runtime_cache, model_dir))
    raise ValueError(f"Unsupported multivector backend: {backend_name}")
