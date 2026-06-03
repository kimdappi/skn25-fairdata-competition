from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class DenseRetrievalBackend(Protocol):
    # dense 검색 경로가 요구하는 질의/문서 임베딩 인터페이스입니다.
    model_dir: Path

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, question: str) -> list[float]: ...


class SparseRetrievalBackend(Protocol):
    # sparse 검색 경로가 요구하는 질의/문서 인코딩 인터페이스입니다.
    model_dir: Path

    def encode_documents(self, texts: list[str]) -> Any: ...

    def encode_query(self, question: str) -> Any: ...


class MultiVectorRetrievalBackend(Protocol):
    # multivector 검색 경로가 요구하는 late-interaction 인터페이스입니다.
    model_dir: Path

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def encode_query(self, question: str) -> np.ndarray: ...

    def score(self, query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float: ...
