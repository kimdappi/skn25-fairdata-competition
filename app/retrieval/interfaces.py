from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


# retrieval 모듈의 상위 로직은 "어떤 모델을 쓰는지"보다
# "문서/질의를 어떻게 인코딩하고 점수를 계산할 수 있는지"에만 관심이 있습니다.
# 이 파일은 그 공통 계약(Protocol)을 정의해서 구현체 교체를 쉽게 만듭니다.

class DenseRetrievalBackend(Protocol):
    # dense 검색 경로가 요구하는 질의/문서 임베딩 인터페이스입니다.
    # 상위 검색 로직은 이 계약만 만족하면 BGE, E5, SBERT 등을 동일하게 다룰 수 있습니다.
    model_dir: Path

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, question: str) -> list[float]: ...


class SparseRetrievalBackend(Protocol):
    # sparse 검색 경로가 요구하는 질의/문서 인코딩 인터페이스입니다.
    # sparse 표현은 구현마다 자료형이 달라질 수 있어 Any 를 사용합니다.
    # 대신 문서 인코딩, 질의 인코딩, 전체 문서 점수화, 인덱스 저장/복원이라는
    # 동작 단위는 공통으로 맞춰 둡니다.
    model_dir: Path

    def encode_documents(self, texts: list[str]) -> Any: ...

    def encode_query(self, question: str) -> Any: ...

    def score_all(self, document_index: Any, query_representation: Any) -> np.ndarray: ...

    def save_index(self, index_path: Path, payload: Any) -> None: ...

    def load_index(self, index_path: Path) -> Any: ...


class MultiVectorRetrievalBackend(Protocol):
    # multivector 검색 경로가 요구하는 late-interaction 인터페이스입니다.
    # 문서/질의를 단일 벡터가 아니라 토큰 수준 벡터 묶음으로 다루고,
    # 최종 점수는 구현체별 late-interaction 스코어 함수가 계산합니다.
    model_dir: Path

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def encode_query(self, question: str) -> np.ndarray: ...

    def score(self, query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float: ...
