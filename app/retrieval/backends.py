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
from app.utils.text import tokenize_text

# 실제 모델 이름과 구현 옵션을 연결하는 별칭 테이블입니다.
# dense-only SentenceTransformer 계열은 모델마다 query/document prefix 규칙이 달라서
# 런타임 생성 전에 여기서 공통 옵션을 해석합니다.
_DENSE_BACKEND_ALIASES: dict[str, dict[str, Any]] = {
    "e5": {"query_prefix": "query: ", "document_prefix": "passage: ", "trust_remote_code": False},
    "multilingual_e5_large": {
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "trust_remote_code": False,
    },
    "sbert": {"query_prefix": "", "document_prefix": "", "trust_remote_code": False},
    "ko_sbert": {"query_prefix": "", "document_prefix": "", "trust_remote_code": False},
    "jina_v4": {"query_prefix": "", "document_prefix": "", "trust_remote_code": True},
    "gte_multilingual": {"query_prefix": "", "document_prefix": "", "trust_remote_code": False},
    "kure_v1": {"query_prefix": "", "document_prefix": "", "trust_remote_code": False},
    "snowflake_ko": {"query_prefix": "", "document_prefix": "", "trust_remote_code": False},
}


def _resolve_dense_alias(backend_name: str) -> str:
    # 설정 파일의 표기 흔들림(e.g. llama-3, llama_3 같은 형태)을 줄이기 위해
    # backend 이름을 내부 비교용 별칭으로 정규화합니다.
    return backend_name.strip().lower().replace("-", "_")


class BGEM3Runtime:
    # BGE-M3 기반 세 검색 경로가 공유하는 공통 런타임입니다.
    # 하나의 하이브리드 모델에서 dense / sparse / multivector 표현을 모두 뽑을 수 있어
    # 실제 모델 객체는 한 번만 로드하고 각 backend 어댑터가 재사용합니다.
    def __init__(self, model_dir: Path) -> None:
        self.model = BGEM3HybridModel(model_dir)
        self.model_dir = self.model.model_dir


class BGEM3DenseBackend(DenseRetrievalBackend):
    # BGE-M3 런타임을 dense 경로 인터페이스로 노출합니다.
    # 상위 로직은 DenseRetrievalBackend 계약만 보므로,
    # BGE-M3 고유 API를 이 어댑터 안에서 공통 dense 인터페이스로 감쌉니다.
    def __init__(self, runtime: BGEM3Runtime) -> None:
        self.runtime = runtime
        self.model_dir = runtime.model_dir

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        # 문서 배치를 dense 임베딩 행렬로 변환합니다.
        return self.runtime.model.encode_dense(texts)

    def encode_query(self, question: str) -> list[float]:
        # 단일 질문을 dense 검색용 1개 벡터로 변환합니다.
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
        # 문서들을 learned sparse 표현으로 인코딩합니다.
        return self.runtime.model.encode_sparse(texts)

    def encode_query(self, question: str) -> Any:
        # 질문 1건을 sparse 질의 표현으로 변환합니다.
        return self.runtime.model.encode_sparse([question])

    def score_all(self, document_index: Any, query_representation: Any) -> np.ndarray:
        # sparse 행렬 내적 결과를 전체 문서 점수 벡터로 변환합니다.
        if query_representation.shape[1] == 0 or document_index.shape[1] == 0:
            return np.zeros((0,), dtype="float32")
        return (document_index @ query_representation.T).toarray().ravel().astype("float32")

    def save_index(self, index_path: Path, payload: Any) -> None:
        # scipy sparse 행렬을 npz 포맷으로 저장합니다.
        from scipy.sparse import save_npz

        save_npz(index_path, payload)

    def load_index(self, index_path: Path) -> Any:
        # 저장해 둔 sparse 행렬을 메모리로 복원합니다.
        from scipy.sparse import load_npz

        return load_npz(index_path)


class BGEM3MultiVectorBackend(MultiVectorRetrievalBackend):
    # BGE-M3 런타임을 multivector 경로 인터페이스로 노출합니다.
    def __init__(self, runtime: BGEM3Runtime) -> None:
        self.runtime = runtime
        self.model_dir = runtime.model_dir

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        # 문서별 토큰 벡터 시퀀스를 생성합니다.
        return self.runtime.model.encode_multivector(texts)

    def encode_query(self, question: str) -> np.ndarray:
        # 질문 1건의 토큰 벡터 시퀀스를 생성합니다.
        vectors = self.runtime.model.encode_multivector([question])
        if not vectors:
            return np.zeros((0, 0), dtype="float32")
        return vectors[0]

    def score(self, query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
        # backend 내부 late-interaction 스코어 함수를 그대로 위임합니다.
        return float(self.runtime.model.multivector_score(query_vectors, doc_vectors))


class STModelRuntime:
    # dense-only 비교군이 공유하는 SentenceTransformer 런타임입니다.
    # E5, SBERT, GTE 같은 dense 전용 모델은 입력 prefix 규칙만 다르고
    # 실제 encode 흐름은 같아서 공통 런타임으로 묶습니다.
    def __init__(
        self,
        model_dir: Path,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "dense-only backend를 사용하려면 sentence-transformers 패키지가 필요합니다."
            ) from exc

        self.model_dir = model_dir
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.model = SentenceTransformer(
            str(model_dir),
            trust_remote_code=trust_remote_code,
        )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        # 문서 prefix 규칙을 반영해 정규화된 dense 임베딩 행렬을 생성합니다.
        inputs = [f"{self.document_prefix}{text}" for text in texts]
        matrix = self.model.encode(inputs, normalize_embeddings=True)
        return np.asarray(matrix, dtype="float32")

    def encode_query(self, question: str) -> list[float]:
        # 질문 prefix 규칙을 반영해 단일 질의 벡터를 생성합니다.
        query_inputs = [f"{self.query_prefix}{question}"]
        matrix = self.model.encode(query_inputs, normalize_embeddings=True)
        if len(matrix) == 0:
            return []
        return np.asarray(matrix[0], dtype="float32").tolist()


class STDenseBackend(DenseRetrievalBackend):
    # SentenceTransformer 계열 모델을 dense 검색 인터페이스로 노출합니다.
    def __init__(self, runtime: STModelRuntime) -> None:
        self.runtime = runtime
        self.model_dir = runtime.model_dir

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.runtime.encode_documents(texts)

    def encode_query(self, question: str) -> list[float]:
        return self.runtime.encode_query(question)


class BM25Runtime:
    # BM25는 rank_bm25 라이브러리 기반 lexical sparse 검색을 구성합니다.
    # 학습형 임베딩 대신 토큰화 + BM25 점수화만 담당하는 sparse baseline입니다.
    def __init__(self, model_dir: Path, *, k1: float = 1.5, b: float = 0.75) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "BM25 backend를 사용하려면 rank_bm25 패키지가 필요합니다."
            ) from exc

        self._bm25_cls = BM25Okapi
        self.model_dir = model_dir
        self.k1 = k1
        self.b = b

    def encode_documents(self, texts: list[str]) -> dict[str, Any]:
        # 토큰화 코퍼스와 BM25 인덱스 객체를 함께 묶어 반환합니다.
        tokenized_corpus = [tokenize_text(text) for text in texts]
        bm25 = self._bm25_cls(tokenized_corpus, k1=self.k1, b=self.b)
        return {
            "tokenized_corpus": tokenized_corpus,
            "bm25": bm25,
        }

    def encode_query(self, question: str) -> list[str]:
        # BM25는 질문도 같은 토큰화 규칙으로 처리합니다.
        return tokenize_text(question)

    def build_index(self, tokenized_corpus: list[list[str]]) -> Any:
        # 저장된 토큰화 코퍼스로 BM25 객체를 다시 구성합니다.
        return self._bm25_cls(tokenized_corpus, k1=self.k1, b=self.b)


class BM25SparseBackend(SparseRetrievalBackend):
    # 일반 dense 모델과 결합할 수 있는 lexical sparse baseline입니다.
    # BM25 구현 세부사항을 숨기고 sparse 공통 인터페이스에 맞춰 연결합니다.
    def __init__(self, runtime: BM25Runtime) -> None:
        self.runtime = runtime
        self.model_dir = runtime.model_dir
        self.k1 = runtime.k1
        self.b = runtime.b

    def encode_documents(self, texts: list[str]) -> dict[str, Any]:
        return self.runtime.encode_documents(texts)

    def encode_query(self, question: str) -> list[str]:
        return self.runtime.encode_query(question)

    def score_all(self, document_index: Any, query_representation: list[str]) -> np.ndarray:
        # BM25 전체 문서 점수를 float32 배열로 반환합니다.
        if not query_representation:
            return np.zeros((len(document_index["tokenized_corpus"]),), dtype="float32")
        scores = document_index["bm25"].get_scores(query_representation)
        return np.asarray(scores, dtype="float32")

    def save_index(self, index_path: Path, payload: Any) -> None:
        # BM25 객체 자체를 직렬화하지 않고 토큰화 코퍼스와 하이퍼파라미터만 저장합니다.
        meta_path = index_path.with_suffix(".bm25.npz")
        tokenized_corpus = np.asarray(payload["tokenized_corpus"], dtype=object)
        np.savez_compressed(
            meta_path,
            tokenized_corpus=tokenized_corpus,
            k1=np.asarray([self.k1], dtype="float32"),
            b=np.asarray([self.b], dtype="float32"),
        )
        index_path.write_text("bm25\n", encoding="utf-8")

    def load_index(self, index_path: Path) -> Any:
        # 저장된 토큰화 코퍼스로 BM25 인덱스를 다시 복원합니다.
        meta_path = index_path.with_suffix(".bm25.npz")
        meta = np.load(meta_path, allow_pickle=True)
        tokenized_corpus = [
            [str(token) for token in doc]
            for doc in meta["tokenized_corpus"].tolist()
        ]
        return {
            "tokenized_corpus": tokenized_corpus,
            "bm25": self.runtime.build_index(tokenized_corpus),
        }


def _get_bgem3_runtime(runtime_cache: dict[tuple[str, Path], Any], model_dir: Path) -> BGEM3Runtime:
    # 같은 모델을 dense/sparse/multivector가 함께 쓸 수 있으므로 캐시로 재사용합니다.
    cache_key = ("bgem3", model_dir.resolve())
    runtime = runtime_cache.get(cache_key)
    if runtime is None:
        runtime = BGEM3Runtime(model_dir)
        runtime_cache[cache_key] = runtime
    return runtime


def _get_st_runtime(
    runtime_cache: dict[tuple[str, Path], Any],
    backend_name: str,
    model_dir: Path,
) -> STModelRuntime:
    # backend alias 와 model_dir 조합이 같으면 동일 런타임을 재사용합니다.
    alias = _resolve_dense_alias(backend_name)
    options = _DENSE_BACKEND_ALIASES[alias]
    cache_key = ("st", alias, model_dir.resolve())
    runtime = runtime_cache.get(cache_key)
    if runtime is None:
        runtime = STModelRuntime(
            model_dir,
            query_prefix=str(options["query_prefix"]),
            document_prefix=str(options["document_prefix"]),
            trust_remote_code=bool(options["trust_remote_code"]),
        )
        runtime_cache[cache_key] = runtime
    return runtime


def _get_bm25_runtime(runtime_cache: dict[tuple[str, Path], Any], model_dir: Path) -> BM25Runtime:
    # BM25도 인덱스 재구성 비용을 줄이기 위해 런타임을 캐시합니다.
    cache_key = ("bm25", model_dir.resolve())
    runtime = runtime_cache.get(cache_key)
    if runtime is None:
        runtime = BM25Runtime(model_dir)
        runtime_cache[cache_key] = runtime
    return runtime


def build_dense_backend(
    backend_name: str,
    model_dir: Path,
    runtime_cache: dict[tuple[str, Path], Any],
) -> DenseRetrievalBackend:
    # 설정값 문자열을 실제 dense backend 구현체로 변환하는 팩토리 함수입니다.
    alias = _resolve_dense_alias(backend_name)
    if alias in {"bgem3", "bge_m3", "upskyy_bgem3_ko"}:
        return BGEM3DenseBackend(_get_bgem3_runtime(runtime_cache, model_dir))
    if alias in _DENSE_BACKEND_ALIASES:
        return STDenseBackend(_get_st_runtime(runtime_cache, alias, model_dir))
    supported = ", ".join(sorted(["bgem3", "upskyy_bgem3_ko", *_DENSE_BACKEND_ALIASES.keys()]))
    raise ValueError(f"Unsupported dense backend: {backend_name}. Supported: {supported}")


def build_sparse_backend(
    backend_name: str,
    model_dir: Path,
    runtime_cache: dict[tuple[str, Path], Any],
) -> SparseRetrievalBackend:
    # sparse 경로는 현재 BGE-M3 계열과 BM25 두 종류를 지원합니다.
    alias = _resolve_dense_alias(backend_name)
    if alias in {"bgem3", "bge_m3", "upskyy_bgem3_ko"}:
        return BGEM3SparseBackend(_get_bgem3_runtime(runtime_cache, model_dir))
    if alias == "bm25":
        return BM25SparseBackend(_get_bm25_runtime(runtime_cache, model_dir))
    raise ValueError(
        f"Unsupported sparse backend: {backend_name}. Supported: bgem3-family, bm25."
    )


def build_multivector_backend(
    backend_name: str,
    model_dir: Path,
    runtime_cache: dict[tuple[str, Path], Any],
) -> MultiVectorRetrievalBackend:
    # multivector는 late-interaction 구현이 필요한데,
    # 현재 코드베이스에서는 BGE-M3 계열만 이 계약을 지원합니다.
    if _resolve_dense_alias(backend_name) in {"bgem3", "bge_m3", "upskyy_bgem3_ko"}:
        return BGEM3MultiVectorBackend(_get_bgem3_runtime(runtime_cache, model_dir))
    raise ValueError(
        f"Unsupported multivector backend: {backend_name}. Only bgem3-family is supported."
    )
