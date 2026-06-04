from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

# ── Backend 선택 (env → 기본값 bgem3) ────────────────────────────────────
DENSE_BACKEND = os.getenv("FAIRDATA_DENSE_BACKEND", "bgem3")
SPARSE_BACKEND = os.getenv("FAIRDATA_SPARSE_BACKEND", "bgem3")
MULTIVECTOR_BACKEND = os.getenv("FAIRDATA_MULTIVECTOR_BACKEND", "bgem3")
RERANK_BACKEND = os.getenv("FAIRDATA_RERANK_BACKEND", "bge_reranker")
RERANK_TOP_N = int(os.getenv("FAIRDATA_RERANK_TOP_N", "50"))
RERANK_WEIGHT = float(os.getenv("FAIRDATA_RERANK_WEIGHT", "1.0"))
USE_RRF_FUSION = os.getenv("FAIRDATA_USE_RRF_FUSION", "1") == "1"

# ── LLM backend ───────────────────────────────────────────────────────────
LLM_BACKEND = os.getenv("FAIRDATA_LLM_BACKEND", "qwen")


# 원천 코퍼스 디렉터리를 반환합니다.
def resolve_data_dir() -> Path:
    return BASE_DIR / "data" / "raw"


# 검색 경로 공통 기본 BGE-M3 모델 디렉터리를 반환합니다.
def resolve_bgem3_model_dir() -> Path:
    return BASE_DIR / "models" / "bge-m3"


# dense 전용 모델 디렉터리를 반환합니다.
def resolve_dense_model_dir() -> Path:
    custom = os.getenv("FAIRDATA_DENSE_MODEL_DIR")
    if custom:
        return Path(custom)
    return resolve_bgem3_model_dir()


# sparse 전용 모델 디렉터리를 반환합니다.
def resolve_sparse_model_dir() -> Path:
    custom = os.getenv("FAIRDATA_SPARSE_MODEL_DIR")
    if custom:
        return Path(custom)
    return resolve_bgem3_model_dir()


# multivector 전용 모델 디렉터리를 반환합니다.
def resolve_multivector_model_dir() -> Path:
    custom = os.getenv("FAIRDATA_MULTIVECTOR_MODEL_DIR")
    if custom:
        return Path(custom)
    return resolve_bgem3_model_dir()


# reranker 모델 디렉터리를 반환합니다.
def resolve_bge_reranker_model_dir() -> Path:
    custom = os.getenv("FAIRDATA_RERANK_MODEL_DIR")
    if custom:
        return Path(custom)
    return BASE_DIR / "models" / "bge-reranker-v2-m3"


# reranker backend 종류를 반환합니다.
def resolve_reranker_backend_name() -> str:
    return RERANK_BACKEND


# reranker top_n을 반환합니다.
def resolve_reranker_top_n() -> int:
    return RERANK_TOP_N


# reranker fusion weight를 반환합니다.
def resolve_reranker_weight() -> float:
    return RERANK_WEIGHT


# 생성 모델 디렉터리를 반환합니다.
def resolve_qwen_model_dir() -> Path:
    return BASE_DIR / "models" / "qwen2-7b-instruct"


# LLM 모델 디렉터리를 반환합니다 (env 오버라이드 지원).
def resolve_llm_model_dir() -> Path:
    custom = os.getenv("FAIRDATA_LLM_MODEL_DIR")
    if custom:
        return Path(custom)
    return resolve_qwen_model_dir()


# LLM backend 종류를 반환합니다.
def resolve_llm_backend_name() -> str:
    return LLM_BACKEND


# 검색 인덱스 루트 디렉터리를 반환합니다.
def resolve_index_root_dir() -> Path:
    return BASE_DIR / "index"


# 인덱스 namespace를 반환합니다 (실험별 인덱스 격리).
def resolve_index_namespace() -> str:
    return os.getenv("FAIRDATA_INDEX_NAMESPACE", f"{DENSE_BACKEND}_default")


# dense 경로가 사용할 Chroma 인덱스 루트 디렉터리를 반환합니다.
def resolve_chroma_index_dir() -> Path:
    ns = resolve_index_namespace()
    return resolve_index_root_dir() / f"chroma_{ns}"


# sparse 경로 인덱스 파일의 기본 루트 경로를 반환합니다.
def resolve_sparse_index_path() -> Path:
    ns = resolve_index_namespace()
    return resolve_index_root_dir() / f"sparse_{ns}_chunks.npz"


# multivector 경로 인덱스 파일의 기본 루트 경로를 반환합니다.
def resolve_multivector_index_path() -> Path:
    ns = resolve_index_namespace()
    return resolve_index_root_dir() / f"multivector_{ns}_chunks.npz"


# dense 검색 경로 사용 여부를 반환합니다.
def is_dense_enabled() -> bool:
    return os.getenv("FAIRDATA_ENABLE_DENSE", "1") == "1"


# sparse 검색 경로 사용 여부를 반환합니다.
def is_sparse_enabled() -> bool:
    return os.getenv("FAIRDATA_ENABLE_SPARSE", "1") == "1"


# multivector 검색 경로 사용 여부를 반환합니다.
def is_multivector_enabled() -> bool:
    return os.getenv("FAIRDATA_ENABLE_MULTIVECTOR", "1") == "1"


# dense 경로가 사용할 백엔드 종류를 반환합니다.
def resolve_dense_backend_name() -> str:
    return DENSE_BACKEND


# sparse 경로가 사용할 백엔드 종류를 반환합니다.
def resolve_sparse_backend_name() -> str:
    return SPARSE_BACKEND


# multivector 경로가 사용할 백엔드 종류를 반환합니다.
def resolve_multivector_backend_name() -> str:
    return MULTIVECTOR_BACKEND


# 융합 단계에서 RRF를 사용할지 여부를 반환합니다.
def use_rrf_fusion() -> bool:
    return USE_RRF_FUSION


# dense 경로 후보 수는 호출부 기본값을 그대로 사용합니다.
def resolve_dense_path_top_k(default_top_k: int) -> int:
    custom = int(os.getenv("FAIRDATA_DENSE_TOP_K", "0"))
    if custom > 0:
        return custom
    return max(1, default_top_k)


# sparse 경로 후보 수는 호출부 기본값을 그대로 사용합니다.
def resolve_sparse_path_top_k(default_top_k: int) -> int:
    custom = int(os.getenv("FAIRDATA_SPARSE_TOP_K", "0"))
    if custom > 0:
        return custom
    return max(1, default_top_k)


# multivector 경로 후보 수는 호출부 기본값을 그대로 사용합니다.
def resolve_multivector_path_top_k(default_top_k: int) -> int:
    custom = int(os.getenv("FAIRDATA_MULTIVECTOR_TOP_K", "0"))
    if custom > 0:
        return custom
    return max(1, default_top_k)
