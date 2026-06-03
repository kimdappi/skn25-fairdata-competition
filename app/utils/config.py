from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DENSE_BACKEND = "bgem3"
SPARSE_BACKEND = "bgem3"
MULTIVECTOR_BACKEND = "bgem3"
RERANK_BACKEND = "bge_reranker"
USE_RRF_FUSION = True


# 원천 코퍼스 디렉터리를 반환합니다.
def resolve_data_dir() -> Path:
    return BASE_DIR / "data" / "raw"


# 검색 경로 공통 기본 BGE-M3 모델 디렉터리를 반환합니다.
def resolve_bgem3_model_dir() -> Path:
    return BASE_DIR / "models" / "bge-m3"


# dense 전용 모델 디렉터리를 반환합니다.
def resolve_dense_model_dir() -> Path:
    return resolve_bgem3_model_dir()


# sparse 전용 모델 디렉터리를 반환합니다.
def resolve_sparse_model_dir() -> Path:
    return resolve_bgem3_model_dir()


# multivector 전용 모델 디렉터리를 반환합니다.
def resolve_multivector_model_dir() -> Path:
    return resolve_bgem3_model_dir()


# reranker 모델 디렉터리를 반환합니다.
def resolve_bge_reranker_model_dir() -> Path:
    return BASE_DIR / "models" / "bge-reranker-v2-m3"


# reranker backend 종류를 반환합니다.
def resolve_reranker_backend_name() -> str:
    return RERANK_BACKEND


# 생성 모델 디렉터리를 반환합니다.
def resolve_qwen_model_dir() -> Path:
    return BASE_DIR / "models" / "qwen2-7b-instruct"


# 검색 인덱스 루트 디렉터리를 반환합니다.
def resolve_index_root_dir() -> Path:
    return BASE_DIR / "index"


# dense 경로가 사용할 Chroma 인덱스 루트 디렉터리를 반환합니다.
def resolve_chroma_index_dir() -> Path:
    return resolve_index_root_dir() / "chroma_bgem3"


# sparse 경로 인덱스 파일의 기본 루트 경로를 반환합니다.
def resolve_sparse_index_path() -> Path:
    return resolve_index_root_dir() / "sparse_bgem3_chunks.npz"


# multivector 경로 인덱스 파일의 기본 루트 경로를 반환합니다.
def resolve_multivector_index_path() -> Path:
    return resolve_index_root_dir() / "multivector_bgem3_chunks.npz"


# dense 검색 경로를 항상 사용합니다.
def is_dense_enabled() -> bool:
    return True


# sparse 검색 경로를 항상 사용합니다.
def is_sparse_enabled() -> bool:
    return True


# multivector 검색 경로를 항상 사용합니다.
def is_multivector_enabled() -> bool:
    return True


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
    return max(1, default_top_k)


# sparse 경로 후보 수는 호출부 기본값을 그대로 사용합니다.
def resolve_sparse_path_top_k(default_top_k: int) -> int:
    return max(1, default_top_k)


# multivector 경로 후보 수는 호출부 기본값을 그대로 사용합니다.
def resolve_multivector_path_top_k(default_top_k: int) -> int:
    return max(1, default_top_k)
