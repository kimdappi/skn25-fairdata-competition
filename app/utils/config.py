from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


# 환경변수와 기본 후보 경로를 기준으로 사용할 데이터 디렉터리를 결정합니다.
def resolve_data_dir() -> Path:
    candidates = [
        os.getenv("FAIRDATA_RAW_DIR", "").strip(),
        os.getenv("FAIRDATA_DATA_DIR", "").strip(),
        str(BASE_DIR / "data" / "raw"),
        str(BASE_DIR / "raw"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return Path(BASE_DIR / "data" / "raw")


# 환경변수와 기본 후보 경로를 기준으로 사용할 BGE-M3 모델 디렉터리를 결정합니다.
def resolve_bgem3_model_dir() -> Path:
    candidates = [
        os.getenv("FAIRDATA_BGEM3_MODEL_DIR", "").strip(),
        os.getenv("FAIRDATA_MODEL_DIR", "").strip(),
        str(BASE_DIR / "models" / "bge-m3"),
        str(BASE_DIR / "models" / "embedding_bge_m3"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return Path(BASE_DIR / "models" / "bge-m3")


# 환경변수와 기본 후보 경로를 기준으로 사용할 BGE reranker 모델 디렉터리를 결정합니다.
def resolve_bge_reranker_model_dir() -> Path:
    candidates = [
        os.getenv("FAIRDATA_BGE_RERANKER_MODEL_DIR", "").strip(),
        os.getenv("FAIRDATA_RERANKER_MODEL_DIR", "").strip(),
        str(BASE_DIR / "models" / "bge-reranker-v2-m3"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return Path(BASE_DIR / "models" / "bge-reranker-v2-m3")


# 환경변수와 기본 후보 경로를 기준으로 사용할 Qwen 생성 모델 디렉터리를 결정합니다.
def resolve_qwen_model_dir() -> Path:
    candidates = [
        os.getenv("FAIRDATA_QWEN_MODEL_DIR", "").strip(),
        os.getenv("FAIRDATA_GENERATION_MODEL_DIR", "").strip(),
        str(BASE_DIR / "models" / "qwen2-7b-instruct"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return Path(BASE_DIR / "models" / "qwen2-7b-instruct")


# 환경변수와 기본 후보 경로를 기준으로 사용할 Chroma 인덱스 디렉터리를 결정합니다.
def resolve_index_root_dir() -> Path:
    env_candidates = [
        os.getenv("FAIRDATA_INDEX_ROOT_DIR", "").strip(),
        os.getenv("FAIRDATA_INDEX_DIR", "").strip(),
    ]
    for candidate in env_candidates:
        if candidate:
            return Path(candidate)
    return Path(BASE_DIR / "index")


# 환경변수와 기본 후보 경로를 기준으로 사용할 Chroma 인덱스 디렉터리를 결정합니다.
def resolve_chroma_index_dir() -> Path:
    explicit_path = os.getenv("FAIRDATA_CHROMA_DIR", "").strip()
    if explicit_path:
        return Path(explicit_path)
    return resolve_index_root_dir() / "chroma_bgem3"


def resolve_chroma_manifest_path() -> Path:
    return resolve_index_root_dir() / "chroma_bgem3_manifest.json"


def resolve_sparse_index_path() -> Path:
    return resolve_index_root_dir() / "sparse_bgem3_chunks.npz"


def resolve_sparse_manifest_path() -> Path:
    return resolve_index_root_dir() / "sparse_bgem3_chunks_manifest.json"


def resolve_multivector_index_path() -> Path:
    return resolve_index_root_dir() / "multivector_bgem3_chunks.npz"


def resolve_multivector_manifest_path() -> Path:
    return resolve_index_root_dir() / "multivector_bgem3_chunks_manifest.json"
