from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

# 이 파일은 프로젝트 전역 설정의 "단일 해석 지점" 역할을 합니다.
# 단순히 env 값을 읽는 데서 끝나지 않고,
# 1) 다양한 표기 입력을 내부 backend 키로 정규화하고
# 2) backend 별 기본 모델 경로를 결정하고
# 3) 검색 경로 조합이 현재 구현 제약에 맞는지 검증합니다.

# 외부 env 값은 표기법이 제각각일 수 있으므로 내부에서 한 번 정규화합니다.
def _normalize_backend_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(".", "_").replace("/", "_")


# 다운로드 스크립트/패치노트에서 보이는 이름을 현재 코드의 내부 backend 키로 맞춥니다.
# 너무 과감하게 구조를 바꾸지 않고, env 입력만 유연하게 받도록 정리한 alias 테이블입니다.
DENSE_BACKEND_ALIASES: dict[str, str] = {
    "bgem3": "bgem3",
    "bge_m3": "bgem3",
    "baai_bge_m3": "bgem3",
    "upskyy_bgem3_ko": "upskyy_bgem3_ko",
    "upskyy_bge_m3_korean": "upskyy_bgem3_ko",
    "e5": "e5",
    "multilingual_e5_large": "e5",
    "intfloat_multilingual_e5_large": "e5",
    "sbert": "sbert",
    "ko_sbert": "sbert",
    "ko_sbert_nli": "sbert",
    "jhgan_ko_sbert_nli": "sbert",
    "jina_v4": "jina_v4",
    "jina_embeddings_v4": "jina_v4",
    "jinaai_jina_embeddings_v4": "jina_v4",
    "gte_multilingual": "gte_multilingual",
    "gte_multilingual_base": "gte_multilingual",
    "alibaba_nlp_gte_multilingual_base": "gte_multilingual",
    "kure_v1": "kure_v1",
    "nlpai_lab_kure_v1": "kure_v1",
    "snowflake_ko": "snowflake_ko",
    "snowflake_arctic_embed_l_v2_0_ko": "snowflake_ko",
    "dragonkue_snowflake_arctic_embed_l_v2_0_ko": "snowflake_ko",
}

SPARSE_BACKEND_ALIASES: dict[str, str] = {
    "bgem3": "bgem3",
    "bge_m3": "bgem3",
    "baai_bge_m3": "bgem3",
    "upskyy_bgem3_ko": "upskyy_bgem3_ko",
    "upskyy_bge_m3_korean": "upskyy_bgem3_ko",
    "bm25": "bm25",
}

MULTIVECTOR_BACKEND_ALIASES: dict[str, str] = {
    "bgem3": "bgem3",
    "bge_m3": "bgem3",
    "baai_bge_m3": "bgem3",
    "upskyy_bgem3_ko": "upskyy_bgem3_ko",
    "upskyy_bge_m3_korean": "upskyy_bgem3_ko",
}

RERANK_BACKEND_ALIASES: dict[str, str] = {
    "bge_reranker": "bge_reranker",
    "bge_reranker_v2_m3": "bge_reranker_v2_m3",
    "baai_bge_reranker_v2_m3": "bge_reranker_v2_m3",
    "bge_reranker_v2_gemma": "bge_reranker_v2_gemma",
    "baai_bge_reranker_v2_gemma": "bge_reranker_v2_gemma",
    "bge_reranker_v2_5_gemma2": "bge_reranker_v2_5_gemma2",
    "bge_reranker_v2_5_gemma2_lightweight": "bge_reranker_v2_5_gemma2",
    "baai_bge_reranker_v2_5_gemma2_lightweight": "bge_reranker_v2_5_gemma2",
    "jina_reranker_v3": "jina_reranker_v3",
    "jinaai_jina_reranker_v3": "jina_reranker_v3",
    "ko_reranker": "ko_reranker",
    "dongjin_kr_ko_reranker": "ko_reranker",
    "minilm_l6": "minilm_l6",
    "minilm_l12": "minilm_l12",
}

LLM_BACKEND_ALIASES: dict[str, str] = {
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen25": "qwen",
    "qwen2_5_7b": "qwen",
    "qwen2_5_7b_instruct": "qwen",
    "qwen_qwen2_5_7b_instruct": "qwen",
    "qwen3": "qwen",
    "qwen3_8b": "qwen",
    "qwen_qwen3_8b": "qwen",
    "qwen3_4b": "qwen",
    "qwen_qwen3_4b": "qwen",
    "exaone": "exaone",
    "exaone_3_5_7_8b_instruct": "exaone",
    "lgai_exaone_exaone_3_5_7_8b_instruct": "exaone",
    "llama3": "llama3",
    "llama_3_open_ko_8b": "llama3",
    "beomi_llama_3_open_ko_8b": "llama3",
    "llama_varco_8b_instruct": "llama3",
    "ncsoft_llama_varco_8b_instruct": "llama3",
}

# LLM은 같은 backend family 안에서도 실제 모델 디렉터리 선택지가 여러 개이므로
# family용 alias와 별도로 "어느 체크포인트를 쓸지"를 보존하는 selection map을 둡니다.
LLM_MODEL_SELECTION_ALIASES: dict[str, str] = {
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen25": "qwen",
    "qwen2_5_7b": "qwen",
    "qwen2_5_7b_instruct": "qwen",
    "qwen_qwen2_5_7b_instruct": "qwen",
    "qwen3": "qwen3_8b",
    "qwen3_8b": "qwen3_8b",
    "qwen_qwen3_8b": "qwen3_8b",
    "qwen3_4b": "qwen3_4b",
    "qwen_qwen3_4b": "qwen3_4b",
    "exaone": "exaone",
    "exaone_3_5_7_8b_instruct": "exaone",
    "lgai_exaone_exaone_3_5_7_8b_instruct": "exaone",
    "llama3": "llama3",
    "llama_3_open_ko_8b": "llama3",
    "beomi_llama_3_open_ko_8b": "llama3",
    "llama_varco_8b_instruct": "llama_varco",
    "ncsoft_llama_varco_8b_instruct": "llama_varco",
}

DENSE_MODEL_DIR_BY_BACKEND: dict[str, Path] = {
    "bgem3": BASE_DIR / "models" / "bge-m3",
    "upskyy_bgem3_ko": BASE_DIR / "models" / "upskyy-bge-m3-korean",
    "e5": BASE_DIR / "models" / "multilingual-e5-large",
    "sbert": BASE_DIR / "models" / "ko-sbert-nli",
    "jina_v4": BASE_DIR / "models" / "jina-embeddings-v4",
    "gte_multilingual": BASE_DIR / "models" / "gte-multilingual-base",
    "kure_v1": BASE_DIR / "models" / "KURE-v1",
    "snowflake_ko": BASE_DIR / "models" / "snowflake-arctic-embed-l-v2.0-ko",
}

SPARSE_MODEL_DIR_BY_BACKEND: dict[str, Path] = {
    "bgem3": BASE_DIR / "models" / "bge-m3",
    "upskyy_bgem3_ko": BASE_DIR / "models" / "upskyy-bge-m3-korean",
    # BM25는 학습된 체크포인트가 필요 없지만 stable model_tag를 위해 고정 경로를 둡니다.
    "bm25": BASE_DIR / "models" / "bm25",
}

MULTIVECTOR_MODEL_DIR_BY_BACKEND: dict[str, Path] = {
    "bgem3": BASE_DIR / "models" / "bge-m3",
    "upskyy_bgem3_ko": BASE_DIR / "models" / "upskyy-bge-m3-korean",
}

RETRIEVAL_BACKEND_CAPABILITIES: dict[str, dict[str, bool]] = {
    "bgem3": {"dense": True, "sparse": True, "multivector": True},
    "upskyy_bgem3_ko": {"dense": True, "sparse": True, "multivector": True},
    "e5": {"dense": True, "sparse": False, "multivector": False},
    "sbert": {"dense": True, "sparse": False, "multivector": False},
    "jina_v4": {"dense": True, "sparse": False, "multivector": False},
    "gte_multilingual": {"dense": True, "sparse": False, "multivector": False},
    "kure_v1": {"dense": True, "sparse": False, "multivector": False},
    "snowflake_ko": {"dense": True, "sparse": False, "multivector": False},
    "bm25": {"dense": False, "sparse": True, "multivector": False},
}

SPARSE_BACKEND_KINDS: dict[str, str] = {
    "bgem3": "learned_sparse",
    "upskyy_bgem3_ko": "learned_sparse",
    "bm25": "lexical_sparse",
}

RERANK_MODEL_DIR_BY_BACKEND: dict[str, Path] = {
    "bge_reranker": BASE_DIR / "models" / "bge-reranker-v2-m3",
    "bge_reranker_v2_m3": BASE_DIR / "models" / "bge-reranker-v2-m3",
    "bge_reranker_v2_gemma": BASE_DIR / "models" / "bge-reranker-v2-gemma",
    "bge_reranker_v2_5_gemma2": BASE_DIR / "models" / "bge-reranker-v2.5-gemma2-lightweight",
    "jina_reranker_v3": BASE_DIR / "models" / "jina-reranker-v3",
    "minilm_l6": BASE_DIR / "models" / "ms-marco-MiniLM-L-6-v2",
    "minilm_l12": BASE_DIR / "models" / "ms-marco-MiniLM-L-12-v2",
    "ko_reranker": BASE_DIR / "models" / "ko-reranker",
}

LLM_MODEL_DIR_BY_BACKEND: dict[str, Path] = {
    # qwen 계열 기본값은 다운로드 스크립트에 맞춰 Qwen2.5-7B-Instruct로 둡니다.
    "qwen": BASE_DIR / "models" / "qwen2.5-7b-instruct",
    "qwen3_8b": BASE_DIR / "models" / "qwen3-8b",
    "qwen3_4b": BASE_DIR / "models" / "qwen3-4b",
    "exaone": BASE_DIR / "models" / "EXAONE-3.5-7.8B-Instruct",
    # llama 계열 기본값은 현재 스크립트에 포함된 한국어 계열 후보로 둡니다.
    "llama3": BASE_DIR / "models" / "Llama-3-Open-Ko-8B",
    "llama_varco": BASE_DIR / "models" / "Llama-VARCO-8B-Instruct",
}


# env 파싱 유틸리티입니다.
# 개별 모듈이 직접 os.getenv 를 호출하지 않게 해서 파싱 규칙을 한곳에 모읍니다.
def _get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _get_env_bool(name: str, default: bool) -> bool:
    raw = _get_env_str(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def _get_env_int(name: str, default: int) -> int:
    return int(_get_env_str(name, str(default)))


def _get_env_float(name: str, default: float) -> float:
    return float(_get_env_str(name, str(default)))


def _resolve_model_dir(
    env_name: str,
    backend_name: str,
    default_map: dict[str, Path],
    fallback: Path,
) -> Path:
    # 특정 경로 env 가 있으면 그것을 최우선으로 쓰고,
    # 없으면 backend 별 기본 경로 테이블에서 찾아 반환합니다.
    custom = os.getenv(env_name)
    if custom:
        return Path(custom)
    return default_map.get(backend_name, fallback)


def _resolve_backend_alias(env_name: str, default: str, alias_map: dict[str, str]) -> str:
    # env 입력은 실제 HF repo 이름, 로컬 폴더명, 축약형 등 다양할 수 있으므로
    # 내부 비교/분기에는 항상 canonical backend 키를 사용합니다.
    raw = _get_env_str(env_name, default)
    normalized = _normalize_backend_name(raw)
    return alias_map.get(normalized, normalized)


def _resolve_llm_model_selection() -> str:
    # LLM은 backend family(qwen/exaone/llama3)와 실제 체크포인트 선택을 분리합니다.
    # 예를 들어 FAIRDATA_LLM_BACKEND=qwen3-8b 이면 family는 qwen 이지만
    # 실제 모델 디렉터리는 qwen3_8b 쪽을 선택해야 하므로 별도 selection map을 씁니다.
    raw = _get_env_str("FAIRDATA_LLM_BACKEND", "qwen")
    normalized = _normalize_backend_name(raw)
    return LLM_MODEL_SELECTION_ALIASES.get(normalized, resolve_llm_backend_name())


# 원천 코퍼스 디렉터리를 반환합니다.
def resolve_data_dir() -> Path:
    return BASE_DIR / "data" / "raw"


# 검색 경로 공통 기본 BGE-M3 모델 디렉터리를 반환합니다.
def resolve_bgem3_model_dir() -> Path:
    return BASE_DIR / "models" / "bge-m3"


# dense 전용 모델 디렉터리를 반환합니다.
def resolve_dense_model_dir() -> Path:
    return _resolve_model_dir(
        "FAIRDATA_DENSE_MODEL_DIR",
        resolve_dense_backend_name(),
        DENSE_MODEL_DIR_BY_BACKEND,
        resolve_bgem3_model_dir(),
    )


# sparse 전용 모델 디렉터리를 반환합니다.
def resolve_sparse_model_dir() -> Path:
    return _resolve_model_dir(
        "FAIRDATA_SPARSE_MODEL_DIR",
        resolve_sparse_backend_name(),
        SPARSE_MODEL_DIR_BY_BACKEND,
        resolve_bgem3_model_dir(),
    )


# multivector 전용 모델 디렉터리를 반환합니다.
def resolve_multivector_model_dir() -> Path:
    return _resolve_model_dir(
        "FAIRDATA_MULTIVECTOR_MODEL_DIR",
        resolve_multivector_backend_name(),
        MULTIVECTOR_MODEL_DIR_BY_BACKEND,
        resolve_bgem3_model_dir(),
    )


# reranker 모델 디렉터리를 반환합니다.
def resolve_bge_reranker_model_dir() -> Path:
    return _resolve_model_dir(
        "FAIRDATA_RERANK_MODEL_DIR",
        resolve_reranker_backend_name(),
        RERANK_MODEL_DIR_BY_BACKEND,
        BASE_DIR / "models" / "bge-reranker-v2-m3",
    )


# reranker backend 종류를 반환합니다.
def resolve_reranker_backend_name() -> str:
    # 예: bge-reranker-v2.5-gemma2-lightweight -> bge_reranker_v2_5_gemma2
    return _resolve_backend_alias("FAIRDATA_RERANK_BACKEND", "bge_reranker", RERANK_BACKEND_ALIASES)


# reranker top_n을 반환합니다.
def resolve_reranker_top_n() -> int:
    return _get_env_int("FAIRDATA_RERANK_TOP_N", 50)


# reranker fusion weight를 반환합니다.
def resolve_reranker_weight() -> float:
    return _get_env_float("FAIRDATA_RERANK_WEIGHT", 1.0)


# 생성 모델 디렉터리를 반환합니다.
def resolve_qwen_model_dir() -> Path:
    return BASE_DIR / "models" / "qwen2.5-7b-instruct"


# LLM 모델 디렉터리를 반환합니다 (env 오버라이드 지원).
def resolve_llm_model_dir() -> Path:
    custom = os.getenv("FAIRDATA_LLM_MODEL_DIR")
    if custom:
        return Path(custom)
    return LLM_MODEL_DIR_BY_BACKEND.get(_resolve_llm_model_selection(), resolve_qwen_model_dir())


# LLM backend 종류를 반환합니다.
def resolve_llm_backend_name() -> str:
    # 예: qwen3-8b -> qwen, EXAONE-3.5-7.8B-Instruct -> exaone
    return _resolve_backend_alias("FAIRDATA_LLM_BACKEND", "qwen", LLM_BACKEND_ALIASES)


# 검색 인덱스 루트 디렉터리를 반환합니다.
def resolve_index_root_dir() -> Path:
    return BASE_DIR / "index"


# 인덱스 namespace를 반환합니다 (실험별 인덱스 격리).
def resolve_index_namespace() -> str:
    # 어떤 backend 조합으로 만든 인덱스인지 namespace 에 반영해
    # 실험별 인덱스 파일/디렉터리가 서로 덮어쓰지 않게 합니다.
    default_parts = [
        f"dense_{resolve_dense_backend_name()}" if is_dense_enabled() else "dense_off",
        f"sparse_{resolve_sparse_backend_name()}" if is_sparse_enabled() else "sparse_off",
        (
            f"multivector_{resolve_multivector_backend_name()}"
            if is_multivector_enabled()
            else "multivector_off"
        ),
    ]
    return _get_env_str("FAIRDATA_INDEX_NAMESPACE", "__".join(default_parts))


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
    return _get_env_bool("FAIRDATA_ENABLE_DENSE", True)


# sparse 검색 경로 사용 여부를 반환합니다.
def is_sparse_enabled() -> bool:
    return _get_env_bool("FAIRDATA_ENABLE_SPARSE", True)


# multivector 검색 경로 사용 여부를 반환합니다.
def is_multivector_enabled() -> bool:
    return _get_env_bool("FAIRDATA_ENABLE_MULTIVECTOR", True)


# dense 경로가 사용할 백엔드 종류를 반환합니다.
def resolve_dense_backend_name() -> str:
    # 예: multilingual-e5-large -> e5, jina-embeddings-v4 -> jina_v4
    return _resolve_backend_alias("FAIRDATA_DENSE_BACKEND", "bgem3", DENSE_BACKEND_ALIASES)


# sparse 경로가 사용할 백엔드 종류를 반환합니다.
def resolve_sparse_backend_name() -> str:
    return _resolve_backend_alias("FAIRDATA_SPARSE_BACKEND", "bgem3", SPARSE_BACKEND_ALIASES)


# multivector 경로가 사용할 백엔드 종류를 반환합니다.
def resolve_multivector_backend_name() -> str:
    return _resolve_backend_alias("FAIRDATA_MULTIVECTOR_BACKEND", "bgem3", MULTIVECTOR_BACKEND_ALIASES)


# 융합 단계에서 RRF를 사용할지 여부를 반환합니다.
def use_rrf_fusion() -> bool:
    return _get_env_bool("FAIRDATA_USE_RRF_FUSION", True)


# dense 경로 후보 수는 호출부 기본값을 그대로 사용합니다.
def resolve_dense_path_top_k(default_top_k: int) -> int:
    custom = _get_env_int("FAIRDATA_DENSE_TOP_K", 0)
    if custom > 0:
        return custom
    return max(1, default_top_k)


# sparse 경로 후보 수는 호출부 기본값을 그대로 사용합니다.
def resolve_sparse_path_top_k(default_top_k: int) -> int:
    custom = _get_env_int("FAIRDATA_SPARSE_TOP_K", 0)
    if custom > 0:
        return custom
    return max(1, default_top_k)


# multivector 경로 후보 수는 호출부 기본값을 그대로 사용합니다.
def resolve_multivector_path_top_k(default_top_k: int) -> int:
    custom = _get_env_int("FAIRDATA_MULTIVECTOR_TOP_K", 0)
    if custom > 0:
        return custom
    return max(1, default_top_k)


def get_backend_capabilities(backend_name: str) -> dict[str, bool]:
    # 상위 파이프라인이 "이 backend 가 dense/sparse/multivector 중 무엇을 지원하는가"를
    # 빠르게 조회할 수 있도록 capability 테이블을 노출합니다.
    normalized = _normalize_backend_name(backend_name)
    return RETRIEVAL_BACKEND_CAPABILITIES.get(
        normalized,
        {"dense": False, "sparse": False, "multivector": False},
    )


def resolve_sparse_backend_kind() -> str:
    # sparse 는 BM25 같은 lexical 계열과 BGE-M3 같은 learned sparse 계열이 다르므로
    # fusion/profile/검증 로직에서 사용할 추상 분류를 제공합니다.
    return SPARSE_BACKEND_KINDS.get(resolve_sparse_backend_name(), "unknown_sparse")


def is_lexical_sparse_backend() -> bool:
    return resolve_sparse_backend_kind() == "lexical_sparse"


def is_learned_sparse_backend() -> bool:
    return resolve_sparse_backend_kind() == "learned_sparse"


def resolve_retrieval_profile() -> str:
    # 현재 enable 플래그와 backend 종류를 바탕으로
    # 실험/로그/메트릭에 남길 retrieval 조합 이름을 계산합니다.
    dense_enabled = is_dense_enabled()
    sparse_enabled = is_sparse_enabled()
    multivector_enabled = is_multivector_enabled()
    sparse_backend = resolve_sparse_backend_name()
    multivector_backend = resolve_multivector_backend_name()

    if dense_enabled and sparse_enabled and multivector_enabled:
        if sparse_backend in {"bgem3", "upskyy_bgem3_ko"} and multivector_backend in {"bgem3", "upskyy_bgem3_ko"}:
            return "full_hybrid"
        return "dense_sparse_multivector"
    if dense_enabled and sparse_enabled and not multivector_enabled:
        if is_lexical_sparse_backend():
            return "dense_lexical_hybrid"
        if is_learned_sparse_backend():
            return "dense_learned_sparse_hybrid"
        return "dense_sparse"
    if dense_enabled and not sparse_enabled and not multivector_enabled:
        return "dense_only"
    if not dense_enabled and sparse_enabled and not multivector_enabled:
        return "sparse_only"
    if dense_enabled and not sparse_enabled and multivector_enabled:
        return "dense_multivector"
    return "custom"


def validate_retrieval_configuration() -> None:
    # 현재 구현이 지원하지 않는 backend 조합은 검색 시작 전에 즉시 차단합니다.
    # 특히 learned sparse 와 multivector 는 동일 family 공유를 전제로 구현되어 있어
    # 설정은 가능해 보여도 런타임에서 깨질 조합을 여기서 미리 막습니다.
    dense_enabled = is_dense_enabled()
    sparse_enabled = is_sparse_enabled()
    multivector_enabled = is_multivector_enabled()

    if not any([dense_enabled, sparse_enabled, multivector_enabled]):
        raise ValueError("At least one retrieval path must be enabled.")

    dense_backend = resolve_dense_backend_name()
    sparse_backend = resolve_sparse_backend_name()
    multivector_backend = resolve_multivector_backend_name()

    if dense_enabled and not get_backend_capabilities(dense_backend)["dense"]:
        raise ValueError(f"Dense backend '{dense_backend}' does not support dense retrieval.")
    if sparse_enabled and not get_backend_capabilities(sparse_backend)["sparse"]:
        raise ValueError(f"Sparse backend '{sparse_backend}' does not support sparse retrieval.")
    if multivector_enabled and not get_backend_capabilities(multivector_backend)["multivector"]:
        raise ValueError(
            f"Multi-vector backend '{multivector_backend}' does not support multi-vector retrieval."
        )

    # learned sparse / multivector 계열은 같은 family를 공유하는 조합만 허용합니다.
    if multivector_enabled and not dense_enabled:
        raise ValueError("Multi-vector retrieval requires a dense backend to be enabled.")

    if multivector_enabled and dense_backend != multivector_backend:
        raise ValueError(
            "Dense backend and multi-vector backend must match for the current implementation. "
            f"Got dense='{dense_backend}', multivector='{multivector_backend}'."
        )

    if sparse_enabled and is_learned_sparse_backend() and dense_enabled:
        if dense_backend != sparse_backend:
            raise ValueError(
                "Learned sparse retrieval currently requires the same dense backend family. "
                f"Got dense='{dense_backend}', sparse='{sparse_backend}'."
            )

    if multivector_enabled and sparse_enabled and sparse_backend != multivector_backend:
        raise ValueError(
            "When sparse and multi-vector are both enabled, they must use the same learned backend family. "
            f"Got sparse='{sparse_backend}', multivector='{multivector_backend}'."
        )
