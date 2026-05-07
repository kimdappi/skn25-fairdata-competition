import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_COLAB_PROJECT_DIR = Path("/content/drive/MyDrive/skn폴더/FairCompetition")
DEFAULT_COLAB_MODEL_DIR = DEFAULT_COLAB_PROJECT_DIR / "models"


def _candidate_project_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_project_dir = os.getenv("FAIRCOMP_PROJECT_DIR", "").strip()
    env_model_dir = os.getenv("FAIRCOMP_MODEL_DIR", "").strip()
    if env_project_dir:
        candidates.append(Path(env_project_dir).expanduser())
    if env_model_dir:
        candidates.append(Path(env_model_dir).expanduser().parent)
    candidates.extend([DEFAULT_COLAB_PROJECT_DIR, BASE_DIR])

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _optional_env_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def _resolve_project_dir() -> Path:
    return _first_existing(_candidate_project_dirs()) or DEFAULT_COLAB_PROJECT_DIR


PROJECT_DIR = _resolve_project_dir()
MODEL_DIR = _first_existing(
    [path for path in [_optional_env_path("FAIRCOMP_MODEL_DIR"), PROJECT_DIR / "models", DEFAULT_COLAB_MODEL_DIR, BASE_DIR / "models"] if path is not None]
) or DEFAULT_COLAB_MODEL_DIR
DATA_DIR = _first_existing(
    [path for path in [_optional_env_path("FAIRCOMP_DATA_DIR"), PROJECT_DIR / "search_raw_data", BASE_DIR / "search_raw_data"] if path is not None]
) or (PROJECT_DIR / "search_raw_data")
METRICS_PATH = _first_existing(
    [path for path in [_optional_env_path("FAIRCOMP_METRICS_PATH"), PROJECT_DIR / "metrics.json", BASE_DIR / "metrics.json"] if path is not None]
) or (PROJECT_DIR / "metrics.json")
ARTIFACT_DIR = _optional_env_path("FAIRCOMP_ARTIFACT_DIR") or (PROJECT_DIR / "artifacts")
RESULT_DIR = _optional_env_path("FAIRCOMP_RESULT_DIR") or (PROJECT_DIR / "results")
EMBEDDING_CACHE_DIR = ARTIFACT_DIR / "embedding_cache"

DEFAULT_EMBEDDING_MODEL = "embedding_ko_legal_sbert"
EVALUATION_MODEL_NAMES = [DEFAULT_EMBEDDING_MODEL]

TOP_K_DOCS = 3
TOP_K_CHUNKS = 5
DOC_FAISS_K = 80
CHUNK_FAISS_K = 120
DOC_SVD_COMPONENTS = 64
CHUNK_SVD_COMPONENTS = 0
DOC_WORD_MAX_FEATURES = 12000
DOC_CHAR_MAX_FEATURES = 16000
CHUNK_WORD_MAX_FEATURES = 15000
CHUNK_CHAR_MAX_FEATURES = 20000

RRF_K = 60
DOC_RRF_BM25_WEIGHT = 1.0
DOC_RRF_DENSE_WEIGHT = 0.8
CHUNK_RRF_BM25_WEIGHT = 1.0
CHUNK_RRF_DENSE_WEIGHT = 0.75

BGE_M3_USE_SPARSE = True
BGE_M3_USE_COLBERT = True
BGE_M3_EAGER_SPARSE_CACHE_BUILD = False
BGE_M3_DOC_RRF_SPARSE_WEIGHT = 0.95
BGE_M3_CHUNK_RRF_SPARSE_WEIGHT = 1.05
BGE_M3_DOC_COLBERT_RERANK_K = 12
BGE_M3_CHUNK_COLBERT_RERANK_K = 48
BGE_M3_DOC_COLBERT_SCORE_WEIGHT = 0.08
BGE_M3_CHUNK_COLBERT_SCORE_WEIGHT = 0.12

OLLAMA_ROUTER_ENABLED = True
OLLAMA_ROUTER_MODEL = "Qwen2.5-7B-Instruct"
OLLAMA_ROUTER_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_ROUTER_TIMEOUT_SECONDS = 120
ROUTER_BACKEND = "hf"
HF_ROUTER_MODEL = "Qwen2.5-7B-Instruct"
HF_ROUTER_LOCAL_FILES_ONLY = True
ROUTER_LOG_ENABLED = True
STAGE_LOG_ENABLED = True
QUERY_TIMING_LOG_ENABLED = True


def running_in_colab() -> bool:
    return os.getenv("COLAB_GPU", "").strip() != "" or Path("/content").exists()


REQUIRE_CUDA = os.getenv("FAIRCOMP_REQUIRE_CUDA", "true" if running_in_colab() else "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def validate_runtime_paths() -> None:
    missing = []
    if not MODEL_DIR.exists():
        missing.append(f"MODEL_DIR={MODEL_DIR}")
    if not DATA_DIR.exists():
        missing.append(f"DATA_DIR={DATA_DIR}")
    require_metrics_path = os.getenv("FAIRCOMP_REQUIRE_METRICS_PATH", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if require_metrics_path and not METRICS_PATH.exists():
        missing.append(f"METRICS_PATH={METRICS_PATH}")
    if missing:
        raise FileNotFoundError(
            "Required runtime paths are missing. "
            + " | ".join(missing)
            + " | Set FAIRCOMP_PROJECT_DIR / FAIRCOMP_MODEL_DIR / FAIRCOMP_DATA_DIR / FAIRCOMP_METRICS_PATH explicitly if needed."
        )
