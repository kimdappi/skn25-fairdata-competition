from __future__ import annotations

from pathlib import Path

from app.preprocessing.corpus import CorpusStore
from app.rerank.interfaces import RerankerBackend
from app.rerank.reranker import (
    BGERerankerV2M3,
    BGERerankerV25Gemma2,
    JinaRerankerV3,
    KoReranker,
    MiniLMReranker,
)


def _normalize_backend_name(backend_name: str) -> str:
    return backend_name.strip().lower().replace("-", "_").replace(".", "_")


def build_reranker_backend(
    backend_name: str,
    corpus_store: CorpusStore,
    model_dir: Path,
) -> RerankerBackend:
    alias = _normalize_backend_name(backend_name)
    if alias in {"bge_reranker", "bge_reranker_v2_m3"}:
        return BGERerankerV2M3(corpus_store, model_dir)
    if alias in {"bge_reranker_v2_gemma", "bge_reranker_v2_5_gemma2"}:
        return BGERerankerV25Gemma2(corpus_store, model_dir)
    if alias == "jina_reranker_v3":
        return JinaRerankerV3(corpus_store, model_dir)
    if alias in {"minilm_l6", "minilm_l12"}:
        return MiniLMReranker(corpus_store, model_dir)
    if alias == "ko_reranker":
        return KoReranker(corpus_store, model_dir)
    supported = [
        "bge_reranker",
        "bge_reranker_v2_gemma",
        "bge_reranker_v2_5_gemma2",
        "jina_reranker_v3",
        "minilm_l6",
        "minilm_l12",
        "ko_reranker",
    ]
    raise ValueError(f"Unsupported reranker backend: {backend_name}. Supported: {', '.join(supported)}")
