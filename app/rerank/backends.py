from __future__ import annotations

from pathlib import Path

from app.preprocessing.corpus import CorpusStore
from app.rerank.interfaces import RerankerBackend
from app.rerank.reranker import BGERerankerV2M3


def build_reranker_backend(
    backend_name: str,
    corpus_store: CorpusStore,
    model_dir: Path,
) -> RerankerBackend:
    if backend_name == "bge_reranker":
        return BGERerankerV2M3(corpus_store, model_dir)
    raise ValueError(f"Unsupported reranker backend: {backend_name}")
