from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.preprocessing.corpus import load_corpus
from app.retrieval.backends import (
    build_dense_backend,
    build_multivector_backend,
    build_sparse_backend,
)
from app.retrieval.engines import DenseSearchEngine, MultiVectorSearchEngine, SparseSearchEngine
from app.retrieval.router import QueryRouter
from app.utils.config import (
    is_dense_enabled,
    is_multivector_enabled,
    is_sparse_enabled,
    resolve_data_dir,
    resolve_dense_backend_name,
    resolve_dense_model_dir,
    resolve_index_root_dir,
    resolve_multivector_backend_name,
    resolve_multivector_model_dir,
    resolve_sparse_backend_name,
    resolve_sparse_model_dir,
)


def main() -> None:
    data_dir = resolve_data_dir()
    index_dir = resolve_index_root_dir()

    print(f"[build_indexes] data_dir={data_dir}")
    print(f"[build_indexes] index_dir={index_dir}")

    router = QueryRouter()
    corpus = load_corpus(data_dir, router.route_from_text)
    print(
        f"[build_indexes] loaded corpus: documents={len(corpus.documents)} chunks={len(corpus.chunks)}"
    )

    runtime_cache: dict[tuple[str, Path], object] = {}

    if is_dense_enabled():
        dense_backend = build_dense_backend(
            resolve_dense_backend_name(),
            resolve_dense_model_dir(),
            runtime_cache,
        )
        print(
            f"[build_indexes] dense backend={resolve_dense_backend_name()} "
            f"model_dir={dense_backend.model_dir}"
        )
        DenseSearchEngine(corpus, dense_backend)
        print("[build_indexes] dense index ready")

    if is_sparse_enabled():
        sparse_backend = build_sparse_backend(
            resolve_sparse_backend_name(),
            resolve_sparse_model_dir(),
            runtime_cache,
        )
        print(
            f"[build_indexes] sparse backend={resolve_sparse_backend_name()} "
            f"model_dir={sparse_backend.model_dir}"
        )
        SparseSearchEngine(corpus, sparse_backend)
        print("[build_indexes] sparse index ready")

    if is_multivector_enabled():
        multivector_backend = build_multivector_backend(
            resolve_multivector_backend_name(),
            resolve_multivector_model_dir(),
            runtime_cache,
        )
        print(
            f"[build_indexes] multivector backend={resolve_multivector_backend_name()} "
            f"model_dir={multivector_backend.model_dir}"
        )
        MultiVectorSearchEngine(corpus, multivector_backend)
        print("[build_indexes] multivector index ready")

    print("[build_indexes] completed")


if __name__ == "__main__":
    main()
