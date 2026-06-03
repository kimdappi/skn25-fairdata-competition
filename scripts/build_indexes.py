from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.preprocessing.corpus import load_corpus
from app.retrieval.bgem3 import BGEM3HybridModel
from app.retrieval.engines import DenseSearchEngine, MultiVectorSearchEngine, SparseSearchEngine
from app.retrieval.router import QueryRouter
from app.utils.config import resolve_bgem3_model_dir, resolve_data_dir, resolve_index_root_dir


def main() -> None:
    data_dir = resolve_data_dir()
    model_dir = resolve_bgem3_model_dir()
    index_dir = resolve_index_root_dir()

    print(f"[build_indexes] data_dir={data_dir}")
    print(f"[build_indexes] model_dir={model_dir}")
    print(f"[build_indexes] index_dir={index_dir}")

    router = QueryRouter()
    corpus = load_corpus(data_dir, router.route_from_text)
    print(
        f"[build_indexes] loaded corpus: documents={len(corpus.documents)} chunks={len(corpus.chunks)}"
    )

    model = BGEM3HybridModel(model_dir)
    print("[build_indexes] loaded BGE-M3 model")

    DenseSearchEngine(corpus, model)
    print("[build_indexes] dense index ready")

    SparseSearchEngine(corpus, model)
    print("[build_indexes] sparse index ready")

    MultiVectorSearchEngine(corpus, model)
    print("[build_indexes] multivector index ready")

    print("[build_indexes] completed")


if __name__ == "__main__":
    main()
