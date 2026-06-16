import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from app.generation.generator import build_llm_backend
from app.preprocessing.corpus import load_corpus
from app.retrieval.retriever import HybridRetriever
from app.retrieval.router import QueryRouter
from app.utils.config import (
    resolve_data_dir,
    resolve_dense_backend_name,
    resolve_multivector_backend_name,
    resolve_retrieval_profile,
    resolve_sparse_backend_name,
    resolve_sparse_backend_kind,
    validate_retrieval_configuration,
)


class PredictRequest(BaseModel):
    id: str
    question: str


class PredictResponse(BaseModel):
    id: str
    retrieved_chunk_ids: List[str]
    answer: str


APP_TITLE = "꽃보다 의결 FairData Submission"
DATA_DIR = resolve_data_dir()

validate_retrieval_configuration()

app = FastAPI(title=APP_TITLE)
router = QueryRouter()
corpus = load_corpus(DATA_DIR, router.route_from_text)
retriever = HybridRetriever(corpus, router)
generator = build_llm_backend()


@app.on_event("startup")
def warmup_models() -> None:
    """필요 시 startup eager warmup을 수행합니다.

    FAIRDATA_SKIP_LLM_WARMUP=1 이면 startup block을 피하고,
    첫 요청 시 lazy load 하도록 넘깁니다.
    """
    if os.getenv("FAIRDATA_SKIP_LLM_WARMUP", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return
    retriever.pipeline.ensure_runtime()
    if retriever.pipeline.reranker is not None:
        retriever.pipeline.reranker.ensure_runtime()
    generator.ensure_runtime()


@app.get("/health")
def health() -> dict[str, str | int | bool]:
    return {
        "status": "ok",
        "corpus_chunks": len(corpus.chunks),
        "data_dir": str(DATA_DIR),
        "retrieval_profile": resolve_retrieval_profile(),
        "dense_backend": resolve_dense_backend_name(),
        "sparse_backend": resolve_sparse_backend_name(),
        "sparse_backend_kind": resolve_sparse_backend_kind(),
        "multivector_backend": resolve_multivector_backend_name(),
        "models_loaded": (
            retriever.pipeline.reranker is not None
            and retriever.pipeline.reranker.model is not None
            and generator.model is not None
        ),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    chunks = retriever.search(req.question, top_k=5, query_id=req.id)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    answer = generator.generate(req.question, chunks)
    return PredictResponse(id=req.id, retrieved_chunk_ids=chunk_ids, answer=answer)
