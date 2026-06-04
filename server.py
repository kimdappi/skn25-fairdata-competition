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
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    chunks = retriever.search(req.question, top_k=5)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    answer = generator.generate(req.question, chunks)
    return PredictResponse(id=req.id, retrieved_chunk_ids=chunk_ids, answer=answer)
