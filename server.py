from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from app.generation.generator import build_llm_backend
from app.preprocessing.corpus import load_corpus
from app.retrieval.retriever import HybridRetriever
from app.retrieval.router import QueryRouter
from app.retrieval.route_tags import OnlineLLMQuestionRouter, RouteTagStore
from app.utils.config import (
    resolve_data_dir,
    resolve_dense_backend_name,
    resolve_multivector_backend_name,
    resolve_route_tags_path,
    resolve_retrieval_profile,
    resolve_sparse_backend_name,
    resolve_sparse_backend_kind,
    validate_selected_model_directories,
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
validate_selected_model_directories()

app = FastAPI(title=APP_TITLE)
fallback_router = QueryRouter()
route_tag_store = RouteTagStore.load(resolve_route_tags_path())
generator = build_llm_backend()
router = OnlineLLMQuestionRouter(fallback_router, generator.route_question)
corpus = load_corpus(
    DATA_DIR,
    fallback_router.route_from_text,
    route_document_fn=lambda doc_id, text: route_tag_store.route_document(
        doc_id,
        text,
        fallback_router.route_from_text,
    ),
)
retriever = HybridRetriever(corpus, router)


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
        "route_tags_loaded": bool(route_tag_store.documents or route_tag_store.questions_by_key),
        "question_routing": "online_llm",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    chunks = retriever.search(req.question, top_k=5)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    answer = generator.generate(req.question, chunks)
    return PredictResponse(id=req.id, retrieved_chunk_ids=chunk_ids, answer=answer)

@app.on_event("startup")
def warmup_models() -> None:
    retriever.pipeline.ensure_runtime()          # ① BGE-M3 (dense/sparse/multivector 공통)
    if retriever.pipeline.reranker is not None:
        retriever.pipeline.reranker.ensure_runtime()  # ② Reranker
    generator.ensure_runtime()                   # ③ LLM
