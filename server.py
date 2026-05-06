import os
from functools import lru_cache

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("STAGE_LOG_ENABLED", "false")
os.environ.setdefault("QUERY_TIMING_LOG_ENABLED", "false")

from fastapi import FastAPI
from pydantic import BaseModel

from baseline_rag.config import DEFAULT_EMBEDDING_MODEL
from baseline_rag.predictor import RAGPredictor

app = FastAPI(title="FairCompetition RAG Submission")


class PredictRequest(BaseModel):
    id: str
    question: str


class PredictResponse(BaseModel):
    id: str
    retrieved_chunk_ids: list[str]
    answer: str


@lru_cache(maxsize=1)
def get_predictor() -> RAGPredictor:
    model_name = os.getenv("FAIRCOMP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return RAGPredictor(embedding_model_name=model_name)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    result = get_predictor().predict(request.question)
    return PredictResponse(
        id=request.id,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        answer=result.answer,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
