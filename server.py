from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from app.corpus import load_corpus
from app.generator import GroundedGenerator
from app.retriever import HybridRetriever
from app.schemas import PredictRequest, PredictResponse


APP_TITLE = "꽃보다 의결 FairData Submission"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FAIRDATA_RAW_DIR", BASE_DIR / "raw"))

app = FastAPI(title=APP_TITLE)
corpus = load_corpus(DATA_DIR)
retriever = HybridRetriever(corpus)
generator = GroundedGenerator()


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "corpus_chunks": len(corpus)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    chunks = retriever.search(req.question, top_k=5)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != 5 or len(set(chunk_ids)) != 5:
        raise RuntimeError("Retriever must return exactly 5 unique chunk_ids")

    answer = generator.generate(req.question, chunks)
    return PredictResponse(
        id=req.id,
        retrieved_chunk_ids=chunk_ids,
        answer=answer,
    )
