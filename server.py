from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
 
app = FastAPI()
 


from app.generation.generator import GroundedGenerator
from app.preprocessing.corpus import load_corpus
from app.retrieval.router import QueryRouter
from app.retrieval.retriever import HybridRetriever
from app.utils.config import resolve_data_dir
from pydantic import BaseModel

class PredictRequest(BaseModel):
    id: str
    question: str
 
class PredictResponse(BaseModel):
    id: str
    retrieved_chunk_ids: List[str]
    answer: str

    
APP_TITLE = "꽃보다 의결 FairData Submission"
DATA_DIR = resolve_data_dir()

app = FastAPI(title=APP_TITLE)
router = QueryRouter()
corpus = load_corpus(DATA_DIR, router.route_from_text)
retriever = HybridRetriever(corpus, router)
generator = GroundedGenerator()


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "corpus_chunks": len(corpus.chunks), "data_dir": str(DATA_DIR)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    chunks = retriever.search(req.question, top_k=5)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    answer = generator.generate(req.question, chunks)
    return PredictResponse(
        id=req.id,
        retrieved_chunk_ids=chunk_ids,
        answer=answer,
    )
