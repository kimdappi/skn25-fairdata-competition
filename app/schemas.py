from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    id: str = Field(..., description="Evaluation request identifier")
    question: str = Field(..., description="Natural language question")


class PredictResponse(BaseModel):
    id: str
    retrieved_chunk_ids: List[str]
    answer: str
