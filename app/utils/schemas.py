from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    id: str = Field(..., description="평가 요청 식별자")
    question: str = Field(..., description="공정위 의결서 질의문")


class PredictResponse(BaseModel):
    id: str
    retrieved_chunk_ids: List[str]
    answer: str


class RouteDecision(BaseModel):
    theme: str = Field(description="질문의 핵심 위반 분야")
    company_size: str = Field(description="질문에 드러난 기업 규모")
    legal_role: str = Field(description="질문에 드러난 법적 지위")
    industry: str = Field(description="질문에 드러난 업종")
    focus: str = Field(description="질문 초점")
    keywords: List[str] = Field(description="질문 핵심 키워드")
