from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # 평가 서버와 로컬 평가 스크립트가 보내는 요청 본문 형식입니다.
    id: str = Field(..., description="평가 요청 식별자")
    question: str = Field(..., description="공정위 의결서 질의문")


class PredictResponse(BaseModel):
    # 제출 가이드가 요구하는 `/predict` 응답 형식입니다.
    id: str
    retrieved_chunk_ids: List[str]
    answer: str


class RouteDecision(BaseModel):
    # 라우터가 질문/문서에서 추출한 의도 메타데이터 묶음입니다.
    theme: str = Field(description="질문의 핵심 위반 분야")
    company_size: str = Field(description="질문에 드러난 기업 규모")
    legal_role: str = Field(description="질문에 드러난 법적 지위")
    industry: str = Field(description="질문에 드러난 업종")
    focus: str = Field(description="질문 초점")
    keywords: List[str] = Field(description="질문 핵심 키워드")


class LLMRouteDecision(BaseModel):
    # LLM 라우터가 생성하는 태그 원본입니다. 검색에는 RouteDecision으로 변환해 사용합니다.
    theme: str
    company_size: str
    legal_role: str
    industry: str
    focus: str
    confidence: float = 0.0
    reason: str = ""
