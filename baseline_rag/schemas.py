from typing import List

from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    query: str = Field(description="공정거래위원회 의결서 검색용 한국어 질문")


class RouteDecision(BaseModel):
    theme: str = Field(description="질문의 핵심 위반 분야")
    company_size: str = Field(description="질문에서 암시되는 기업 규모")
    legal_role: str = Field(description="질문에서 암시되는 법적 지위")
    industry: str = Field(description="질문에서 암시되는 업종")
    focus: str = Field(description="질문의 초점. 예: 처분, 사실관계, 위법성")
    keywords: List[str] = Field(description="질문에서 추출된 핵심 키워드 목록")


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(description="평가에 제출할 청크 식별자")
    title: str = Field(description="의결서 제목")
    score: float = Field(description="최종 검색 점수")
