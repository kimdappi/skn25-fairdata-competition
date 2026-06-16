from __future__ import annotations

from dataclasses import dataclass, field

from app.preprocessing.corpus import Chunk
from app.utils.schemas import RouteDecision


@dataclass(frozen=True)
class QueryAnalysis:
    # 질문 원문과 라우팅/토큰화 결과를 함께 들고 다니는 검색 입력 객체입니다.
    question: str
    route: RouteDecision
    tokens: tuple[str, ...]
    normalized_question: str
    query_id: str | None = None


@dataclass(frozen=True)
class RetrievalHit:
    # 개별 검색 경로가 반환한 청크 후보 1건과 그 경로 점수입니다.
    chunk_id: str
    score: float
    source: str


@dataclass
class RetrievalTrace:
    # dense, sparse, multivector, fused 결과를 단계별로 추적하기 위한 컨테이너입니다.
    dense_hits: list[RetrievalHit] = field(default_factory=list)
    sparse_hits: list[RetrievalHit] = field(default_factory=list)
    multivector_hits: list[RetrievalHit] = field(default_factory=list)
    fused_hits: list[RetrievalHit] = field(default_factory=list)


@dataclass(frozen=True)
class RankedChunk:
    # reranker 이후 최종 점수와 반영 근거를 묶은 최종 랭킹 단위입니다.
    chunk: Chunk
    score: float
    reasons: tuple[str, ...]
