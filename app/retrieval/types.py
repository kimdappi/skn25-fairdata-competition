from __future__ import annotations

from dataclasses import dataclass, field

from app.preprocessing.corpus import Chunk
from app.utils.schemas import RouteDecision


@dataclass(frozen=True)
class QueryAnalysis:
    question: str
    route: RouteDecision
    tokens: tuple[str, ...]
    normalized_question: str


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    score: float
    source: str


@dataclass
class RetrievalTrace:
    dense_hits: list[RetrievalHit] = field(default_factory=list)
    sparse_hits: list[RetrievalHit] = field(default_factory=list)
    multivector_hits: list[RetrievalHit] = field(default_factory=list)
    fused_hits: list[RetrievalHit] = field(default_factory=list)


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float
    reasons: tuple[str, ...]
