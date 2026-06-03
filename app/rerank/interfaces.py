from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.retrieval.types import QueryAnalysis, RankedChunk


class RerankerBackend(Protocol):
    # 검색 융합 결과를 최종 청크 순위로 재정렬하는 리랭커 인터페이스입니다.
    model_dir: Path

    def rerank_chunks(
        self,
        analysis: QueryAnalysis,
        fused_scores: dict[str, float],
        *,
        top_n: int = 50,
    ) -> list[RankedChunk]: ...
