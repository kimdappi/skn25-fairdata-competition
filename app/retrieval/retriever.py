from __future__ import annotations

from app.preprocessing.corpus import Chunk, CorpusStore
from app.retrieval.pipeline import HybridSearchPipeline
from app.retrieval.router import QueryRouter


class HybridRetriever:
    # 외부 API가 사용할 검색기 인터페이스를 초기화합니다.
    def __init__(self, corpus_store: CorpusStore, router: QueryRouter) -> None:
        self.pipeline = HybridSearchPipeline(corpus_store, router)

    # 질문 하나를 입력받아 최종 상위 청크를 반환합니다.
    def search(self, question: str, top_k: int = 5, query_id: str | None = None) -> list[Chunk]:
        return self.pipeline.search(question, top_k=top_k, query_id=query_id)
