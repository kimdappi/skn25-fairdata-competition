from __future__ import annotations

from app.preprocessing.corpus import Chunk, CorpusStore, Document
from app.rerank.reranker import BGERerankerV2M3
from app.retrieval.bgem3 import BGEM3HybridModel
from app.retrieval.engines import DenseSearchEngine, MultiVectorSearchEngine, SparseSearchEngine
from app.retrieval.router import QueryRouter
from app.retrieval.types import QueryAnalysis, RetrievalHit, RetrievalTrace
from app.utils.config import resolve_bge_reranker_model_dir, resolve_bgem3_model_dir
from app.utils.text import normalize_name, tokenize_text


def ensure_top_k_chunks(
    corpus_store: CorpusStore,
    ranked_chunks: list,
    top_k: int,
) -> list[Chunk]:
    selected_chunks: list[Chunk] = []
    seen_chunk_ids: set[str] = set()

    for ranked in ranked_chunks:
        chunk = ranked.chunk
        if chunk.chunk_id in seen_chunk_ids:
            continue
        selected_chunks.append(chunk)
        seen_chunk_ids.add(chunk.chunk_id)
        if len(selected_chunks) == top_k:
            return selected_chunks

    for chunk in corpus_store.chunks:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        selected_chunks.append(chunk)
        seen_chunk_ids.add(chunk.chunk_id)
        if len(selected_chunks) == top_k:
            break
    return selected_chunks


class RRFFusion:
    # 여러 검색 경로 결과를 RRF 방식으로 합산하는 유틸리티를 초기화합니다.
    def __init__(self, k: int = 60) -> None:
        self.k = k

    # 여러 검색 결과 순위를 RRF 방식으로 합산합니다.
    def fuse(self, rankings: dict[str, list[str]], weights: dict[str, float]) -> dict[str, float]:
        fused_scores: dict[str, float] = {}
        for name, ranking in rankings.items():
            weight = weights.get(name, 1.0)
            for rank, item_id in enumerate(ranking, start=1):
                fused_scores[item_id] = fused_scores.get(item_id, 0.0) + weight / (self.k + rank)
        return fused_scores

    # 점수 사전을 상위 순위 목록으로 변환합니다.
    def rank_ids(self, scores: dict[str, float], top_k: int) -> list[str]:
        return [
            item_id
            for item_id, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:top_k]
        ]


class DocumentCandidateSelector:
    # 문서 후보군을 먼저 좁히는 선택기를 초기화합니다.
    def __init__(self, documents: list[Document], fusion: RRFFusion) -> None:
        self.documents = documents
        self.fusion = fusion

    # 질문과 문서 메타데이터 겹침을 이용해 문서 점수를 계산합니다.
    def score_documents(self, analysis: QueryAnalysis) -> dict[str, float]:
        scores: dict[str, float] = {}
        query_token_set = set(analysis.tokens)
        for document in self.documents:
            overlap = len(query_token_set & set(document.tokens))
            if overlap == 0:
                continue
            title_bonus = 1.0 if document.normalized_doc_name in analysis.normalized_question else 0.0
            company_bonus = 0.0
            for company_name in document.company_names:
                if normalize_name(company_name) in analysis.normalized_question:
                    company_bonus += 1.0
            scores[document.doc_id] = overlap + title_bonus + company_bonus
        return scores

    # 문서 점수 상위 결과만 후보 문서 집합으로 반환합니다.
    def select(self, analysis: QueryAnalysis, *, top_k: int) -> set[str]:
        scores = self.score_documents(analysis)
        return set(self.fusion.rank_ids(scores, top_k))


class HybridSearchPipeline:
    # dense, sparse, fusion, rerank 단계를 묶는 하이브리드 검색 파이프라인을 초기화합니다.
    def __init__(self, corpus_store: CorpusStore, router: QueryRouter) -> None:
        self.corpus_store = corpus_store
        self.router = router
        self.model: BGEM3HybridModel | None = None
        self.dense_engine: DenseSearchEngine | None = None
        self.sparse_engine: SparseSearchEngine | None = None
        self.multivector_engine: MultiVectorSearchEngine | None = None
        self.reranker: BGERerankerV2M3 | None = None
        self.fusion = RRFFusion(k=60)
        self.document_selector = DocumentCandidateSelector(self.corpus_store.documents, self.fusion)

    # 실제 검색이 호출될 때 BGE-M3 모델과 각 검색 엔진을 지연 초기화합니다.
    def ensure_runtime(self) -> None:
        if self.model is not None:
            return
        self.model = BGEM3HybridModel(resolve_bgem3_model_dir())
        self.dense_engine = DenseSearchEngine(self.corpus_store, self.model)
        self.sparse_engine = SparseSearchEngine(self.corpus_store, self.model)
        self.multivector_engine = MultiVectorSearchEngine(self.corpus_store, self.model)
        self.reranker = BGERerankerV2M3(self.corpus_store, resolve_bge_reranker_model_dir())

    # 질문을 토큰과 라우팅 메타데이터로 분석합니다.
    def analyze_query(self, question: str) -> QueryAnalysis:
        route = self.router.route_from_text(question)
        return QueryAnalysis(
            question=question,
            route=route,
            tokens=tuple(tokenize_text(question)),
            normalized_question=normalize_name(question),
        )

    # 문서 후보를 먼저 추려 청크 단계 검색 범위를 줄입니다.
    def select_candidate_documents(self, analysis: QueryAnalysis, *, top_k: int) -> set[str]:
        return self.document_selector.select(analysis, top_k=top_k)

    # dense, sparse, multivector 경로를 각각 실행하고 추적 정보를 남깁니다.
    def run_retrieval_paths(
        self,
        analysis: QueryAnalysis,
        *,
        candidate_doc_ids: set[str],
        top_k: int,
    ) -> RetrievalTrace:
        self.ensure_runtime()
        trace = RetrievalTrace()
        path_top_k = max(top_k * 6, 30)
        trace.dense_hits = self.dense_engine.search(analysis, candidate_doc_ids=candidate_doc_ids, top_k=path_top_k)
        trace.sparse_hits = self.sparse_engine.search(analysis, candidate_doc_ids=candidate_doc_ids, top_k=path_top_k)
        trace.multivector_hits = self.multivector_engine.search(analysis, candidate_doc_ids=candidate_doc_ids, top_k=path_top_k)
        return trace

    # 각 경로 결과를 RRF로 융합해 단일 랭킹으로 만듭니다.
    def fuse_hits(self, trace: RetrievalTrace) -> dict[str, float]:
        rankings = {
            "dense": [hit.chunk_id for hit in trace.dense_hits],
            "sparse": [hit.chunk_id for hit in trace.sparse_hits],
            "multivector": [hit.chunk_id for hit in trace.multivector_hits],
        }
        weights = {
            "dense": 0.95,
            "sparse": 1.0,
            "multivector": 1.05,
        }
        fused_scores = self.fusion.fuse(rankings, weights)
        trace.fused_hits = sorted(
            [RetrievalHit(chunk_id=chunk_id, score=score, source="rrf") for chunk_id, score in fused_scores.items()],
            key=lambda item: (item.score, item.chunk_id),
            reverse=True,
        )
        return fused_scores

    # 최종 파이프라인을 실행해 평가용 상위 청크를 반환합니다.
    def search(self, question: str, top_k: int = 5) -> list[Chunk]:
        analysis = self.analyze_query(question)
        candidate_doc_ids = self.select_candidate_documents(analysis, top_k=12)
        trace = self.run_retrieval_paths(analysis, candidate_doc_ids=candidate_doc_ids, top_k=top_k)
        fused_scores = self.fuse_hits(trace)
        reranked_chunks = self.reranker.rerank_chunks(analysis, fused_scores, top_n=max(top_k * 10, 50))
        return ensure_top_k_chunks(self.corpus_store, reranked_chunks, top_k)
