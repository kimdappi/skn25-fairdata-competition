from __future__ import annotations

from pathlib import Path

from app.preprocessing.corpus import Chunk, CorpusStore
from app.rerank.backends import build_reranker_backend
from app.rerank.interfaces import RerankerBackend
from app.retrieval.backends import (
    build_dense_backend,
    build_multivector_backend,
    build_sparse_backend,
)
from app.retrieval.engines import DenseSearchEngine, MultiVectorSearchEngine, SparseSearchEngine
from app.retrieval.interfaces import (
    DenseRetrievalBackend,
    MultiVectorRetrievalBackend,
    SparseRetrievalBackend,
)
from app.retrieval.router import QueryRouter
from app.retrieval.types import QueryAnalysis, RetrievalHit, RetrievalTrace
from app.utils.config import (
    is_dense_enabled,
    is_multivector_enabled,
    is_sparse_enabled,
    resolve_bge_reranker_model_dir,
    resolve_dense_backend_name,
    resolve_dense_model_dir,
    resolve_dense_path_top_k,
    resolve_multivector_backend_name,
    resolve_multivector_model_dir,
    resolve_multivector_path_top_k,
    resolve_sparse_backend_name,
    resolve_sparse_model_dir,
    resolve_sparse_path_top_k,
    resolve_reranker_backend_name,
    use_rrf_fusion,
)
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


def compute_route_bonus(analysis: QueryAnalysis, chunk: Chunk) -> float:
    bonus = 0.0
    query_route = analysis.route
    chunk_route = chunk.route

    if query_route.theme != "기타" and query_route.theme == chunk_route.theme:
        bonus += 0.012
    if query_route.focus != "일반" and query_route.focus == chunk_route.focus:
        bonus += 0.010
    if query_route.legal_role != "기타" and query_route.legal_role == chunk_route.legal_role:
        bonus += 0.008
    if query_route.industry != "기타" and query_route.industry == chunk_route.industry:
        bonus += 0.008
    if query_route.company_size != "기타" and query_route.company_size == chunk_route.company_size:
        bonus += 0.005
    return bonus


class ScoreFusion:
    # RRF를 쓰지 않을 때 경로별 raw score를 정규화해 합산합니다.
    def fuse(self, traces: dict[str, list[RetrievalHit]], weights: dict[str, float]) -> dict[str, float]:
        fused_scores: dict[str, float] = {}
        for name, hits in traces.items():
            if not hits:
                continue
            weight = weights.get(name, 1.0)
            scores = [hit.score for hit in hits]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            for hit in hits:
                normalized_score = 1.0 if score_range == 0 else (hit.score - min_score) / score_range
                fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + (weight * normalized_score)
        return fused_scores


class HybridSearchPipeline:
    # dense, sparse, multivector, fusion, rerank 단계를 묶는 하이브리드 검색 파이프라인을 초기화합니다.
    def __init__(self, corpus_store: CorpusStore, router: QueryRouter) -> None:
        self.corpus_store = corpus_store
        self.router = router
        self.backend_runtime_cache: dict[tuple[str, Path], object] = {}
        self.dense_backend: DenseRetrievalBackend | None = None
        self.sparse_backend: SparseRetrievalBackend | None = None
        self.multivector_backend: MultiVectorRetrievalBackend | None = None
        self.dense_engine: DenseSearchEngine | None = None
        self.sparse_engine: SparseSearchEngine | None = None
        self.multivector_engine: MultiVectorSearchEngine | None = None
        self.reranker: RerankerBackend | None = None
        self.fusion = RRFFusion(k=60)
        self.score_fusion = ScoreFusion()
        self.enable_dense = is_dense_enabled()
        self.enable_sparse = is_sparse_enabled()
        self.enable_multivector = is_multivector_enabled()
        self.use_rrf = use_rrf_fusion()

        if not any([self.enable_dense, self.enable_sparse, self.enable_multivector]):
            raise ValueError("At least one retrieval path must be enabled.")

    # 실제 검색이 호출될 때 BGE-M3 모델과 각 검색 엔진을 지연 초기화합니다.
    def ensure_runtime(self) -> None:
        if self.reranker is not None:
            return

        if self.enable_dense:
            self.dense_backend = build_dense_backend(
                resolve_dense_backend_name(),
                resolve_dense_model_dir(),
                self.backend_runtime_cache,
            )
            self.dense_engine = DenseSearchEngine(self.corpus_store, self.dense_backend)
        if self.enable_sparse:
            self.sparse_backend = build_sparse_backend(
                resolve_sparse_backend_name(),
                resolve_sparse_model_dir(),
                self.backend_runtime_cache,
            )
            self.sparse_engine = SparseSearchEngine(self.corpus_store, self.sparse_backend)
        if self.enable_multivector:
            self.multivector_backend = build_multivector_backend(
                resolve_multivector_backend_name(),
                resolve_multivector_model_dir(),
                self.backend_runtime_cache,
            )
            self.multivector_engine = MultiVectorSearchEngine(self.corpus_store, self.multivector_backend)
        self.reranker = build_reranker_backend(
            resolve_reranker_backend_name(),
            self.corpus_store,
            resolve_bge_reranker_model_dir(),
        )

    # 질문을 토큰과 라우팅 메타데이터로 분석합니다.
    def analyze_query(self, question: str) -> QueryAnalysis:
        route = self.router.route_from_text(question)
        return QueryAnalysis(
            question=question,
            route=route,
            tokens=tuple(tokenize_text(question)),
            normalized_question=normalize_name(question),
        )

    # dense, sparse, multivector 경로를 전체 코퍼스 기준으로 각각 실행하고 추적 정보를 남깁니다.
    def run_retrieval_paths(
        self,
        analysis: QueryAnalysis,
        *,
        top_k: int,
    ) -> RetrievalTrace:
        self.ensure_runtime()
        trace = RetrievalTrace()
        default_path_top_k = max(top_k * 6, 30)

        if self.enable_dense and self.dense_engine is not None:
            dense_top_k = resolve_dense_path_top_k(default_path_top_k)
            trace.dense_hits = self.dense_engine.search(analysis, top_k=dense_top_k)
        if self.enable_sparse and self.sparse_engine is not None:
            sparse_top_k = resolve_sparse_path_top_k(default_path_top_k)
            trace.sparse_hits = self.sparse_engine.search(analysis, top_k=sparse_top_k)
        if self.enable_multivector and self.multivector_engine is not None:
            multivector_top_k = resolve_multivector_path_top_k(default_path_top_k)
            trace.multivector_hits = self.multivector_engine.search(analysis, top_k=multivector_top_k)
        return trace

    # 각 경로 결과를 RRF로 융합하고 라우팅 일치도를 soft bonus로 반영합니다.
    def fuse_hits(self, analysis: QueryAnalysis, trace: RetrievalTrace) -> dict[str, float]:
        path_hits = {
            "dense": trace.dense_hits,
            "sparse": trace.sparse_hits,
            "multivector": trace.multivector_hits,
        }
        active_path_hits = {name: hits for name, hits in path_hits.items() if hits}
        weights = {
            "dense": 0.95,
            "sparse": 1.0,
            "multivector": 1.05,
        }
        if self.use_rrf:
            rankings = {
                name: [hit.chunk_id for hit in hits]
                for name, hits in active_path_hits.items()
            }
            fused_scores = self.fusion.fuse(rankings, weights)
            trace_source = "rrf_route"
        else:
            fused_scores = self.score_fusion.fuse(active_path_hits, weights)
            trace_source = "score_route"
        for chunk_id in list(fused_scores.keys()):
            chunk = self.corpus_store.chunk_map.get(chunk_id)
            if chunk is None:
                continue
            fused_scores[chunk_id] += compute_route_bonus(analysis, chunk)
        trace.fused_hits = sorted(
            [
                RetrievalHit(chunk_id=chunk_id, score=score, source=trace_source)
                for chunk_id, score in fused_scores.items()
            ],
            key=lambda item: (item.score, item.chunk_id),
            reverse=True,
        )
        return fused_scores

    # 최종 파이프라인을 실행해 평가용 상위 청크를 반환합니다.
    def search(self, question: str, top_k: int = 5) -> list[Chunk]:
        analysis = self.analyze_query(question)
        trace = self.run_retrieval_paths(analysis, top_k=top_k)
        fused_scores = self.fuse_hits(analysis, trace)
        reranked_chunks = self.reranker.rerank_chunks(analysis, fused_scores, top_n=max(top_k * 10, 50))
        return ensure_top_k_chunks(self.corpus_store, reranked_chunks, top_k)
