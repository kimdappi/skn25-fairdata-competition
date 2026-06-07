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
from app.retrieval.types import QueryAnalysis, RankedChunk, RetrievalHit, RetrievalTrace
from app.utils.config import (
    get_backend_capabilities,
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
    is_route_filter_enabled,
    resolve_route_min_candidate_docs,
    resolve_sparse_backend_name,
    resolve_sparse_model_dir,
    resolve_sparse_path_top_k,
    resolve_reranker_backend_name,
    resolve_reranker_top_n,
    resolve_reranker_weight,
    resolve_retrieval_profile,
    use_rrf_fusion,
    validate_retrieval_configuration,
)
from app.utils.text import normalize_name, tokenize_text


def ensure_top_k_chunks(
    corpus_store: CorpusStore,
    ranked_chunks: list,
    top_k: int,
) -> list[Chunk]:
    # 상위 단계에서 중복 chunk_id 가 섞여 들어와도 최종 반환은 고유 청크 기준으로 맞춥니다.
    # 후보가 top_k 보다 적으면 코퍼스 순회로 남은 자리를 채워 평가/추론 인터페이스를 고정합니다.
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


def select_candidate_doc_ids(
    analysis: QueryAnalysis,
    corpus_store: CorpusStore,
    *,
    min_candidate_docs: int,
) -> set[str] | None:
    route = analysis.route
    if route.theme == "기타":
        return None

    theme_doc_ids = {
        document.doc_id
        for document in corpus_store.documents
        if document.route.theme == route.theme
    }
    if len(theme_doc_ids) < min_candidate_docs:
        return None

    if route.industry != "기타":
        refined_doc_ids = {
            document.doc_id
            for document in corpus_store.documents
            if document.doc_id in theme_doc_ids and document.route.industry == route.industry
        }
        if len(refined_doc_ids) >= min_candidate_docs:
            return refined_doc_ids

    return theme_doc_ids


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
        # backend 팩토리가 반환한 런타임을 경로 간에 재사용합니다.
        # 예: BGE-M3 하나를 dense/sparse/multivector가 함께 쓰는 경우 중복 로드를 피합니다.
        self.backend_runtime_cache: dict[tuple[str, Path], object] = {}
        # interfaces.py 에서 정의한 공통 계약 타입으로 backend 인스턴스를 보관합니다.
        # 상위 파이프라인은 구체 모델명을 몰라도 동일한 메서드로 경로를 실행할 수 있습니다.
        self.dense_backend: DenseRetrievalBackend | None = None
        self.sparse_backend: SparseRetrievalBackend | None = None
        self.multivector_backend: MultiVectorRetrievalBackend | None = None
        # engines.py 의 검색 엔진은 "코퍼스 + backend"를 받아 실제 인덱스 구축과 검색을 담당합니다.
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
        self.rerank_top_n = resolve_reranker_top_n()
        self.rerank_weight = resolve_reranker_weight()
        self.retrieval_profile = resolve_retrieval_profile()
        self.enable_route_filter = is_route_filter_enabled()
        self.route_min_candidate_docs = resolve_route_min_candidate_docs()
        validate_retrieval_configuration()

        if not any([self.enable_dense, self.enable_sparse, self.enable_multivector]):
            raise ValueError("At least one retrieval path must be enabled.")

    # 실제 검색이 호출될 때 BGE-M3 모델과 각 검색 엔진을 지연 초기화합니다.
    def ensure_runtime(self) -> None:
        if self.reranker is not None:
            return

        if self.enable_dense:
            # 설정값 문자열을 backends.py 팩토리에 넘겨 실제 dense 구현체를 선택합니다.
            self.dense_backend = build_dense_backend(
                resolve_dense_backend_name(),
                resolve_dense_model_dir(),
                self.backend_runtime_cache,
            )
            # 선택된 backend 는 engines.py 의 DenseSearchEngine 에 주입되어
            # 공통 dense 검색 흐름(인덱스/질의/조회)에 연결됩니다.
            self.dense_engine = DenseSearchEngine(self.corpus_store, self.dense_backend)
        if self.enable_sparse:
            sparse_caps = get_backend_capabilities(resolve_sparse_backend_name())
            if not sparse_caps["sparse"]:
                raise ValueError(
                    f"Sparse backend '{resolve_sparse_backend_name()}' does not support sparse retrieval."
                )
            # sparse 도 같은 방식으로 설정 -> backend 구현체 -> engine 순서로 조립합니다.
            self.sparse_backend = build_sparse_backend(
                resolve_sparse_backend_name(),
                resolve_sparse_model_dir(),
                self.backend_runtime_cache,
            )
            self.sparse_engine = SparseSearchEngine(self.corpus_store, self.sparse_backend)
        if self.enable_multivector:
            multivector_caps = get_backend_capabilities(resolve_multivector_backend_name())
            if not multivector_caps["multivector"]:
                raise ValueError(
                    "Multi-vector backend "
                    f"'{resolve_multivector_backend_name()}' does not support multi-vector retrieval."
                )
            # multivector 는 현재 BGE-M3 계열만 지원하지만,
            # 파이프라인 코드는 공통 인터페이스만 사용하므로 확장 지점은 유지됩니다.
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
        if hasattr(self.reranker, "rerank_weight"):
            self.reranker.rerank_weight = self.rerank_weight

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
        default_path_top_k = max(top_k * 20, 100) if self.enable_route_filter else max(top_k * 6, 30)

        # 각 engine 은 동일한 search 시그니처를 가지므로,
        # 파이프라인은 경로별 top_k 조정만 하고 실행 자체는 공통 흐름으로 다룹니다.
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
            trace_source = f"rrf_{self.retrieval_profile}"
        else:
            fused_scores = self.score_fusion.fuse(active_path_hits, weights)
            trace_source = f"score_{self.retrieval_profile}"
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

    def filter_fused_scores_by_route(
        self,
        analysis: QueryAnalysis,
        fused_scores: dict[str, float],
        *,
        top_k: int,
    ) -> dict[str, float]:
        if not self.enable_route_filter or not fused_scores:
            return fused_scores

        candidate_doc_ids = select_candidate_doc_ids(
            analysis,
            self.corpus_store,
            min_candidate_docs=self.route_min_candidate_docs,
        )
        if not candidate_doc_ids:
            return fused_scores

        candidate_chunk_ids = {
            chunk_id
            for doc_id in candidate_doc_ids
            for chunk_id in self.corpus_store.document_to_chunk_ids.get(doc_id, [])
        }
        filtered_scores = {
            chunk_id: score
            for chunk_id, score in fused_scores.items()
            if chunk_id in candidate_chunk_ids
        }
        if len(filtered_scores) < top_k:
            return fused_scores
        return filtered_scores

    # 최종 파이프라인을 실행해 평가용 상위 청크를 반환합니다.
    def search(self, question: str, top_k: int = 5) -> list[Chunk]:
        # 전체 흐름:
        # 1) 질문 분석 2) 경로별 검색 3) fusion 4) rerank(optional) 5) 최종 chunk 반환
        analysis = self.analyze_query(question)
        trace = self.run_retrieval_paths(analysis, top_k=top_k)
        fused_scores = self.fuse_hits(analysis, trace)
        fused_scores = self.filter_fused_scores_by_route(analysis, fused_scores, top_k=top_k)
        if self.rerank_weight <= 0:
            ranked_chunks = [
                RankedChunk(
                    chunk=self.corpus_store.chunk_map[chunk_id],
                    score=score,
                    reasons=("fusion",),
                )
                for chunk_id, score in sorted(
                    fused_scores.items(),
                    key=lambda item: (item[1], item[0]),
                    reverse=True,
                )
                if chunk_id in self.corpus_store.chunk_map
            ]
            return ensure_top_k_chunks(self.corpus_store, ranked_chunks, top_k)

        reranked_chunks = self.reranker.rerank_chunks(
            analysis,
            fused_scores,
            top_n=max(top_k * 10, self.rerank_top_n),
        )
        return ensure_top_k_chunks(self.corpus_store, reranked_chunks, top_k)
