"""
================================================================================
ProofRAG Consensus Retriever — 모형 3 (해시체인 + Consensus RRF)
================================================================================

기존 LegalRAGRetriever (모형 1) + ProofRAG Consensus RRF (모형 3)를 결합한
통합 리트리버입니다. 기존 retriever의 LangGraph 파이프라인은 유지하면서,
chunk-level RRF에 Consensus Weight를 추가합니다.
================================================================================
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from baseline_rag.config import (
    CHUNK_FAISS_K,
    CHUNK_RRF_BM25_WEIGHT,
    CHUNK_RRF_DENSE_WEIGHT,
    RRF_K,
    STAGE_LOG_ENABLED,
    TOP_K_CHUNKS,
)
from baseline_rag.proof_rag import (
    ConsensusRRFConfig,
    ConsensusWeightedRRF,
    HashChainRecord,
    SPVVerifier,
    build_hash_chain,
)
from baseline_rag.retrieval_types import ChunkRecord, DocumentRecord, RetrievalState
from baseline_rag.retrieval_utils import (
    min_max_scale,
    ranked_indices_from_scores,
    tokenize_text,
)
from baseline_rag.schemas import RetrievedChunk


class ProofRAGConsensusRetriever:
    """
    ProofRAG Consensus-Weighted Hybrid Retriever (모형 3)

    기존 LegalRAGRetriever의 document-level RRF + route_boost에
    ProofRAG의 해시체인 검증과 Consensus Weighted RRF를 추가한 버전.

    주요 개선점:
    1. Hash Chain: 각 청크에 SHA-256 체인 적용 → 무결성 보장
    2. Consensus RRF: 동일 위반유형 청크 다수 발견 시 가중치 부여
    3. SPV 검증: chunk_id + hash로 경량 검증 가능
    """

    def __init__(
        self,
        *,
        base_retriever,  # LegalRAGRetriever 인스턴스
        consensus_config: Optional[ConsensusRRFConfig] = None,
    ):
        self.base = base_retriever
        self.consensus_config = consensus_config or ConsensusRRFConfig()
        self.consensus_rrf = ConsensusWeightedRRF(self.consensus_config)
        self.stage_log_enabled = os.getenv(
            "STAGE_LOG_ENABLED", str(STAGE_LOG_ENABLED)
        ).lower() in {"1", "true", "yes", "on"}

        # Hash chain infrastructure
        self._hash_chain: List[HashChainRecord] = []
        self._spv_verifier: Optional[SPVVerifier] = None
        self._build_hash_chain()

    def _log_stage(self, stage: str, started_at: float, *, extra: str = "") -> None:
        if not self.stage_log_enabled:
            return
        elapsed = time.perf_counter() - started_at
        suffix = f" {extra}" if extra else ""
        print(f"[stage] proofrag.{stage} {elapsed:.3f}s{suffix}")

    def _build_hash_chain(self) -> None:
        """코퍼스 청크로부터 해시체인 생성"""
        started_at = time.perf_counter()
        chunk_dicts = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "enriched_text": chunk.enriched_text,
                "metadata": {
                    "title": chunk.title,
                    "violation": getattr(chunk, "route_theme", ""),
                    "company": getattr(chunk, "route_legal_role", ""),
                },
            }
            for chunk in self.base.chunks
        ]
        self._hash_chain = build_hash_chain(chunk_dicts)
        self._spv_verifier = SPVVerifier(self._hash_chain)
        self._log_stage(
            "build_hash_chain",
            started_at,
            extra=f"chunks={len(self._hash_chain)}",
        )

    def retrieve_chunks_with_consensus(
        self, state: RetrievalState
    ) -> RetrievalState:
        """
        Consensus-Weighted RRF를 적용한 청크 검색

        기존 retrieve_chunks 로직에서 RRF 단계를 ConsensusRRF로 대체합니다.
        """
        started_at = time.perf_counter()
        query = state["query"]
        route = state["route"]
        timings = dict(state.get("timings", {}))
        candidate_doc_ids = state["candidate_doc_ids"][:3]

        candidate_indices = self.base._collect_candidate_chunk_indices(candidate_doc_ids)
        if not candidate_indices:
            timings["chunks"] = time.perf_counter() - started_at
            return {"results": [], "timings": timings}

        # Dense & BM25 검색
        dense_query_vec = self.base._embed_chunk_dense_query(query)
        query_tokens = self.base.tokenize(query)
        bm25_scores = (
            np.array(self.base.chunk_bm25.get_scores(query_tokens), dtype="float32")
            if self.base.options.use_bm25
            else np.zeros(len(self.base.chunks), dtype="float32")
        )

        dense_ranking = [
            candidate_indices[idx]
            for idx in self._rank_dense(candidate_indices, dense_query_vec)
        ]
        bm25_ranking = [
            candidate_indices[idx]
            for idx in self._rank_bm25(candidate_indices, bm25_scores)
        ]

        # ── ProofRAG Consensus RRF ──
        corpus_info = [
            {
                "metadata": {
                    "violation": chunk.route_theme,
                    "company": chunk.route_legal_role,
                    "title": chunk.title,
                }
            }
            for chunk in self.base.chunks
        ]

        consensus_scores = self.consensus_rrf.compute_rrf_scores(
            dense_ranking=dense_ranking,
            bm25_ranking=bm25_ranking,
            corpus_info=corpus_info,
            candidate_indices=candidate_indices,
        )

        # 기존 scoring + consensus RRF 통합
        lexical_scores = np.zeros(len(candidate_indices), dtype="float32")
        if self.base.options.use_chunk_lexical_score:
            sparse_query_vec = self.base._embed_chunk_query(query)
            candidate_matrix = self.base.chunk_sparse_matrix[candidate_indices]
            lexical_scores = (
                (candidate_matrix @ sparse_query_vec.T).toarray().ravel().astype("float32")
            )

        scored_chunks = self._score_with_consensus(
            query=query,
            route=route,
            candidate_doc_ids=candidate_doc_ids,
            candidate_indices=candidate_indices,
            lexical_scores=lexical_scores,
            consensus_scores=consensus_scores,
        )
        scored_chunks = self.base._rerank_chunks_with_bge_colbert(query, scored_chunks)
        top_chunks = self.base._select_top_chunks(scored_chunks)

        # SPV 검증 정보 추가
        verified_count = self._verify_chunks_batch(
            [(c.chunk_id, self._get_hash(c.chunk_id)) for c, _ in scored_chunks[:5]]
        )

        timings["chunks"] = time.perf_counter() - started_at
        timings["spv_verified"] = verified_count
        self._log_stage(
            "retrieve_chunks_consensus",
            started_at,
            extra=f"query={query!r} top={len(top_chunks)} verified={verified_count}",
        )
        return {"results": top_chunks, "timings": timings}

    def _rank_dense(self, candidate_indices: List[int], query_vec) -> List[int]:
        candidate_dense = self.base.chunk_dense[candidate_indices]
        dense_scores = (candidate_dense @ query_vec[0]).astype("float32")
        return ranked_indices_from_scores(
            dense_scores, min(CHUNK_FAISS_K, len(candidate_indices))
        )

    def _rank_bm25(self, candidate_indices: List[int], bm25_scores) -> List[int]:
        return ranked_indices_from_scores(
            bm25_scores[candidate_indices],
            min(CHUNK_FAISS_K, len(candidate_indices)),
        )

    def _score_with_consensus(
        self,
        *,
        query: str,
        route,
        candidate_doc_ids: List[str],
        candidate_indices: List[int],
        lexical_scores: np.ndarray,
        consensus_scores: Dict[int, float],
    ) -> List[Tuple[ChunkRecord, float]]:
        """Consensus RRF 점수 + 기존 boost 점수 통합"""
        lexical_scaled = min_max_scale(lexical_scores)
        query_norm = self.base._retrieval_norm(query)
        doc_rank_map = {doc_id: rank for rank, doc_id in enumerate(candidate_doc_ids)}
        scored_chunks = []

        for local_idx, chunk_idx in enumerate(candidate_indices):
            chunk = self.base.chunks[chunk_idx]
            # Consensus RRF 점수 (0.0 ~ 1.0 범위로 정규화)
            score = consensus_scores.get(chunk_idx, 0.0) * 3.0  # scale up

            # 기존 boost (retriever.py와 동일)
            score += min(0.12, float(lexical_scaled[local_idx]) * 0.12)
            if self.base.options.use_doc_rank_boost:
                score += max(0.0, 0.18 - doc_rank_map[chunk.doc_id] * 0.06)
            if self.base.options.use_route_boost:
                score += self.base._chunk_route_bonus(chunk, route)

            overlap = len(set(self.base.tokenize(query)) & self.base.chunk_query_tokens[chunk_idx])
            score += min(0.18, overlap * 0.01)

            if self.base.options.use_chunk_structure_boost:
                score += self.base._chunk_focus_bonus(chunk, route)
                if chunk.normalized_title[:12] and chunk.normalized_title[:12] in query_norm:
                    score += 0.08

            scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda item: item[1], reverse=True)
        return scored_chunks

    def _retrieval_norm(self, text: str) -> str:
        """retrieval_utils.normalize_name wrapper"""
        from baseline_rag.retrieval_utils import normalize_name
        return normalize_name(text)

    def _get_hash(self, chunk_id: str) -> str:
        for record in self._hash_chain:
            if record.chunk_id == chunk_id:
                return record.hash_value
        return ""

    def _verify_chunks_batch(self, chunk_hashes: List[Tuple[str, str]]) -> int:
        if self._spv_verifier is None:
            return 0
        results = self._spv_verifier.verify_batch(chunk_hashes)
        return sum(1 for r in results if r["is_verified"])

    def search_with_consensus(self, query: str, route) -> List[RetrievedChunk]:
        """Consensus RRF 검색 진입점 (모형 3)"""
        started_at = time.perf_counter()
        state: RetrievalState = {
            "query": query,
            "route": route,
            "timings": {"route": 0.0},
        }
        state.update(self.base.retrieve_documents(state))
        state.update(self.retrieve_chunks_with_consensus(state))
        results = state.get("results", [])
        self._log_stage(
            "search_consensus_total",
            started_at,
            extra=f"query={query!r} top={len(results)}",
        )
        return results
