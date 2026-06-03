"""
================================================================================
ProofRAG: 블록체인 기반 해시체인 검증 + Consensus Weighted RRF 모듈
================================================================================

이 모듈은 비트코인의 선형 해시체인(Linear Hash Chain) 개념을 RAG 파이프라인에
적용하여, 검색된 청크의 데이터 무결성과 출처 신뢰도를 검증합니다.

[핵심 기능]
1. SHA-256 해시체인 생성: 각 청크에 prev_chunk_hash 포함 → 위변조 탐지
2. 타임스탬프 기반 출처 증명 (Data Provenance)
3. Consensus Weighted RRF: 동일 위반유형 청크가 3개 이상 → 가중치 부여
4. SPV 경량 검증: chunk_id + hash만으로 체인 검증 가능

Ref: skidroww/ProofRAG, Nakamoto (2008) Bitcoin Whitepaper
================================================================================
"""

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# Hash Chain Utilities
# ──────────────────────────────────────────────

def compute_sha256(text: str) -> str:
    """SHA-256 해시 생성 (ProofRAG 호환)"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class HashChainRecord:
    """해시체인이 포함된 청크 레코드"""
    chunk_id: str
    doc_id: str
    text: str
    enriched_text: str
    hash_value: str                    # 현재 청크의 해시
    prev_hash: str                     # 이전 청크 해시 (체인)
    timestamp: str                     # ISO 8601 타임스탬프
    metadata: Dict[str, str] = field(default_factory=dict)
    chain_index: int = 0               # 문서 내 청크 순번

    def verify(self, prev_record: Optional["HashChainRecord"]) -> bool:
        """이전 청크와의 체인 연결 검증"""
        if prev_record is None:
            # 첫 청크는 prev_hash가 빈 문자열이어야 함
            return self.prev_hash == ""
        expected = compute_sha256(prev_record.text)
        return self.prev_hash == expected


def build_hash_chain(chunks: List[Dict]) -> List[HashChainRecord]:
    """
    청크 리스트로부터 선형 해시체인을 생성합니다.
    각 청크는 sha256(prev_chunk.text)를 prev_hash로 가집니다.

    Args:
        chunks: [{"chunk_id": ..., "text": ..., "metadata": ...}, ...]

    Returns:
        해시체인이 포함된 청크 리스트
    """
    records = []
    prev_hash = ""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for idx, chunk in enumerate(chunks):
        text = chunk.get("enriched_text", chunk.get("text", ""))
        hash_val = compute_sha256(text)
        records.append(HashChainRecord(
            chunk_id=chunk.get("chunk_id", f"chunk_{idx}"),
            doc_id=chunk.get("doc_id", ""),
            text=text,
            enriched_text=text,
            hash_value=hash_val,
            prev_hash=prev_hash,
            timestamp=timestamp,
            metadata=chunk.get("metadata", {}),
            chain_index=idx,
        ))
        prev_hash = hash_val

    return records


def verify_hash_chain(records: List[HashChainRecord]) -> Tuple[bool, List[str]]:
    """
    전체 해시체인의 무결성을 검증합니다.

    Returns:
        (is_valid, errors): 유효성 여부와 오류 메시지 리스트
    """
    errors = []
    for i, record in enumerate(records):
        prev = records[i - 1] if i > 0 else None
        if not record.verify(prev):
            errors.append(f"Chain broken at chunk {record.chunk_id} (index {i})")
    return len(errors) == 0, errors


# ──────────────────────────────────────────────
# Consensus Weighted RRF (ProofRAG 핵심 알고리즘)
# ──────────────────────────────────────────────

@dataclass
class ConsensusRRFConfig:
    """Consensus RRF 가중치 설정 (기획서 표 기반)"""
    vector_weight: float = 0.60    # Dense 벡터 검색 가중치
    bm25_weight: float = 0.25      # BM25 키워드 검색 가중치
    consensus_weight: float = 0.15  # 합의 가중치 (가장 긴 체인 원리)
    rrf_k: int = 60                 # RRF k 상수
    consensus_threshold: int = 3    # 동일 위반유형 청크 최소 개수
    top_k: int = 5                  # 최종 반환 청크 수


class ConsensusWeightedRRF:
    """
    Consensus Weighted Reciprocal Rank Fusion

    기존 RRF에 '합의 가중치(Consensus Weight)'를 추가하여,
    동일 위반유형(또는 기업명)의 청크가 다수 검색될 경우
    해당 도메인의 모든 청크에 가중치를 부여합니다.

    이는 비트코인의 '가장 긴 체인' 원리와 동일:
    - 더 많은 노드(청크)가 동일한 위반유형을 지지 → 더 높은 신뢰도
    - 소수의 이질적 청크는 낮은 가중치
    """

    def __init__(self, config: Optional[ConsensusRRFConfig] = None):
        self.config = config or ConsensusRRFConfig()

    def compute_rrf_scores(
        self,
        *,
        dense_ranking: List[int],
        bm25_ranking: List[int],
        corpus_info: List[Dict],
        candidate_indices: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """
        Consensus-weighted RRF 점수 계산

        Args:
            dense_ranking: Dense 검색 랭킹 (인덱스 리스트)
            bm25_ranking: BM25 검색 랭킹 (인덱스 리스트)
            corpus_info: 청크 메타데이터 리스트 (violation, company 등 포함)
            candidate_indices: 필터링할 후보 인덱스 (없으면 전체)

        Returns:
            {chunk_index: rrf_score} 딕셔너리
        """
        k = self.config.rrf_k
        vw = self.config.vector_weight
        bw = self.config.bm25_weight
        cw = self.config.consensus_weight
        threshold = self.config.consensus_threshold

        # ── Step 1: 기본 RRF 점수 계산 ──
        rrf_scores: Dict[int, float] = defaultdict(float)

        for rank, idx in enumerate(dense_ranking):
            rrf_scores[idx] += vw / (k + rank + 1)
        for rank, idx in enumerate(bm25_ranking):
            rrf_scores[idx] += bw / (k + rank + 1)

        # ── Step 2: Domain Consensus 계산 ──
        # 동일 위반유형 / 기업명을 가진 청크의 출현 빈도 계산
        violation_counts: Dict[str, int] = defaultdict(int)
        company_counts: Dict[str, int] = defaultdict(int)

        for idx in rrf_scores.keys():
            if idx < len(corpus_info):
                meta = corpus_info[idx].get("metadata", {})
                violation = meta.get("violation", meta.get("위반유형", ""))
                company = meta.get("company", meta.get("피심인기업명", ""))
                if violation:
                    violation_counts[violation] += 1
                if company:
                    company_counts[company] += 1

        # ── Step 3: Consensus 가중치 부여 ──
        # 동일 위반유형이 threshold 이상 → 모든 해당 청크에 가점
        for idx in list(rrf_scores.keys()):
            if idx < len(corpus_info):
                meta = corpus_info[idx].get("metadata", {})
                violation = meta.get("violation", meta.get("위반유형", ""))
                company = meta.get("company", meta.get("피심인기업명", ""))

                # Violation consensus
                if violation and violation_counts.get(violation, 0) >= threshold:
                    rrf_scores[idx] += cw * 0.6  # 위반유형 합의 보너스

                # Company consensus
                if company and company_counts.get(company, 0) >= threshold:
                    rrf_scores[idx] += cw * 0.4  # 기업 합의 보너스

        # ── Step 4: Candidate filter ──
        if candidate_indices is not None:
            rrf_scores = {
                idx: score
                for idx, score in rrf_scores.items()
                if idx in candidate_indices
            }

        return dict(rrf_scores)

    def top_k_chunks(
        self,
        rrf_scores: Dict[int, float],
        corpus_info: List[Dict],
    ) -> List[Tuple[int, float, Dict]]:
        """
        RRF 점수 기준 상위 K개 청크 반환

        Returns:
            [(chunk_index, score, metadata), ...]
        """
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in sorted_items[:self.config.top_k]:
            meta = corpus_info[idx].get("metadata", {}) if idx < len(corpus_info) else {}
            results.append((idx, score, meta))
        return results


# ──────────────────────────────────────────────
# SPV (Simplified Payment Verification) 경량 검증
# ──────────────────────────────────────────────

class SPVVerifier:
    """
    SPV 경량 검증기

    클라이언트는 전체 체인을 다운로드하지 않고,
    chunk_id + hash_value + timestamp만으로 검증 가능.

    Ref: Bitcoin SPV (Nakamoto, 2008)
    """

    def __init__(self, hash_chain: List[HashChainRecord]):
        self._chain = hash_chain
        # 빠른 조회를 위한 인덱스
        self._chunk_map: Dict[str, HashChainRecord] = {
            r.chunk_id: r for r in hash_chain
        }

    def verify_chunk(self, chunk_id: str, provided_hash: str) -> Tuple[bool, str]:
        """
        단일 청크의 무결성을 해시만으로 검증

        Returns:
            (is_verified, message)
        """
        record = self._chunk_map.get(chunk_id)
        if record is None:
            return False, f"Chunk {chunk_id} not found in chain"
        if record.hash_value != provided_hash:
            return False, (
                f"Hash mismatch: provided={provided_hash[:16]}..., "
                f"expected={record.hash_value[:16]}..."
            )
        return True, f"Chunk {chunk_id} verified (timestamp: {record.timestamp})"

    def verify_batch(
        self, chunk_hashes: List[Tuple[str, str]]
    ) -> List[Dict]:
        """
        여러 청크를 일괄 검증

        Args:
            chunk_hashes: [(chunk_id, hash_value), ...]

        Returns:
            [{"chunk_id": ..., "is_verified": bool, "message": str}, ...]
        """
        return [
            {
                "chunk_id": cid,
                "is_verified": self.verify_chunk(cid, hv)[0],
                "message": self.verify_chunk(cid, hv)[1],
            }
            for cid, hv in chunk_hashes
        ]
