"""
================================================================================
제2회 공정위 AI·데이터 활용 공모전 — 3개 방법론 비교 벤치마크
================================================================================

[벤치마크 대상]
1. Model 1 (baseline): Hybrid RAG (Dense + BM25 + RRF, No Router)
2. Model 2 (langgraph): Hybrid RAG + QueryRouter + LangGraph StateGraph
3. Model 3 (proofrag): Hash Chain + Consensus Weighted RRF

[평가 지표]
- Recall@5: 상위 5개 청크 중 정답 청크 포함 비율
- MRR (Mean Reciprocal Rank): 첫 정답 청크의 역순위 평균
- Latency (ms): 검색 소요 시간
- SPV Verify Rate: 해시체인 검증 성공률 (모형 3 전용)

[출력]
- results/benchmarks/comparison_YYYYMMDD_HHMMSS.csv
- results/benchmarks/comparison_YYYYMMDD_HHMMSS.json

Usage:
    python benchmark_three_models.py
    python benchmark_three_models.py --queries 50  # 50개 쿼리만 샘플링
================================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from baseline_rag.config import (
    DATA_DIR,
    METRICS_PATH,
    RESULT_DIR,
    STAGE_LOG_ENABLED,
)
from baseline_rag.retriever import LegalRAGRetriever
from baseline_rag.experiments import RetrieverOptions
from baseline_rag.router import QueryRouter
from baseline_rag.schemas import RetrievedChunk, RouteDecision

# Optional: ProofRAG components
try:
    from baseline_rag.proof_rag import ConsensusRRFConfig
    from baseline_rag.proof_retriever import ProofRAGConsensusRetriever
    PROOFRAG_AVAILABLE = True
except ImportError:
    PROOFRAG_AVAILABLE = False
    print("[WARN] ProofRAG modules not available; Model 3 will be skipped.")


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_metrics() -> List[Dict]:
    """metrics.json에서 평가 쿼리 로드"""
    if not METRICS_PATH.exists():
        print(f"[ERROR] METRICS_PATH={METRICS_PATH} not found")
        return []
    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("queries", data.get("items", []))
    return []


# ──────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────

def compute_recall_at_k(predicted_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
    """Recall@K: 상위 K개 예측 중 정답 포함 비율"""
    if not ground_truth_ids:
        return 0.0
    top_k = predicted_ids[:k]
    hits = sum(1 for gt in ground_truth_ids if gt in top_k)
    return hits / len(ground_truth_ids)


def compute_mrr(predicted_ids: List[str], ground_truth_ids: List[str]) -> float:
    """Mean Reciprocal Rank: 첫 정답의 역순위"""
    for rank, pred_id in enumerate(predicted_ids, start=1):
        if pred_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


# ──────────────────────────────────────────────
# Benchmark Runner
# ──────────────────────────────────────────────

class BenchmarkRunner:
    """3개 모형 비교 벤치마크 실행기"""

    def __init__(self, sample_size: int = None):
        self.metrics_data = load_metrics()
        if sample_size and sample_size < len(self.metrics_data):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(self.metrics_data), sample_size, replace=False)
            self.metrics_data = [self.metrics_data[i] for i in sorted(indices)]

        self.results: Dict[str, Dict[str, Any]] = {}
        self.retriever_base: LegalRAGRetriever = None
        self.proof_retriever: ProofRAGConsensusRetriever = None

        # 결과 디렉토리
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.benchmark_dir = RESULT_DIR / "benchmarks"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.benchmark_dir / f"comparison_{timestamp}"

        print(f"\n{'='*60}")
        print(f"Benchmark: {len(self.metrics_data)} queries × 3 models")
        print(f"Output: {self.out_path}.csv / .json")
        print(f"{'='*60}\n")

    def _init_models(self):
        """모든 모형 초기화"""
        print("[1/3] Initializing Model 1 (baseline: Hybrid RAG, no router)...")
        t0 = time.perf_counter()
        options_m1 = RetrieverOptions(
            router_mode="off",
            use_dense=True, use_bm25=True,
            use_routing=False, use_route_boost=False,
            use_entity_boost=False, use_chunk_structure_boost=False,
            use_chunk_lexical_score=True, use_doc_rank_boost=False,
        )
        self.retriever_m1 = LegalRAGRetriever(options=options_m1)
        print(f"  Done ({time.perf_counter() - t0:.1f}s)")

        print("[2/3] Initializing Model 2 (langgraph: Hybrid RAG + Router)...")
        t0 = time.perf_counter()
        options_m2 = RetrieverOptions(
            router_mode="rule",
            use_dense=True, use_bm25=True,
            use_routing=True, use_route_boost=True,
            use_entity_boost=True, use_chunk_structure_boost=True,
            use_chunk_lexical_score=True, use_doc_rank_boost=True,
        )
        self.retriever_m2 = LegalRAGRetriever(options=options_m2)
        print(f"  Done ({time.perf_counter() - t0:.1f}s)")

        if PROOFRAG_AVAILABLE:
            print("[3/3] Initializing Model 3 (proofrag: Hash Chain + Consensus RRF)...")
            t0 = time.perf_counter()
            config = ConsensusRRFConfig(
                vector_weight=0.60, bm25_weight=0.25,
                consensus_weight=0.15, top_k=5,
                consensus_threshold=3,
            )
            self.proof_retriever = ProofRAGConsensusRetriever(
                base_retriever=self.retriever_m2,
                consensus_config=config,
            )
            print(f"  Done ({time.perf_counter() - t0:.1f}s)")
        else:
            print("[3/3] Model 3 SKIPPED (ProofRAG not available)")

    def _resolve_route(self, query: str) -> RouteDecision:
        """질문에 대한 라우팅 결정 (rule-based fallback)"""
        router = self.retriever_m2.router
        return router.route_from_text(query)

    def _extract_ground_truth(self, item: Dict) -> Tuple[str, List[str]]:
        """metrics.json 항목에서 query와 ground truth chunk_ids 추출"""
        query = item.get("query", item.get("question", ""))
        gt_ids = []

        # answer_chunks: [{chunk_id: ..., answer_reason: ...}, ...] (metrics.json 실제 형식)
        answer_chunks = item.get("answer_chunks", [])
        if isinstance(answer_chunks, list):
            for ac in answer_chunks:
                if isinstance(ac, dict):
                    cid = ac.get("chunk_id", "")
                    if cid:
                        gt_ids.append(str(cid))
                elif isinstance(ac, str):
                    gt_ids.append(ac)

        # fallback: 다른 키워드 시도
        if not gt_ids:
            for key in ["chunk_ids", "ground_truth", "참고파일일련번호"]:
                val = item.get(key, [])
                if isinstance(val, list):
                    gt_ids = [str(v) for v in val]
                    break
                if isinstance(val, str):
                    gt_ids = [val]
                    break

        # metadata에 chunk_id가 있는 경우
        if not gt_ids and isinstance(item.get("metadata"), dict):
            cid = item["metadata"].get("chunk_id", "")
            if cid:
                gt_ids = [str(cid)]

        return query, gt_ids

    def run_model_1(self, query: str) -> Tuple[List[str], float]:
        """모형 1: Hybrid RAG (No Router)"""
        t0 = time.perf_counter()
        results = self.retriever_m1.search(query)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        return [r.chunk_id for r in results], elapsed

    def run_model_2(self, query: str, route: RouteDecision) -> Tuple[List[str], float]:
        """모형 2: LangGraph RAG (Router + Graph)"""
        t0 = time.perf_counter()
        results = self.retriever_m2.search_with_route(query, route)
        elapsed = (time.perf_counter() - t0) * 1000
        return [r.chunk_id for r in results], elapsed

    def run_model_3(self, query: str, route: RouteDecision) -> Tuple[List[str], float]:
        """모형 3: ProofRAG Consensus RRF"""
        if not PROOFRAG_AVAILABLE or self.proof_retriever is None:
            return [], 0.0
        t0 = time.perf_counter()
        results = self.proof_retriever.search_with_consensus(query, route)
        elapsed = (time.perf_counter() - t0) * 1000
        return [r.chunk_id for r in results], elapsed

    def run(self):
        """전체 벤치마크 실행"""
        self._init_models()

        all_records = []
        total = len(self.metrics_data)

        for idx, item in enumerate(self.metrics_data):
            query, gt_ids = self._extract_ground_truth(item)
            if not query or not gt_ids:
                print(f"  [{idx+1}/{total}] SKIP: no query or ground truth")
                continue

            route = self._resolve_route(query)

            # Run all three models
            pred_m1, lat_m1 = self.run_model_1(query)
            pred_m2, lat_m2 = self.run_model_2(query, route)
            pred_m3, lat_m3 = self.run_model_3(query, route)

            rec5_m1 = compute_recall_at_k(pred_m1, gt_ids)
            rec5_m2 = compute_recall_at_k(pred_m2, gt_ids)
            rec5_m3 = compute_recall_at_k(pred_m3, gt_ids) if pred_m3 else 0.0

            mrr_m1 = compute_mrr(pred_m1, gt_ids)
            mrr_m2 = compute_mrr(pred_m2, gt_ids)
            mrr_m3 = compute_mrr(pred_m3, gt_ids) if pred_m3 else 0.0

            record = {
                "index": idx,
                "query": query[:100],
                "gt_ids": gt_ids,
                "route_focus": route.focus,
                "route_theme": route.theme,
                # Model 1
                "m1_pred": pred_m1[:5],
                "m1_recall5": round(rec5_m1, 4),
                "m1_mrr": round(mrr_m1, 4),
                "m1_latency_ms": round(lat_m1, 1),
                # Model 2
                "m2_pred": pred_m2[:5],
                "m2_recall5": round(rec5_m2, 4),
                "m2_mrr": round(mrr_m2, 4),
                "m2_latency_ms": round(lat_m2, 1),
                # Model 3
                "m3_pred": pred_m3[:5] if pred_m3 else [],
                "m3_recall5": round(rec5_m3, 4),
                "m3_mrr": round(mrr_m3, 4),
                "m3_latency_ms": round(lat_m3, 1),
            }
            all_records.append(record)

            if (idx + 1) % 10 == 0 or idx == total - 1:
                self._print_progress(idx + 1, total, all_records)

        # Aggregate statistics
        self._compute_summary(all_records)
        self._save_results(all_records)

    def _print_progress(self, done: int, total: int, records: List[Dict]):
        n = len(records)
        if n == 0:
            return
        m1_r5 = np.mean([r["m1_recall5"] for r in records])
        m2_r5 = np.mean([r["m2_recall5"] for r in records])
        m3_r5 = np.mean([r["m3_recall5"] for r in records]) if records[0]["m3_recall5"] > 0 else 0.0
        print(
            f"  [{done}/{total}] "
            f"M1-R@5={m1_r5:.3f} | M2-R@5={m2_r5:.3f} | M3-R@5={m3_r5:.3f}"
        )

    def _compute_summary(self, records: List[Dict]):
        """집계 통계"""
        def stats(key: str) -> Dict:
            vals = [r[key] for r in records if r.get(key, 0) > 0 or key.endswith("_ms")]
            if not vals:
                return {"mean": 0, "median": 0, "std": 0}
            return {
                "mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        self.summary = {
            "num_queries": len(records),
            "model_1_baseline": {
                "recall_at_5": stats("m1_recall5"),
                "mrr": stats("m1_mrr"),
                "latency_ms": stats("m1_latency_ms"),
            },
            "model_2_langgraph": {
                "recall_at_5": stats("m2_recall5"),
                "mrr": stats("m2_mrr"),
                "latency_ms": stats("m2_latency_ms"),
            },
            "model_3_proofrag": {
                "recall_at_5": stats("m3_recall5"),
                "mrr": stats("m3_mrr"),
                "latency_ms": stats("m3_latency_ms"),
            },
        }

    def _save_results(self, records: List[Dict]):
        """결과 저장"""
        output = {
            "summary": self.summary,
            "config": {
                "queries": len(records),
                "models": {
                    "m1": "Hybrid RAG (Dense+BM25+RRF, No Router)",
                    "m2": "LangGraph RAG (Dense+BM25+RRF+Router+Boosts)",
                    "m3": "ProofRAG (Hash Chain + Consensus Weighted RRF)",
                },
            },
            "records": records,
        }

        # JSON
        json_path = str(self.out_path) + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # CSV
        csv_path = str(self.out_path) + ".csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            header = (
                "index,query,route_focus,m1_recall5,m1_mrr,m1_latency_ms,"
                "m2_recall5,m2_mrr,m2_latency_ms,"
                "m3_recall5,m3_mrr,m3_latency_ms"
            )
            f.write(header + "\n")
            for r in records:
                line = (
                    f"{r['index']},\"{r['query']}\",{r['route_focus']},"
                    f"{r['m1_recall5']},{r['m1_mrr']},{r['m1_latency_ms']},"
                    f"{r['m2_recall5']},{r['m2_mrr']},{r['m2_latency_ms']},"
                    f"{r['m3_recall5']},{r['m3_mrr']},{r['m3_latency_ms']}"
                )
                f.write(line + "\n")

        print(f"\n{'='*60}")
        print("✅ Benchmark complete!")
        print(f"   JSON: {json_path}")
        print(f"   CSV:  {csv_path}")
        print(f"{'='*60}")
        self._print_summary_table()

    def _print_summary_table(self):
        """결과 요약 테이블 출력"""
        s = self.summary
        print(f"\n{'─'*70}")
        print(f"{'Metric':<20} {'Model 1 (Baseline)':<18} {'Model 2 (LangGraph)':<18} {'Model 3 (ProofRAG)':<18}")
        print(f"{'─'*70}")

        for metric_label, key in [
            ("Recall@5", "recall_at_5"),
            ("MRR", "mrr"),
            ("Latency (ms)", "latency_ms"),
        ]:
            m1 = s["model_1_baseline"][key]
            m2 = s["model_2_langgraph"][key]
            m3 = s["model_3_proofrag"][key]
            print(
                f"{metric_label:<20} "
                f"{m1['mean']:.4f} ±{m1['std']:.2f}    "
                f"{m2['mean']:.4f} ±{m2['std']:.2f}    "
                f"{m3['mean']:.4f} ±{m3['std']:.2f}"
            )
        print(f"{'─'*70}")


# ──────────────────────────────────────────────
# CLI Entry
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3-Model RAG Benchmark")
    parser.add_argument(
        "--queries", type=int, default=None,
        help="Number of queries to sample (default: all)"
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(sample_size=args.queries)
    runner.run()


if __name__ == "__main__":
    main()
