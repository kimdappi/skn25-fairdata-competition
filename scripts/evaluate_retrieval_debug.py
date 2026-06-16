from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, cast


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.evaluation.metrics import compute_mrr_metrics, compute_recall_metrics, load_eval_dataset
from app.preprocessing.corpus import load_corpus
from app.retrieval.pipeline import HybridSearchPipeline
from app.retrieval.router import QueryRouter
from app.utils.config import resolve_data_dir, resolve_reranker_weight


@dataclass(frozen=True)
class DebugExperimentConfig:
    name: str
    dense_top_k: int
    sparse_top_k: int
    final_top_k: int
    rerank_weight: float | None


STAGE_NAMES = ("dense", "sparse", "fused", "reranked", "final")


def parse_csv_ints(text: str, *, name: str) -> list[int]:
    raw_items = [item.strip() for item in text.split(",") if item.strip()]
    if not raw_items:
        raise ValueError(f"{name} must not be empty.")
    values = sorted({int(item) for item in raw_items})
    if values[0] <= 0:
        raise ValueError(f"{name} must contain positive integers only.")
    return values


def align_experiment_topks(dense_topks: list[int], sparse_topks: list[int], final_topks: list[int]) -> list[tuple[int, int, int]]:
    lengths = {len(dense_topks), len(sparse_topks), len(final_topks)}
    if len(lengths) == 1:
        return list(zip(dense_topks, sparse_topks, final_topks))

    max_len = max(lengths)
    normalized_lists: list[list[int]] = []
    for values in (dense_topks, sparse_topks, final_topks):
        if len(values) == 1:
            normalized_lists.append(values * max_len)
            continue
        if len(values) != max_len:
            raise ValueError(
                "dense-topks, sparse-topks, final-topks must have the same number of entries "
                "or a single value for broadcasting."
            )
        normalized_lists.append(values)
    return list(zip(*normalized_lists))


def build_experiment_configs(args: argparse.Namespace) -> list[DebugExperimentConfig]:
    dense_topks = parse_csv_ints(args.dense_topks, name="dense-topks")
    sparse_topks = parse_csv_ints(args.sparse_topks, name="sparse-topks")
    final_topks = parse_csv_ints(args.final_topks, name="final-topks")

    experiments: list[DebugExperimentConfig] = []
    for dense_top_k, sparse_top_k, final_top_k in align_experiment_topks(dense_topks, sparse_topks, final_topks):
        base_name = f"dense{dense_top_k}_sparse{sparse_top_k}_final{final_top_k}"
        if args.compare_reranker:
            experiments.append(
                DebugExperimentConfig(
                    name=f"{base_name}_rerank_on",
                    dense_top_k=dense_top_k,
                    sparse_top_k=sparse_top_k,
                    final_top_k=final_top_k,
                    rerank_weight=None,
                )
            )
            experiments.append(
                DebugExperimentConfig(
                    name=f"{base_name}_rerank_off",
                    dense_top_k=dense_top_k,
                    sparse_top_k=sparse_top_k,
                    final_top_k=final_top_k,
                    rerank_weight=0.0,
                )
            )
        else:
            experiments.append(
                DebugExperimentConfig(
                    name=f"{base_name}_rerank_current",
                    dense_top_k=dense_top_k,
                    sparse_top_k=sparse_top_k,
                    final_top_k=final_top_k,
                    rerank_weight=None,
                )
            )
    return experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense/Sparse/Fused/Reranked 단계별 retrieval debug 평가를 수행합니다.")
    parser.add_argument("--dataset-path", required=True, help="평가셋 경로 (.json/.jsonl)")
    parser.add_argument("--output-dir", required=True, help="debug 결과 출력 디렉터리")
    parser.add_argument("--recall-ks", default="5,10,20,30,40,50", help="쉼표 구분 Recall@K 목록")
    parser.add_argument("--mrr-ks", default="5,10,20,30,40,50", help="쉼표 구분 MRR@K 목록")
    parser.add_argument("--dense-topks", default="30,50,70", help="실험별 dense path top-k 목록")
    parser.add_argument("--sparse-topks", default="30,50,70", help="실험별 sparse path top-k 목록")
    parser.add_argument("--final-topks", default="50,75,100", help="실험별 final top-k 목록")
    parser.add_argument("--compare-reranker", action="store_true", help="reranker on/off 짝 실험 생성")
    parser.add_argument("--limit", type=int, default=0, help="평가할 최대 문항 수. 0이면 전체 문항.")
    parser.add_argument("--offset", type=int, default=0, help="앞에서부터 건너뛸 문항 수.")
    parser.add_argument("--save-rows", action="store_true", help="문항별 rows.jsonl 저장")
    parser.add_argument("--tag", default="baseline_debug", help="summary.json의 experiment tag")
    return parser.parse_args()


@contextmanager
def temporary_env(overrides: dict[str, str | None]) -> Iterator[None]:
    original: dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def compute_stage_metrics(
    chunk_ids: list[str],
    gold_chunk_ids: list[str],
    recall_ks: list[int],
    mrr_ks: list[int],
) -> dict[str, dict[str, float]]:
    return {
        "recall_metrics": compute_recall_metrics(chunk_ids, gold_chunk_ids, recall_ks),
        "mrr_metrics": compute_mrr_metrics(chunk_ids, gold_chunk_ids, mrr_ks),
    }


def aggregate_stage_metrics(rows: list[dict[str, Any]], stage_name: str) -> dict[str, dict[str, float]]:
    if not rows:
        return {"recall_metrics": {}, "mrr_metrics": {}}

    recall_keys = list(rows[0]["stage_recalls"][stage_name].keys())
    mrr_keys = list(rows[0]["stage_mrr"][stage_name].keys())
    return {
        "recall_metrics": {
            key: sum(float(row["stage_recalls"][stage_name][key]) for row in rows) / len(rows)
            for key in recall_keys
        },
        "mrr_metrics": {
            key: sum(float(row["stage_mrr"][stage_name][key]) for row in rows) / len(rows)
            for key in mrr_keys
        },
    }


def build_rows_for_experiment(
    *,
    pipeline: HybridSearchPipeline,
    examples: list[Any],
    experiment: DebugExperimentConfig,
    recall_ks: list[int],
    mrr_ks: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        result = pipeline.search_with_trace(example.question, final_top_k=experiment.final_top_k)
        stage_chunk_ids = {
            "dense": cast(list[str], result["dense_chunk_ids"]),
            "sparse": cast(list[str], result["sparse_chunk_ids"]),
            "fused": cast(list[str], result["fused_chunk_ids"]),
            "reranked": cast(list[str], result["reranked_chunk_ids"]),
            "final": cast(list[str], result["final_chunk_ids"]),
        }
        stage_metrics = {
            stage: compute_stage_metrics(stage_chunk_ids[stage], list(example.gold_chunk_ids), recall_ks, mrr_ks)
            for stage in STAGE_NAMES
        }
        row = {
            "id": example.id,
            "question": example.question,
            "gold_chunk_ids": list(example.gold_chunk_ids),
            "experiment": experiment.name,
            "dense_chunk_ids": stage_chunk_ids["dense"],
            "sparse_chunk_ids": stage_chunk_ids["sparse"],
            "fused_chunk_ids": stage_chunk_ids["fused"],
            "reranked_chunk_ids": stage_chunk_ids["reranked"],
            "final_chunk_ids": stage_chunk_ids["final"],
            "stage_recalls": {
                stage: stage_metrics[stage]["recall_metrics"]
                for stage in STAGE_NAMES
            },
            "stage_mrr": {
                stage: stage_metrics[stage]["mrr_metrics"]
                for stage in STAGE_NAMES
            },
        }
        rows.append(row)
        print(
            f"[evaluate_retrieval_debug] {experiment.name} {index}/{len(examples)} id={example.id} "
            f"final={row['final_chunk_ids'][:5]}"
        )
    return rows


def summarize_experiment(
    *,
    rows: list[dict[str, Any]],
    experiment: DebugExperimentConfig,
    effective_rerank_weight: float,
) -> dict[str, Any]:
    return {
        "experiment": experiment.name,
        "config": {
            **asdict(experiment),
            "effective_rerank_weight": effective_rerank_weight,
        },
        "count": len(rows),
        "stage_metrics": {
            stage: aggregate_stage_metrics(rows, stage)
            for stage in STAGE_NAMES
        },
    }


def build_best_experiment_report(experiments: list[dict[str, Any]]) -> dict[str, str | None]:
    def best_by(stage: str, metric_name: str) -> str | None:
        eligible = [
            item for item in experiments
            if metric_name in item["stage_metrics"][stage]["recall_metrics"]
        ]
        if not eligible:
            return None
        best = max(eligible, key=lambda item: item["stage_metrics"][stage]["recall_metrics"][metric_name])
        return str(best["experiment"])

    return {
        "best_fused_recall@50": best_by("fused", "recall@50"),
        "best_final_recall@5": best_by("final", "recall@5"),
    }


def main() -> None:
    args = parse_args()
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    recall_ks = parse_csv_ints(args.recall_ks, name="recall-ks")
    mrr_ks = parse_csv_ints(args.mrr_ks, name="mrr-ks")
    experiments = build_experiment_configs(args)

    dataset_path = Path(args.dataset_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_eval_dataset(dataset_path)
    examples = examples[args.offset :]
    if args.limit > 0:
        examples = examples[: args.limit]

    router = QueryRouter()
    corpus = load_corpus(resolve_data_dir(), router.route_from_text)

    all_rows: list[dict[str, Any]] = []
    experiment_summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        overrides: dict[str, str | None] = {
            "FAIRDATA_DENSE_TOP_K": str(experiment.dense_top_k),
            "FAIRDATA_SPARSE_TOP_K": str(experiment.sparse_top_k),
        }
        if experiment.rerank_weight is not None:
            overrides["FAIRDATA_RERANK_WEIGHT"] = str(experiment.rerank_weight)
        with temporary_env(overrides):
            pipeline = HybridSearchPipeline(corpus, router)
            rows = build_rows_for_experiment(
                pipeline=pipeline,
                examples=examples,
                experiment=experiment,
                recall_ks=recall_ks,
                mrr_ks=mrr_ks,
            )
            all_rows.extend(rows)
            experiment_summaries.append(
                summarize_experiment(
                    rows=rows,
                    experiment=experiment,
                    effective_rerank_weight=resolve_reranker_weight(),
                )
            )

    summary = {
        "tag": args.tag,
        "dataset": str(dataset_path),
        "offset": args.offset,
        "limit": args.limit,
        "recall_ks": recall_ks,
        "mrr_ks": mrr_ks,
        "experiments": experiment_summaries,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best_report = build_best_experiment_report(experiment_summaries)
    best_report_path = output_dir / "best_experiment_report.json"
    best_report_path.write_text(json.dumps(best_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.save_rows:
        rows_path = output_dir / "rows.jsonl"
        with rows_path.open("w", encoding="utf-8") as fp:
            for row in all_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        rows_path = None

    result = {
        "summary_path": str(summary_path),
        "rows_path": str(rows_path) if rows_path is not None else None,
        "best_report_path": str(best_report_path),
        "experiment_count": len(experiment_summaries),
        "example_count": len(examples),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
