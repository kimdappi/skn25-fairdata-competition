import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from baseline_rag.config import EVALUATION_MODEL_NAMES, METRICS_PATH, MODEL_DIR, RESULT_DIR, validate_runtime_paths
from baseline_rag.experiments import ExperimentVariant, RetrieverOptions, build_experiment_variants
from baseline_rag.retriever import LegalRAGRetriever

os.environ.setdefault("FAIRCOMP_MODEL_DIR", str(MODEL_DIR))
os.environ.setdefault("OLLAMA_ROUTER_MODEL", "Qwen2.5-7B-Instruct")
os.environ.setdefault("STAGE_LOG_ENABLED", "false")


def mean(items):
    return sum(items) / len(items) if items else 0.0


def run_name_suffix() -> str:
    run_name = os.getenv("EVAL_RUN_NAME", "").strip()
    return f"_{run_name}" if run_name else ""


def recall_at_5(predicted_ids: list[str], gold_set: set[str]) -> float:
    return 1.0 if any(chunk_id in gold_set for chunk_id in predicted_ids[:5]) else 0.0


def reciprocal_rank(predicted_ids: list[str], gold_set: set[str]) -> float:
    for rank, chunk_id in enumerate(predicted_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def load_active_dataset(dataset):
    limit = int(os.getenv("EVAL_LIMIT", "0"))
    sample_size = int(os.getenv("EVAL_SAMPLE_SIZE", "0"))
    sample_seed = int(os.getenv("EVAL_SAMPLE_SEED", "42"))
    active_dataset = dataset[:limit] if limit > 0 else list(dataset)
    if sample_size > 0:
        rng = random.Random(sample_seed)
        active_dataset = rng.sample(active_dataset, min(sample_size, len(active_dataset)))
    return active_dataset, limit, sample_size, sample_seed


def prediction_path_for_variant(variant: ExperimentVariant) -> Path:
    return RESULT_DIR / f"{variant.run_name}_eval_predictions{run_name_suffix()}.json"


def summary_path_for_variant(variant: ExperimentVariant) -> Path:
    return RESULT_DIR / f"{variant.run_name}_eval_summary{run_name_suffix()}.json"


def build_single_variant(model_name: str, options: RetrieverOptions | None = None) -> ExperimentVariant:
    return ExperimentVariant(model_name=model_name, options=(options or RetrieverOptions()).normalized(model_name))


def evaluate_variant(variant: ExperimentVariant, dataset):
    variant = ExperimentVariant(model_name=variant.model_name, options=variant.options.normalized(variant.model_name))
    retriever_started = time.perf_counter()
    retriever = LegalRAGRetriever(embedding_model_name=variant.model_name, options=variant.options)
    prebuild_bge_sparse = os.getenv("EVAL_PREBUILD_BGE_M3_SPARSE", "false").lower() in {"1", "true", "yes", "on"}
    if prebuild_bge_sparse and variant.options.use_bge_sparse:
        retriever.build_bge_m3_sparse_caches()
    retriever_init_seconds = time.perf_counter() - retriever_started

    prediction_path = prediction_path_for_variant(variant)
    resume_enabled = os.getenv("EVAL_RESUME", "true").lower() in {"1", "true", "yes", "on"}
    active_dataset, limit, sample_size, sample_seed = load_active_dataset(dataset)
    recalls = []
    mrrs = []
    predictions = []
    completed_ids = set()

    if resume_enabled and prediction_path.exists():
        with open(prediction_path, "r", encoding="utf-8") as file:
            predictions = json.load(file)
        completed_ids = {item["id"] for item in predictions}
        recalls = [float(item["recall_at_5"]) for item in predictions]
        mrrs = [float(item["mrr"]) for item in predictions]

    pending_rows = [row for row in active_dataset if row["id"] not in completed_ids]
    query_bar = tqdm(
        pending_rows,
        desc=f"Queries {variant.run_name[:60]}",
        ncols=120,
        leave=False,
    )
    for row in query_bar:
        predicted = retriever.search(row["query"])
        predicted_ids = [item.chunk_id for item in predicted]
        gold_ids = [item["chunk_id"] for item in row["answer_chunks"]]
        gold_set = set(gold_ids)

        row_recall_at_5 = recall_at_5(predicted_ids, gold_set)
        row_mrr = reciprocal_rank(predicted_ids, gold_set)

        recalls.append(row_recall_at_5)
        mrrs.append(row_mrr)
        predictions.append(
            {
                "id": row["id"],
                "query": row["query"],
                "source_doc": row["source_doc"],
                "gold_chunk_ids": gold_ids,
                "predicted_chunk_ids": predicted_ids,
                "recall_at_5": row_recall_at_5,
                "mrr": row_mrr,
            }
        )
        query_bar.set_postfix(
            recall_at_5=f"{mean(recalls):.4f}",
            mrr=f"{mean(mrrs):.4f}",
        )
        with open(prediction_path, "w", encoding="utf-8") as file:
            json.dump(predictions, file, ensure_ascii=False, indent=2)

    summary = {
        "created_at": datetime.now().isoformat(),
        "variant": variant.to_metadata(),
        "dataset_size": len(active_dataset),
        "evaluated_queries": len(predictions),
        "retriever_init_seconds": retriever_init_seconds,
        "setup": {
            "ollama_router_model": os.getenv("OLLAMA_ROUTER_MODEL", "qwen3:8b"),
            "eval_resume": resume_enabled,
            "eval_limit": limit,
            "eval_sample_size": sample_size,
            "eval_sample_seed": sample_seed,
        },
        "recall_at_5": mean(recalls),
        "mrr": mean(mrrs),
    }
    return summary, predictions


def evaluate_model(model_name: str, dataset, options: RetrieverOptions | None = None):
    variant = build_single_variant(model_name, options)
    return evaluate_variant(variant, dataset)


def evaluate_variants(variants: Iterable[ExperimentVariant], dataset):
    summaries = []
    variant_list = list(variants)
    variant_bar = tqdm(variant_list, desc="Variants", ncols=140)
    for index, variant in enumerate(variant_bar, start=1):
        started_at = time.perf_counter()
        summary, predictions = evaluate_variant(variant, dataset)
        elapsed = time.perf_counter() - started_at
        summary["elapsed_seconds"] = elapsed
        summaries.append(summary)
        with open(summary_path_for_variant(variant), "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        with open(prediction_path_for_variant(variant), "w", encoding="utf-8") as file:
            json.dump(predictions, file, ensure_ascii=False, indent=2)
        best_mrr = max(item["mrr"] for item in summaries)
        variant_bar.set_postfix(
            done=f"{index}/{len(variant_list)}",
            best_mrr=f"{best_mrr:.4f}",
            last_seconds=f"{elapsed:.1f}",
        )
    return summaries


def filter_variants(variants: list[ExperimentVariant]) -> list[ExperimentVariant]:
    contains = os.getenv("EVAL_VARIANT_CONTAINS", "").strip()
    limit = int(os.getenv("EVAL_VARIANT_LIMIT", "0"))
    filtered = variants
    if contains:
        filtered = [variant for variant in filtered if contains in variant.run_name]
    if limit > 0:
        filtered = filtered[:limit]
    return filtered


def build_readable_summary(summaries: list[dict]) -> dict:
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_model: dict[str, list[dict]] = defaultdict(list)

    for summary in summaries:
        variant = summary["variant"]
        by_group[variant["group_label"]].append(summary)
        by_model[variant["model_name"]].append(summary)

    best_by_group = {}
    for group, items in by_group.items():
        best = max(items, key=lambda item: (item["mrr"], item["recall_at_5"]))
        best_by_group[group] = {
            "short_name": best["variant"]["short_name"],
            "run_name": best["variant"]["run_name"],
            "model_name": best["variant"]["model_name"],
            "mrr": best["mrr"],
            "recall_at_5": best["recall_at_5"],
        }

    best_by_model = {}
    for model_name, items in by_model.items():
        best = max(items, key=lambda item: (item["mrr"], item["recall_at_5"]))
        best_by_model[model_name] = {
            "short_name": best["variant"]["short_name"],
            "group_label": best["variant"]["group_label"],
            "run_name": best["variant"]["run_name"],
            "mrr": best["mrr"],
            "recall_at_5": best["recall_at_5"],
        }

    overall_best = None
    if summaries:
        overall_best = max(summaries, key=lambda item: (item["mrr"], item["recall_at_5"]))

    return {
        "created_at": datetime.now().isoformat(),
        "variant_count": len(summaries),
        "overall_best": {
            "short_name": overall_best["variant"]["short_name"],
            "group_label": overall_best["variant"]["group_label"],
            "model_name": overall_best["variant"]["model_name"],
            "run_name": overall_best["variant"]["run_name"],
            "mrr": overall_best["mrr"],
            "recall_at_5": overall_best["recall_at_5"],
        }
        if overall_best
        else None,
        "best_by_group": best_by_group,
        "best_by_model": best_by_model,
        "summaries": summaries,
    }


def main():
    validate_runtime_paths()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    mode = os.getenv("EVAL_MODE", "variants").strip().lower()
    if mode == "single":
        summaries = []
        for model_name in EVALUATION_MODEL_NAMES:
            summary, predictions = evaluate_model(model_name, dataset)
            summaries.append(summary)
            variant = build_single_variant(model_name)
            with open(summary_path_for_variant(variant), "w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)
            with open(prediction_path_for_variant(variant), "w", encoding="utf-8") as file:
                json.dump(predictions, file, ensure_ascii=False, indent=2)
    else:
        variants = build_experiment_variants(EVALUATION_MODEL_NAMES)
        variants = filter_variants(variants)
        summaries = evaluate_variants(variants, dataset)

    output_path = RESULT_DIR / f"embedding_eval_summary_all{run_name_suffix()}.json"
    readable_output = build_readable_summary(summaries)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(readable_output, file, ensure_ascii=False, indent=2)
    print(json.dumps(readable_output["overall_best"], ensure_ascii=False, indent=2))
    print(json.dumps(readable_output["best_by_group"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
