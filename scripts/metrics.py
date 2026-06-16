from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ranx import Qrels, Run, evaluate as ranx_evaluate

from app.utils.config import (
    is_dense_enabled,
    is_multivector_enabled,
    use_multivector_only_for_qm,
    is_sparse_enabled,
    resolve_bge_reranker_model_dir,
    resolve_dense_backend_name,
    resolve_dense_model_dir,
    resolve_index_namespace,
    resolve_llm_backend_name,
    resolve_llm_model_dir,
    resolve_multivector_backend_name,
    resolve_multivector_weight,
    resolve_multivector_model_dir,
    resolve_reranker_backend_name,
    resolve_reranker_top_n,
    resolve_reranker_weight,
    resolve_retrieval_profile,
    resolve_sparse_backend_name,
    resolve_sparse_backend_kind,
    resolve_sparse_model_dir,
)


# ── config_snapshot 정규화: 재현성에 필요한 키만 필터링 ──────────────────
_SNAPSHOT_KEYS: frozenset[str] = frozenset({
    "experiment_tag",
    "embed_backend",
    "dense_model_dir",
    "sparse_backend",
    "sparse_backend_kind",
    "sparse_model_dir",
    "multivector_backend",
    "multivector_model_dir",
    "enable_dense",
    "enable_sparse",
    "enable_multivector",
    "retrieval_profile",
    "index_namespace",
    "rerank_backend",
    "rerank_model_dir",
    "rerank_top_n",
    "rerank_weight",
    "llm_backend",
    "llm_model_dir",
    "predict_io_contract",
    "max_new_tokens",
    "prompt_version",
    "eval_file",
    "base_url",
})


def _normalize_snapshot(raw: dict[str, Any] | None) -> dict[str, Any]:
    """실험 설정 중 재현성에 필요한 키만 필터링하여 반환합니다."""
    if raw is None:
        return {}
    return {k: v for k, v in raw.items() if k in _SNAPSHOT_KEYS}


def _auto_config_snapshot() -> dict[str, Any]:
    """config.py 의 현재 값으로 config_snapshot 을 자동 구성합니다."""
    return {
        "embed_backend": resolve_dense_backend_name(),
        "dense_model_dir": str(resolve_dense_model_dir()),
        "sparse_backend": resolve_sparse_backend_name(),
        "sparse_backend_kind": resolve_sparse_backend_kind(),
        "sparse_model_dir": str(resolve_sparse_model_dir()),
        "multivector_backend": resolve_multivector_backend_name(),
        "multivector_model_dir": str(resolve_multivector_model_dir()),
        "enable_dense": is_dense_enabled(),
        "enable_sparse": is_sparse_enabled(),
        "enable_multivector": is_multivector_enabled(),
        "multivector_weight": resolve_multivector_weight(),
        "multivector_qm_only": use_multivector_only_for_qm(),
        "retrieval_profile": resolve_retrieval_profile(),
        "index_namespace": resolve_index_namespace(),
        "rerank_backend": resolve_reranker_backend_name(),
        "rerank_model_dir": str(resolve_bge_reranker_model_dir()),
        "rerank_top_n": resolve_reranker_top_n(),
        "rerank_weight": resolve_reranker_weight(),
        "llm_backend": resolve_llm_backend_name(),
        "llm_model_dir": str(resolve_llm_model_dir()),
        "predict_io_contract": "id,retrieved_chunk_ids,answer",
    }


@dataclass(frozen=True)
class EvalExample:
    id: str
    question: str
    gold_chunk_ids: tuple[str, ...]
    gold_answer: str


# ── ranx 기반 retrieval 지표 ──────────────────────────────────────────────

def unique_preserve_order(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = str(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _normalize_metric_ks(ks: list[int] | None, *, default: list[int]) -> list[int]:
    values = default if ks is None else ks
    normalized = sorted({int(value) for value in values})
    if not normalized:
        raise ValueError("Metric ks must not be empty.")
    if normalized[0] <= 0:
        raise ValueError("Metric ks must be positive integers.")
    return normalized


def _normalize_predicted_chunk_ids(predicted_chunk_ids: list[str], *, k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    return unique_preserve_order(predicted_chunk_ids)[:k]


def _normalize_gold_chunk_ids(gold_chunk_ids: list[str] | tuple[str, ...]) -> list[str]:
    return unique_preserve_order(gold_chunk_ids)


def _build_ranx_inputs(
    rows: list[dict[str, Any]],
    max_k: int,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """rows 로부터 ranx Qrels, Run 입력 딕셔너리를 생성합니다."""
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for row in rows:
        qid = str(row["id"])
        gold_chunks = _normalize_gold_chunk_ids(row["gold_chunk_ids"])
        pred_chunks = _normalize_predicted_chunk_ids(row["predicted_chunk_ids"], k=max_k)

        qrels[qid] = {str(c): 1 for c in gold_chunks}
        run[qid] = {str(c): 1.0 - i * 0.001 for i, c in enumerate(pred_chunks)}

    return qrels, run


def compute_recall_at_k(
    predicted_chunk_ids: list[str],
    gold_chunk_ids: list[str] | tuple[str, ...],
    k: int,
) -> float:
    gold = _normalize_gold_chunk_ids(gold_chunk_ids)
    if not gold:
        return 0.0
    predicted = _normalize_predicted_chunk_ids(predicted_chunk_ids, k=k)
    if not predicted:
        return 0.0
    gold_set = set(gold)
    hit_count = sum(1 for chunk_id in predicted if chunk_id in gold_set)
    return hit_count / len(gold)


def compute_mrr_at_k(
    predicted_chunk_ids: list[str],
    gold_chunk_ids: list[str] | tuple[str, ...],
    k: int,
) -> float:
    gold = _normalize_gold_chunk_ids(gold_chunk_ids)
    if not gold:
        return 0.0
    predicted = _normalize_predicted_chunk_ids(predicted_chunk_ids, k=k)
    gold_set = set(gold)
    for rank, chunk_id in enumerate(predicted, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def compute_recall_metrics(
    predicted_chunk_ids: list[str],
    gold_chunk_ids: list[str] | tuple[str, ...],
    ks: list[int],
) -> dict[str, float]:
    normalized_ks = _normalize_metric_ks(ks, default=[5])
    return {
        f"recall@{k}": compute_recall_at_k(predicted_chunk_ids, gold_chunk_ids, k)
        for k in normalized_ks
    }


def compute_mrr_metrics(
    predicted_chunk_ids: list[str],
    gold_chunk_ids: list[str] | tuple[str, ...],
    ks: list[int],
) -> dict[str, float]:
    normalized_ks = _normalize_metric_ks(ks, default=[5])
    return {
        f"mrr@{k}": compute_mrr_at_k(predicted_chunk_ids, gold_chunk_ids, k)
        for k in normalized_ks
    }


def compute_recall_at_5(
    predicted_chunk_ids: list[str],
    gold_chunk_ids: tuple[str, ...],
) -> float:
    """표준 Recall@5: |gold ∩ top-5| / |gold|."""
    return compute_recall_at_k(predicted_chunk_ids, gold_chunk_ids, 5)


def compute_mrr(
    predicted_chunk_ids: list[str],
    gold_chunk_ids: tuple[str, ...],
) -> float:
    """표준 MRR@5: 첫 정답 청크 등장 순위의 역수."""
    return compute_mrr_at_k(predicted_chunk_ids, gold_chunk_ids, 5)


# ── HF evaluate SQuAD 기반 token F1 ───────────────────────────────────────

_squad_metric: Any = None


def _get_squad_metric() -> Any:
    """HF evaluate SQuAD metric 인스턴스를 지연 로딩합니다."""
    global _squad_metric
    if _squad_metric is None:
        from evaluate import load as hf_load

        _squad_metric = hf_load("squad")
    return _squad_metric


def compute_token_f1(predicted_answer: str, gold_answer: str) -> float:
    """HF evaluate SQuAD metric 기반 토큰 F1 점수 (0.0 ~ 1.0)

    SQuAD metric이 반환하는 백분율(0~100)을 0~1 범위로 정규화합니다.
    """
    if not predicted_answer.strip() or not gold_answer.strip():
        return 0.0

    metric = _get_squad_metric()
    ref = {
        "answers": {"text": [gold_answer], "answer_start": [0]},
        "id": "__per_row__",
    }
    pred = {"prediction_text": predicted_answer, "id": "__per_row__"}
    result = metric.compute(predictions=[pred], references=[ref])
    return float(result["f1"]) / 100.0


# ── BERTScore ─────────────────────────────────────────────────────────────

def compute_bertscore_f1(
    predictions: list[str],
    references: list[str],
    *,
    device: str | None = None,
) -> float | None:
    try:
        import torch
        from bert_score import score as bert_score
    except ImportError:
        return None

    if not predictions:
        return 0.0

    preferred_device = device or os.getenv(
        "FAIRDATA_BERTSCORE_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    try:
        _, _, f1 = bert_score(
            predictions,
            references,
            lang="ko",
            verbose=False,
            device=preferred_device,
        )
    except torch.OutOfMemoryError:
        if preferred_device == "cpu":
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _, _, f1 = bert_score(
            predictions,
            references,
            lang="ko",
            verbose=False,
            device="cpu",
        )
    return float(f1.mean().item())


# ── 종합 점수 ────────────────────────────────────────────────────────────

def compute_final_score(recall_at_5: float, mrr: float, bertscore_f1: float, token_f1: float) -> float:
    return (0.35 * recall_at_5) + (0.15 * mrr) + (0.30 * bertscore_f1) + (0.20 * token_f1)


# ── 데이터셋 로더 ────────────────────────────────────────────────────────

def load_eval_dataset(dataset_path: Path) -> list[EvalExample]:
    text = dataset_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if dataset_path.suffix.lower() == ".json":
        return load_eval_dataset_from_json(dataset_path, text)
    return load_eval_dataset_from_jsonl(dataset_path, text)


def load_eval_dataset_from_jsonl(dataset_path: Path, text: str) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        missing_keys = {"id", "question", "gold_chunk_ids", "gold_answer"} - payload.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"{dataset_path}:{line_number} missing keys: {missing}")
        examples.append(
            EvalExample(
                id=str(payload["id"]),
                question=str(payload["question"]),
                gold_chunk_ids=tuple(str(item) for item in payload["gold_chunk_ids"]),
                gold_answer=str(payload["gold_answer"]),
            )
        )
    return examples


def load_eval_dataset_from_json(dataset_path: Path, text: str) -> list[EvalExample]:
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"{dataset_path} must contain a JSON array")

    examples: list[EvalExample] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{dataset_path}:{index} item must be an object")

        if {"id", "question", "gold_chunk_ids", "gold_answer"} <= item.keys():
            examples.append(
                EvalExample(
                    id=str(item["id"]),
                    question=str(item["question"]),
                    gold_chunk_ids=tuple(str(chunk_id) for chunk_id in item["gold_chunk_ids"]),
                    gold_answer=str(item["gold_answer"]),
                )
            )
            continue

        missing_keys = {"id", "query", "answer_chunks", "reference_answer"} - item.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"{dataset_path}:{index} missing keys: {missing}")

        answer_chunks = item["answer_chunks"]
        if not isinstance(answer_chunks, list):
            raise ValueError(f"{dataset_path}:{index} answer_chunks must be a list")

        gold_chunk_ids = []
        for chunk in answer_chunks:
            if not isinstance(chunk, dict) or "chunk_id" not in chunk:
                raise ValueError(f"{dataset_path}:{index} answer_chunks entries must include chunk_id")
            gold_chunk_ids.append(str(chunk["chunk_id"]))

        examples.append(
            EvalExample(
                id=str(item["id"]),
                question=str(item["query"]),
                gold_chunk_ids=tuple(gold_chunk_ids),
                gold_answer=str(item["reference_answer"]),
            )
        )
    return examples


# ── 평가 실행 ────────────────────────────────────────────────────────────

def evaluate_predictions(
    rows: list[dict[str, Any]],
    *,
    experiment_tag: str = "",
    config_snapshot: dict[str, Any] | None = None,
    bertscore_device: str | None = None,
    recall_ks: list[int] | None = None,
    mrr_ks: list[int] | None = None,
) -> dict[str, Any]:
    """rows 를 평가하고 종합 지표 + 실험 설정 스냅샷을 반환합니다."""
    normalized_snapshot = _normalize_snapshot(config_snapshot)
    if not normalized_snapshot:
        normalized_snapshot = _auto_config_snapshot()

    recall_ks_normalized = sorted(set(_normalize_metric_ks(recall_ks, default=[5]) + [5]))
    mrr_ks_normalized = sorted(set(_normalize_metric_ks(mrr_ks, default=[5]) + [5]))

    if not rows:
        return {
            "count": 0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "token_f1": 0.0,
            "bertscore_f1": None,
            "final_score": None,
            "bertscore_available": False,
            "recall_metrics": {f"recall@{k}": 0.0 for k in recall_ks_normalized},
            "mrr_metrics": {f"mrr@{k}": 0.0 for k in mrr_ks_normalized},
            "experiment_tag": experiment_tag,
            "config_snapshot": normalized_snapshot,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    max_k = max(recall_ks_normalized + mrr_ks_normalized)
    qrels_dict, run_dict = _build_ranx_inputs(rows, max_k=max_k)

    valid_qids = [qid for qid, qrels in qrels_dict.items() if qrels]
    recall_metrics = {f"recall@{k}": 0.0 for k in recall_ks_normalized}
    mrr_metrics = {f"mrr@{k}": 0.0 for k in mrr_ks_normalized}
    if valid_qids:
        valid_qrels = Qrels({qid: qrels_dict[qid] for qid in valid_qids})
        valid_run = Run({qid: run_dict[qid] for qid in valid_qids})
        metric_names = [*[f"recall@{k}" for k in recall_ks_normalized], *[f"mrr@{k}" for k in mrr_ks_normalized]]
        ret_results = ranx_evaluate(valid_qrels, valid_run, metric_names)
        recall_metrics = {f"recall@{k}": float(ret_results[f"recall@{k}"]) for k in recall_ks_normalized}
        mrr_metrics = {f"mrr@{k}": float(ret_results[f"mrr@{k}"]) for k in mrr_ks_normalized}

    recall_at_5 = recall_metrics["recall@5"]
    mrr = mrr_metrics["mrr@5"]

    squad_metric = _get_squad_metric()
    squad_preds: list[dict[str, str]] = []
    squad_refs: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row["id"])
        squad_preds.append({"prediction_text": str(row["predicted_answer"]), "id": qid})
        squad_refs.append(
            {
                "answers": {"text": [str(row["gold_answer"])], "answer_start": [0]},
                "id": qid,
            }
        )
    squad_result = squad_metric.compute(predictions=squad_preds, references=squad_refs)
    token_f1 = float(squad_result["f1"]) / 100.0

    bertscore_f1 = compute_bertscore_f1(
        [str(row["predicted_answer"]) for row in rows],
        [str(row["gold_answer"]) for row in rows],
        device=bertscore_device,
    )

    final_score = (
        compute_final_score(recall_at_5, mrr, bertscore_f1, token_f1)
        if bertscore_f1 is not None
        else None
    )

    return {
        "count": len(rows),
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "token_f1": token_f1,
        "bertscore_f1": bertscore_f1,
        "final_score": final_score,
        "bertscore_available": bertscore_f1 is not None,
        "recall_metrics": recall_metrics,
        "mrr_metrics": mrr_metrics,
        "experiment_tag": experiment_tag,
        "config_snapshot": normalized_snapshot,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
