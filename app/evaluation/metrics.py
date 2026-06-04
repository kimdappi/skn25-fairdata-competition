from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.utils.config import (
    DENSE_BACKEND,
    LLM_BACKEND,
    RERANK_BACKEND,
    RERANK_TOP_N,
    RERANK_WEIGHT,
    resolve_index_namespace,
)


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w가-힣]+", re.UNICODE)

# ── config_snapshot 정규화: 재현성에 필요한 키만 필터링 ──────────────────
_SNAPSHOT_KEYS: frozenset[str] = frozenset({
    "experiment_tag",
    "embed_backend",
    "enable_dense",
    "enable_sparse",
    "enable_multivector",
    "index_namespace",
    "rerank_backend",
    "rerank_top_n",
    "rerank_weight",
    "llm_backend",
    "llm_model_dir",
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
        "embed_backend": DENSE_BACKEND,
        "enable_dense": True,
        "enable_sparse": True,
        "enable_multivector": True,
        "index_namespace": resolve_index_namespace(),
        "rerank_backend": RERANK_BACKEND,
        "rerank_top_n": RERANK_TOP_N,
        "rerank_weight": RERANK_WEIGHT,
        "llm_backend": LLM_BACKEND,
    }


@dataclass(frozen=True)
class EvalExample:
    id: str
    question: str
    gold_chunk_ids: tuple[str, ...]
    gold_answer: str


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = _PUNCT_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", lowered).strip()


def tokenize_for_f1(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def compute_recall_at_5(predicted_chunk_ids: list[str], gold_chunk_ids: tuple[str, ...]) -> float:
    top5 = predicted_chunk_ids[:5]
    if len(top5) != 5 or len(set(top5)) != 5:
        return 0.0
    if not gold_chunk_ids:
        return 0.0
    return 1.0 if any(chunk_id in gold_chunk_ids for chunk_id in top5) else 0.0


def compute_mrr(predicted_chunk_ids: list[str], gold_chunk_ids: tuple[str, ...]) -> float:
    top5 = predicted_chunk_ids[:5]
    if len(top5) != 5 or len(set(top5)) != 5:
        return 0.0
    gold_set = set(gold_chunk_ids)
    for rank, chunk_id in enumerate(top5, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def compute_token_f1(predicted_answer: str, gold_answer: str) -> float:
    pred_tokens = tokenize_for_f1(predicted_answer)
    gold_tokens = tokenize_for_f1(gold_answer)
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bertscore_f1(predictions: list[str], references: list[str]) -> float | None:
    try:
        from bert_score import score as bert_score
    except ImportError:
        return None

    if not predictions:
        return 0.0

    _, _, f1 = bert_score(
        predictions,
        references,
        lang="ko",
        verbose=False,
    )
    return float(f1.mean().item())


def compute_final_score(recall_at_5: float, mrr: float, bertscore_f1: float, token_f1: float) -> float:
    return (0.35 * recall_at_5) + (0.15 * mrr) + (0.30 * bertscore_f1) + (0.20 * token_f1)


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


def evaluate_predictions(
    rows: list[dict[str, Any]],
    *,
    experiment_tag: str = "",
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """rows 를 평가하고 종합 지표 + 실험 설정 스냅샷을 반환합니다.

    Parameters
    ----------
    rows:
        각 row 는 ``predicted_chunk_ids``, ``gold_chunk_ids``,
        ``predicted_answer``, ``gold_answer`` 키를 포함해야 합니다.
    experiment_tag:
        실험 식별 태그 (예: ``"E3_e5_dense"``).
    config_snapshot:
        실험에 사용한 설정 딕셔너리. 생략하거나 ``None`` 이면
        ``config.py`` 의 현재 값으로 자동 구성합니다.
    """
    normalized_snapshot = _normalize_snapshot(config_snapshot)
    if not normalized_snapshot:
        normalized_snapshot = _auto_config_snapshot()

    if not rows:
        return {
            "count": 0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "token_f1": 0.0,
            "bertscore_f1": None,
            "final_score": None,
            "bertscore_available": False,
            "experiment_tag": experiment_tag,
            "config_snapshot": normalized_snapshot,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    recall_scores = [
        compute_recall_at_5(row["predicted_chunk_ids"], row["gold_chunk_ids"])
        for row in rows
    ]
    mrr_scores = [
        compute_mrr(row["predicted_chunk_ids"], row["gold_chunk_ids"])
        for row in rows
    ]
    token_f1_scores = [
        compute_token_f1(row["predicted_answer"], row["gold_answer"])
        for row in rows
    ]
    bertscore_f1 = compute_bertscore_f1(
        [str(row["predicted_answer"]) for row in rows],
        [str(row["gold_answer"]) for row in rows],
    )

    recall_at_5 = sum(recall_scores) / len(recall_scores)
    mrr = sum(mrr_scores) / len(mrr_scores)
    token_f1 = sum(token_f1_scores) / len(token_f1_scores)
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
        "experiment_tag": experiment_tag,
        "config_snapshot": normalized_snapshot,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
