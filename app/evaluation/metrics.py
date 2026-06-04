from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ranx import Qrels, Run, evaluate as ranx_evaluate

from app.utils.config import (
    DENSE_BACKEND,
    LLM_BACKEND,
    RERANK_BACKEND,
    RERANK_TOP_N,
    RERANK_WEIGHT,
    resolve_index_namespace,
)


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


# ── ranx 기반 retrieval 지표 ──────────────────────────────────────────────

def _build_ranx_inputs(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """rows 로부터 ranx Qrels, Run 입력 딕셔너리를 생성합니다."""
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for row in rows:
        qid = str(row["id"])
        gold_chunks = row["gold_chunk_ids"]
        pred_chunks = row["predicted_chunk_ids"][:5]

        qrels[qid] = {str(c): 1 for c in gold_chunks}
        # 점수: 순위 기반 감쇠 (1위=1.0, 2위=0.999, ...)
        run[qid] = {str(c): 1.0 - i * 0.001 for i, c in enumerate(pred_chunks)}

    return qrels, run


def compute_recall_at_5(
    predicted_chunk_ids: list[str],
    gold_chunk_ids: tuple[str, ...],
) -> float:
    """표준 Recall@5: |gold ∩ top-5| / |gold|  (ranx 기반)

    gold_chunk_ids 가 비어 있으면 0.0을 반환합니다.
    """
    if not gold_chunk_ids:
        return 0.0

    gold_set = set(str(c) for c in gold_chunk_ids)
    predicted = [str(c) for c in predicted_chunk_ids[:5]]

    qrels = Qrels({"q": {c: 1 for c in gold_set}})
    run = Run({"q": {c: 1.0 - i * 0.001 for i, c in enumerate(predicted)}})
    result = ranx_evaluate(qrels, run, ["recall@5", "mrr@5"])
    return float(result["recall@5"])


def compute_mrr(
    predicted_chunk_ids: list[str],
    gold_chunk_ids: tuple[str, ...],
) -> float:
    """표준 MRR@5: 첫 정답 청크 등장 순위의 역수 (ranx 기반)

    gold_chunk_ids 가 비어 있으면 0.0을 반환합니다.
    """
    if not gold_chunk_ids:
        return 0.0

    gold_set = set(str(c) for c in gold_chunk_ids)
    predicted = [str(c) for c in predicted_chunk_ids[:5]]

    qrels = Qrels({"q": {c: 1 for c in gold_set}})
    run = Run({"q": {c: 1.0 - i * 0.001 for i, c in enumerate(predicted)}})
    result = ranx_evaluate(qrels, run, ["recall@5", "mrr@5"])
    return float(result["mrr@5"])


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

    # ── ranx 배치 평가: Recall@5 + MRR 동시 계산 ──
    qrels_dict, run_dict = _build_ranx_inputs(rows)
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    # 유효한 query만 필터링 (gold_chunk_ids가 비어 있지 않은 row만)
    valid_qids = [k for k, v in qrels_dict.items() if v]
    if not valid_qids:
        recall_at_5 = 0.0
        mrr = 0.0
    else:
        valid_qrels = Qrels({k: qrels_dict[k] for k in valid_qids})
        valid_run = Run({k: run_dict[k] for k in valid_qids})
        ret_results = ranx_evaluate(valid_qrels, valid_run, ["recall@5", "mrr@5"])
        recall_at_5 = float(ret_results["recall@5"])
        mrr = float(ret_results["mrr@5"])

    # ── Token F1 ──
    squad_metric = _get_squad_metric()
    squad_preds: list[dict[str, str]] = []
    squad_refs: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row["id"])
        squad_preds.append({"prediction_text": str(row["predicted_answer"]), "id": qid})
        squad_refs.append({
            "answers": {"text": [str(row["gold_answer"])], "answer_start": [0]},
            "id": qid,
        })
    squad_result = squad_metric.compute(predictions=squad_preds, references=squad_refs)
    token_f1 = float(squad_result["f1"]) / 100.0

    # ── BERTScore ──
    bertscore_f1 = compute_bertscore_f1(
        [str(row["predicted_answer"]) for row in rows],
        [str(row["gold_answer"]) for row in rows],
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
        "experiment_tag": experiment_tag,
        "config_snapshot": normalized_snapshot,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
