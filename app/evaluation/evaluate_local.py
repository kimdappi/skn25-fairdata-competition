from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.evaluation.metrics import (
    compute_mrr,
    compute_recall_at_5,
    compute_token_f1,
    evaluate_predictions,
    load_eval_dataset,
)
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


def collect_config_snapshot() -> dict[str, Any]:
    """현재 실험 구동 시점의 env 설정을 스냅샷으로 수집합니다."""
    return {
        "experiment_tag": os.getenv("FAIRDATA_EXPERIMENT_TAG", ""),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="server.py의 /predict를 호출해 가이드 기준 로컬 평가를 계산합니다."
    )
    parser.add_argument("--eval-file", required=True, help="JSONL 평가셋 경로")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="호출할 server.py 서버 주소",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="각 /predict 요청 타임아웃(초)",
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_DIR / "results"),
        help="종합 지표와 문항별 예측 결과를 저장할 디렉터리",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="평가할 최대 문항 수. 0이면 전체 문항을 평가합니다.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="앞에서부터 건너뛸 문항 수. 기본값은 0입니다.",
    )
    parser.add_argument(
        "--experiment-tag",
        default=os.getenv("FAIRDATA_EXPERIMENT_TAG", ""),
        help="실험 식별 태그 (env FAIRDATA_EXPERIMENT_TAG 보다 우선).",
    )
    return parser.parse_args()


def build_predictor(base_url: str, timeout: float):
    predict_url = urljoin(base_url.rstrip("/") + "/", "predict")

    def predict(example_id: str, question: str) -> dict[str, Any]:
        payload = json.dumps({"id": example_id, "question": question}, ensure_ascii=False).encode("utf-8")
        request = Request(
            predict_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        data = json.loads(body)
        missing_keys = {"id", "retrieved_chunk_ids", "answer"} - data.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"API response missing keys: {missing}")
        return data

    return predict


def build_result_paths(results_dir: Path, eval_path: Path, offset: int, limit: int) -> tuple[Path, Path, Path]:
    stem = eval_path.stem
    suffix_parts = [f"offset{offset}"]
    suffix_parts.append(f"limit{limit}" if limit > 0 else "limitall")
    run_name = f"{stem}_{'_'.join(suffix_parts)}"
    run_dir = results_dir / run_name
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "predictions.jsonl"
    return run_dir, summary_path, predictions_path


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_file).resolve()
    examples = load_eval_dataset(eval_path)
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    examples = examples[args.offset :]
    if args.limit > 0:
        examples = examples[: args.limit]

    predict = build_predictor(args.base_url, args.timeout)
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir, summary_path, predictions_path = build_result_paths(
        results_dir, eval_path, args.offset, args.limit
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for index, example in enumerate(examples, start=1):
        prediction = predict(example.id, example.question)
        predicted_chunk_ids = [str(chunk_id) for chunk_id in prediction["retrieved_chunk_ids"]]
        predicted_answer = str(prediction["answer"])
        recall_at_5 = compute_recall_at_5(predicted_chunk_ids, example.gold_chunk_ids)
        mrr = compute_mrr(predicted_chunk_ids, example.gold_chunk_ids)
        token_f1 = compute_token_f1(predicted_answer, example.gold_answer)
        rows.append(
            {
                "id": example.id,
                "question": example.question,
                "gold_chunk_ids": list(example.gold_chunk_ids),
                "gold_answer": example.gold_answer,
                "predicted_chunk_ids": predicted_chunk_ids,
                "predicted_answer": predicted_answer,
                "recall_at_5": recall_at_5,
                "mrr": mrr,
                "token_f1": token_f1,
            }
        )
        print(
            f"[evaluate_local] {index}/{len(examples)} id={example.id} "
            f"retrieved={predicted_chunk_ids}"
        )

    # ── 실험 설정 스냅샷 수집 + experiment_tag 주입 ──────────────────
    experiment_tag = args.experiment_tag or os.getenv("FAIRDATA_EXPERIMENT_TAG", "")
    config_snapshot = collect_config_snapshot()
    config_snapshot["eval_file"] = str(eval_path)
    config_snapshot["base_url"] = args.base_url

    with predictions_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = evaluate_predictions(
        rows,
        experiment_tag=experiment_tag,
        config_snapshot=config_snapshot,
    )
    summary["eval_file"] = str(eval_path)
    summary["base_url"] = args.base_url
    summary["offset"] = args.offset
    summary["limit"] = args.limit
    summary["results_dir"] = str(results_dir)
    summary["run_dir"] = str(run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[evaluate_local] wrote summary: {summary_path}")
    print(f"[evaluate_local] wrote predictions: {predictions_path}")


if __name__ == "__main__":
    main()
