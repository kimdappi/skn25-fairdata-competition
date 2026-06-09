from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


DEFAULT_PREDICTIONS_PATH = (
    PROJECT_DIR
    / "results"
    / "V2-E4-E5-BM25-ROUTE"
    / "eval_dataset_260505_offset0_limitall"
    / "predictions.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="기존 predictions.jsonl 파일로부터 summary.json을 재계산합니다."
    )
    parser.add_argument(
        "predictions_path",
        nargs="?",
        default=str(DEFAULT_PREDICTIONS_PATH),
        help=f"predictions.jsonl 경로. 기본값: {DEFAULT_PREDICTIONS_PATH}",
    )
    parser.add_argument(
        "--output",
        default="",
        help="summary 저장 경로. 생략하면 predictions 파일과 같은 디렉터리의 summary.json에 저장합니다.",
    )
    parser.add_argument(
        "--experiment-tag",
        default="V2-E4-E5-BM25-ROUTE",
        help="summary에 기록할 experiment_tag.",
    )
    return parser.parse_args()


def load_prediction_rows(predictions_path: Path) -> list[dict[str, Any]]:
    text = predictions_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{predictions_path} must contain a JSON array or JSONL rows")
        return [dict(row) for row in payload]

    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions_path).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else predictions_path.parent / "summary.json"
    )

    # Avoid Hugging Face aborting when fast transfer is enabled without hf_transfer.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    from app.evaluation.metrics import evaluate_predictions

    rows = load_prediction_rows(predictions_path)
    summary = evaluate_predictions(rows, experiment_tag=args.experiment_tag)
    summary["results_dir"] = str(output_path.parents[1])
    summary["run_dir"] = str(output_path.parent)
    summary["predictions_path"] = str(predictions_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
