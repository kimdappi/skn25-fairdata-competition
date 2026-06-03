from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.evaluation.metrics import evaluate_predictions, load_eval_dataset
from app.generation.generator import GroundedGenerator
from app.preprocessing.corpus import load_corpus
from app.retrieval.retriever import HybridRetriever
from app.retrieval.router import QueryRouter
from app.utils.config import resolve_data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="가이드 기준 Recall@5, MRR, F1, BERTScore를 로컬 평가셋으로 계산합니다."
    )
    parser.add_argument("--eval-file", required=True, help="JSONL 평가셋 경로")
    parser.add_argument(
        "--output-file",
        default="",
        help="문항별 예측 결과를 JSONL로 저장할 경로",
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
    return parser.parse_args()


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

    router = QueryRouter()
    corpus = load_corpus(resolve_data_dir(), router.route_from_text)
    retriever = HybridRetriever(corpus, router)
    generator = GroundedGenerator()

    rows: list[dict] = []
    for index, example in enumerate(examples, start=1):
        chunks = retriever.search(example.question, top_k=5)
        predicted_chunk_ids = [chunk.chunk_id for chunk in chunks]
        predicted_answer = generator.generate(example.question, chunks)
        rows.append(
            {
                "id": example.id,
                "question": example.question,
                "gold_chunk_ids": list(example.gold_chunk_ids),
                "gold_answer": example.gold_answer,
                "predicted_chunk_ids": predicted_chunk_ids,
                "predicted_answer": predicted_answer,
            }
        )
        print(
            f"[evaluate_local] {index}/{len(examples)} id={example.id} "
            f"retrieved={predicted_chunk_ids}"
        )

    summary = evaluate_predictions(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_file:
        output_path = Path(args.output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[evaluate_local] wrote predictions: {output_path}")


if __name__ == "__main__":
    main()
