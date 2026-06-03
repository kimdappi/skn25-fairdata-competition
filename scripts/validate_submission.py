from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import corpus, generator, retriever


def main() -> None:
    sample_question = "공정거래법 위반행위에 대한 시정명령 내용을 설명해줘."
    chunks = retriever.search(sample_question, top_k=5)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(corpus) > 0, "Corpus is empty"
    assert len(chunk_ids) == 5, "Retriever must return exactly five chunk ids"
    assert len(set(chunk_ids)) == 5, "Chunk ids must be unique"

    answer = generator.generate(sample_question, chunks)
    assert answer.strip(), "Answer must not be empty"

    print("Submission validation passed")
    print(f"Corpus chunks: {len(corpus)}")
    print("Retrieved chunk ids:")
    for chunk_id in chunk_ids:
        print(f"- {chunk_id}")


if __name__ == "__main__":
    main()
