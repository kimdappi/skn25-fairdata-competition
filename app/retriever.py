from __future__ import annotations

from collections import Counter
from typing import Iterable

from app.corpus import Chunk, tokenize


KOREAN_STOPWORDS = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "와",
    "과",
    "도",
    "로",
    "으로",
    "한",
    "하다",
    "대한",
    "관련",
    "무엇",
    "인가",
    "어떤",
    "설명",
    "질문",
}


def _normalize_query(question: str) -> list[str]:
    tokens = tokenize(question)
    return [token for token in tokens if len(token) > 1 and token not in KOREAN_STOPWORDS]


class HybridRetriever:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)
        self.chunk_ids = {chunk.chunk_id for chunk in self.chunks}

    def search(self, question: str, top_k: int = 5) -> list[Chunk]:
        if top_k <= 0:
            return []

        query_tokens = _normalize_query(question)
        if not query_tokens:
            return self.chunks[:top_k]

        query_counter = Counter(query_tokens)
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            chunk_counter = Counter(chunk.tokens)
            overlap = 0.0
            for token, query_weight in query_counter.items():
                if token in chunk_counter:
                    overlap += min(query_weight, chunk_counter[token])

            header_bonus = 0.0
            header_text = f"{chunk.doc_name} {chunk.header} {chunk.section}".lower()
            for token in query_counter:
                if token in header_text:
                    header_bonus += 1.5

            coverage = len(set(query_tokens) & set(chunk.tokens)) / len(set(query_tokens))
            score = overlap + header_bonus + coverage
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (item[0], item[1].chunk_id), reverse=True)
        results: list[Chunk] = []
        seen_ids: set[str] = set()

        for _, chunk in scored:
            if chunk.chunk_id in seen_ids:
                continue
            results.append(chunk)
            seen_ids.add(chunk.chunk_id)
            if len(results) == top_k:
                return results

        for chunk in self.chunks:
            if chunk.chunk_id in seen_ids:
                continue
            results.append(chunk)
            seen_ids.add(chunk.chunk_id)
            if len(results) == top_k:
                break
        return results
