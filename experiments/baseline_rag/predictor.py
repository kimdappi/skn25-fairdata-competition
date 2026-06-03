import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from baseline_rag.config import DEFAULT_EMBEDDING_MODEL
from baseline_rag.experiments import RetrieverOptions
from baseline_rag.generator import LocalGenerator
from baseline_rag.retrieval_types import ChunkRecord
from baseline_rag.retrieval_utils import tokenize_text
from baseline_rag.retriever import LegalRAGRetriever


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?<=다)\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts.keys() & ref_counts.keys())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    return 2 * precision * recall / (precision + recall)


@dataclass
class PredictionResult:
    retrieved_chunk_ids: list[str]
    answer: str


class RAGPredictor:
    def __init__(
        self,
        *,
        embedding_model_name: str | None = None,
        generation_model_name: str | None = None,
        options: RetrieverOptions | None = None,
    ) -> None:
        options = options or RetrieverOptions(router_mode="off")
        self.retriever = LegalRAGRetriever(
            embedding_model_name=embedding_model_name or DEFAULT_EMBEDDING_MODEL,
            options=options,
        )
        self.generator = LocalGenerator(model_name=generation_model_name)
        self.fallback_chunk_ids = [chunk.chunk_id for chunk in self.retriever.chunks[:5]]
        self.max_answer_chars = int(os.getenv("FAIRCOMP_MAX_ANSWER_CHARS", "1000"))
        self.max_answer_sentences = int(os.getenv("FAIRCOMP_MAX_ANSWER_SENTENCES", "4"))

    def predict(self, question: str) -> PredictionResult:
        retrieved = self.retriever.search(question)
        retrieved_chunk_ids = self._ensure_exactly_five([item.chunk_id for item in retrieved])
        chunk_records = [self.retriever.chunk_map[chunk_id] for chunk_id in retrieved_chunk_ids if chunk_id in self.retriever.chunk_map]
        answer = self.generator.generate(question, chunk_records) or self._build_extractive_answer(question, chunk_records)
        return PredictionResult(retrieved_chunk_ids=retrieved_chunk_ids, answer=answer)

    def _build_extractive_answer(self, question: str, chunk_records: Sequence[ChunkRecord]) -> str:
        question_tokens = set(tokenize_text(question))
        scored_sentences: list[tuple[float, int, str]] = []
        seen = set()

        for chunk_rank, chunk in enumerate(chunk_records):
            header = _clean_text(chunk.header).lower()
            for sentence_index, sentence in enumerate(_split_sentences(chunk.page_content)):
                normalized = sentence.lower()
                if len(sentence) < 20 or normalized in seen:
                    continue
                seen.add(normalized)
                sentence_tokens = set(tokenize_text(sentence))
                overlap = len(question_tokens & sentence_tokens)
                header_bonus = 0.15 if header and any(token in header for token in question_tokens) else 0.0
                leading_bonus = max(0.0, 0.12 - (sentence_index * 0.02))
                rank_bonus = max(0.0, 0.10 - (chunk_rank * 0.02))
                score = overlap + header_bonus + leading_bonus + rank_bonus
                scored_sentences.append((score, chunk_rank * 100 + sentence_index, sentence))

        scored_sentences.sort(key=lambda item: (-item[0], item[1]))
        selected = []
        total_chars = 0
        for _, _, sentence in scored_sentences:
            if len(selected) >= self.max_answer_sentences:
                break
            if total_chars + len(sentence) > self.max_answer_chars and selected:
                break
            selected.append(sentence)
            total_chars += len(sentence) + 1

        if not selected:
            selected = [
                _clean_text(chunk.page_content)[: self.max_answer_chars]
                for chunk in chunk_records[:2]
                if _clean_text(chunk.page_content)
            ]

        answer = " ".join(selected).strip()
        return answer[: self.max_answer_chars] if answer else "관련 청크를 바탕으로 답변을 생성하지 못했습니다."

    def _ensure_exactly_five(self, chunk_ids: Iterable[str]) -> list[str]:
        unique_ids = []
        seen = set()
        for chunk_id in chunk_ids:
            if chunk_id in seen:
                continue
            unique_ids.append(chunk_id)
            seen.add(chunk_id)
            if len(unique_ids) == 5:
                return unique_ids

        for chunk_id in self.fallback_chunk_ids:
            if chunk_id in seen:
                continue
            unique_ids.append(chunk_id)
            seen.add(chunk_id)
            if len(unique_ids) == 5:
                break
        return unique_ids[:5]
