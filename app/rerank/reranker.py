from __future__ import annotations

from pathlib import Path

import torch

from app.preprocessing.corpus import CorpusStore, Document
from app.retrieval.types import QueryAnalysis, RankedChunk
from app.utils.text import normalize_name


class TransformerSequenceClassificationReranker:
    # cross-encoder 계열 리랭커 구현이 공유하는 공통 본체입니다.
    def __init__(self, corpus_store: CorpusStore, model_dir: Path) -> None:
        self.corpus_store = corpus_store
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.rerank_weight = 1.0

    def ensure_runtime(self) -> None:
        # 실제 추론이 필요할 때만 tokenizer/model을 로드합니다.
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                f"{self.__class__.__name__}를 사용하려면 transformers 패키지가 필요합니다."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(self.device)

    def entity_boost(self, analysis: QueryAnalysis, document: Document) -> float:
        # 질문에 문서명이나 회사명이 직접 등장하면 soft boost를 더합니다.
        boost = 0.0
        if document.normalized_doc_name and document.normalized_doc_name in analysis.normalized_question:
            boost += 3.0
        for company_name in document.company_names:
            normalized_company = normalize_name(company_name)
            if normalized_company and normalized_company in analysis.normalized_question:
                boost += 2.0
        return boost

    def build_pair_text(self, analysis: QueryAnalysis, chunk_id: str) -> tuple[str, str] | None:
        # cross-encoder 입력용 (query, passage) 쌍을 구성합니다.
        chunk = self.corpus_store.chunk_map.get(chunk_id)
        if chunk is None:
            return None
        passage = "\n".join(
            part
            for part in [
                f"문서명: {chunk.doc_name}",
                f"헤더: {chunk.header}" if chunk.header else "",
                f"섹션: {chunk.section}" if chunk.section else "",
                chunk.content,
            ]
            if part
        )
        return analysis.question, passage

    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        # query-passage 쌍 전체를 한 번에 점수화해 리랭커 raw score를 반환합니다.
        self.ensure_runtime()
        if not pairs:
            return []
        encoded = self.tokenizer(
            [query for query, _ in pairs],
            [passage for _, passage in pairs],
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model(**encoded)
        logits = output.logits
        if logits.ndim > 1:
            logits = logits[:, 0]
        return [float(score) for score in logits.view(-1).float().cpu().tolist()]

    def rerank_chunks(
        self,
        analysis: QueryAnalysis,
        fused_scores: dict[str, float],
        *,
        top_n: int = 50,
    ) -> list[RankedChunk]:
        # fusion 상위 후보만 cross-encoder로 다시 보고,
        # fusion 점수 + reranker 점수 + entity boost를 합쳐 최종 정렬합니다.
        candidate_ids = [
            chunk_id
            for chunk_id, _ in sorted(fused_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:top_n]
        ]
        pairs: list[tuple[str, str]] = []
        valid_ids: list[str] = []
        for chunk_id in candidate_ids:
            pair = self.build_pair_text(analysis, chunk_id)
            if pair is None:
                continue
            valid_ids.append(chunk_id)
            pairs.append(pair)

        rerank_scores = self.score_pairs(pairs)
        rerank_map = dict(zip(valid_ids, rerank_scores))

        ranked_chunks: list[RankedChunk] = []
        for chunk_id, fused_score in fused_scores.items():
            chunk = self.corpus_store.chunk_map.get(chunk_id)
            if chunk is None:
                continue
            document = self.corpus_store.document_map[chunk.doc_id]
            reason_scores = {
                "fusion": fused_score,
                "reranker": rerank_map.get(chunk_id, 0.0) * self.rerank_weight,
                "entity": self.entity_boost(analysis, document),
            }
            final_score = sum(reason_scores.values())
            reasons = tuple(name for name, score in reason_scores.items() if score > 0)
            ranked_chunks.append(RankedChunk(chunk=chunk, score=final_score, reasons=reasons))
        return sorted(ranked_chunks, key=lambda item: (item.score, item.chunk.chunk_id), reverse=True)


class BGERerankerV2M3(TransformerSequenceClassificationReranker):
    # 기존 기본 리랭커를 별도 이름으로 유지합니다.
    pass


class BGERerankerV25Gemma2(TransformerSequenceClassificationReranker):
    # BGE v2.5 gemma2 계열 비교군입니다.
    pass


class JinaRerankerV3(TransformerSequenceClassificationReranker):
    # jina reranker v3 비교군입니다.
    pass


class MiniLMReranker(TransformerSequenceClassificationReranker):
    # MiniLM 계열 경량 리랭커 비교군입니다.
    pass


class KoReranker(TransformerSequenceClassificationReranker):
    # 한국어 특화 리랭커 비교군입니다.
    pass
