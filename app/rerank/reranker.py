from __future__ import annotations

from pathlib import Path

import torch

from app.preprocessing.corpus import CorpusStore, Document
from app.retrieval.types import QueryAnalysis, RankedChunk
from app.utils.text import normalize_name


class BGERerankerV2M3:
    # BGE reranker v2 m3 cross-encoder와 메타데이터 보정기를 초기화합니다.
    def __init__(self, corpus_store: CorpusStore, model_dir: Path) -> None:
        self.corpus_store = corpus_store
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    # 실제 재정렬이 호출될 때 reranker 모델을 지연 초기화합니다.
    def ensure_runtime(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "bge-reranker-v2-m3를 사용하려면 transformers 패키지가 필요합니다."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(self.device)

    # 질문에 등장한 회사명과 문서명이 일치하면 추가 가중치를 부여합니다.
    def entity_boost(self, analysis: QueryAnalysis, document: Document) -> float:
        boost = 0.0
        if document.normalized_doc_name and document.normalized_doc_name in analysis.normalized_question:
            boost += 3.0
        for company_name in document.company_names:
            normalized_company = normalize_name(company_name)
            if normalized_company and normalized_company in analysis.normalized_question:
                boost += 2.0
        return boost

    # 라우팅 메타데이터와 문서 메타데이터의 일치 여부를 재정렬 점수에 반영합니다.
    def route_boost(self, analysis: QueryAnalysis, document: Document) -> float:
        boost = 0.0
        if analysis.route.theme != "기타" and analysis.route.theme == document.route.theme:
            boost += 1.5
        if analysis.route.company_size != "기타" and analysis.route.company_size == document.route.company_size:
            boost += 0.8
        if analysis.route.legal_role != "기타" and analysis.route.legal_role == document.route.legal_role:
            boost += 0.8
        if analysis.route.industry != "기타" and analysis.route.industry == document.route.industry:
            boost += 0.8
        return boost

    # reranker 모델이 사용할 text pair 입력을 구성합니다.
    def build_pair_text(self, analysis: QueryAnalysis, chunk_id: str) -> tuple[str, str] | None:
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

    # cross-encoder reranker 점수를 한 번에 계산합니다.
    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
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
        logits = output.logits.view(-1).float().cpu().tolist()
        return [float(score) for score in logits]

    # 융합 점수 상위 후보에 reranker 점수를 더해 최종 순위를 만듭니다.
    def rerank_chunks(
        self,
        analysis: QueryAnalysis,
        fused_scores: dict[str, float],
        *,
        top_n: int = 50,
    ) -> list[RankedChunk]:
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
                "reranker": rerank_map.get(chunk_id, 0.0),
                "entity": self.entity_boost(analysis, document),
                "route": self.route_boost(analysis, document),
            }
            final_score = sum(reason_scores.values())
            reasons = tuple(name for name, score in reason_scores.items() if score > 0)
            ranked_chunks.append(RankedChunk(chunk=chunk, score=final_score, reasons=reasons))
        return sorted(ranked_chunks, key=lambda item: (item.score, item.chunk.chunk_id), reverse=True)
