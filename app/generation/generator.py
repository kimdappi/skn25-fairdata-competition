from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

import torch

from app.preprocessing.corpus import Chunk
from app.retrieval.keywords import (
    ENTERPRISE_SIZE_KEYWORDS,
    INDUSTRY_KEYWORDS,
    LEGAL_ROLE_KEYWORDS,
    THEME_KEYWORDS,
)
from app.retrieval.llm_router import build_keyword_hint_text
from app.retrieval.router import OTHER
from app.utils.config import is_llm_trust_remote_code_enabled, resolve_llm_model_dir
from app.utils.schemas import LLMRouteDecision, RouteDecision
from app.utils.text import tokenize_text


FOCUS_LABELS = {"처분", "위법성", "사실관계", "일반"}
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def select_representative_line(content: str) -> str:
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return ""


class LLMBackend(Protocol):
    def route_question(self, question: str) -> RouteDecision: ...
    def generate(self, question: str, chunks: list[Chunk]) -> str: ...


class GroundedCausalLMBackend:
    def __init__(self, model_dir: Path | None = None, max_evidence_items: int = 10) -> None:
        self.max_evidence_items = max_evidence_items
        self.model_dir = Path(model_dir) if model_dir is not None else resolve_llm_model_dir()
        preferred_device = os.getenv("FAIRDATA_LLM_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        if preferred_device.startswith("cuda") and not torch.cuda.is_available():
            preferred_device = "cpu"
        self.device = torch.device(preferred_device)
        self.tokenizer = None
        self.model = None
        self.max_input_chars = int(os.getenv("FAIRDATA_GENERATION_MAX_INPUT_CHARS", "6000"))
        self.max_new_tokens = int(os.getenv("FAIRDATA_GENERATION_MAX_NEW_TOKENS", "384"))
        self.route_max_new_tokens = int(os.getenv("FAIRDATA_ROUTE_MAX_NEW_TOKENS", "256"))

    def ensure_runtime(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                f"{self.__class__.__name__}를 사용하려면 transformers 패키지가 필요합니다."
            ) from exc

        trust_remote_code = is_llm_trust_remote_code_enabled()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs = {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
        }
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)

    def build_context(self, chunks: list[Chunk]) -> str:
        sections: list[str] = []
        for idx, chunk in enumerate(chunks[:self.max_evidence_items], start=1):
            hierarchy = " > ".join(filter(None, [chunk.header, chunk.section]))
            section = (
                f"[근거 {idx}]\n"
                f"[주제] {hierarchy}\n"
                f"{chunk.content}"
            )
            sections.append(section)
        return "\n\n".join(sections).strip()[:self.max_input_chars]

    def build_messages(self, question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
        context = self.build_context(chunks)
        system_prompt = (
            "당신은 공정거래위원회 의결서 질의응답 도우미다.\n"
            "규칙:\n"
            "1. 반드시 제공된 근거 청크의 표현을 그대로 사용하라. 바꿔 말하지 마라.\n"
            "2. 근거 청크에 없는 내용은 절대 추가하지 마라.\n"
            "3. 질문과 관련된 모든 사실을 누락 없이 포함하라.\n"
            "4. 근거가 없으면 '근거 청크에서 확인되지 않습니다.'라고만 답하라.\n"
            "5. 답변만 출력하고, 이유나 절차는 출력하지 마라."
        )
        user_prompt = "\n\n".join(
            [
                f"질문:\n{question}",
                f"근거 청크:\n{context}" if context else "근거 청크:\n없음",
                "답변 지침:\n"
                    "- 반드시 근거 청크에 있는 정보만 사용하라.\n"
                    "- 질문에 직접적으로 답하라.\n"
                    "- 질문과 관련된 핵심 사실을 빠짐없이 포함하라.\n"
                    "- 근거에 여러 사실이 존재하면 중요한 내용을 모두 포함하라.\n"
                    "- 근거에 없는 내용은 추측하지 말라.\n"
                    "- 근거가 부족하면 '근거 청크에서 확인되지 않습니다.'라고 답하라.\n"
                    "답변 작성 절차:\n"
                    "1. 근거 청크에서 질문과 관련된 사실을 모두 찾는다.\n"
                    "2. 중복되거나 유사한 사실은 통합한다.\n"
                    "3. 질문에 답하기 위해 필요한 핵심 사실이 누락되지 않았는지 확인한다.\n"
                    "4. 최종 답변을 작성한다.\n"
                    "5. 위 절차는 내부적으로만 수행하고, 최종 답변만 출력한다.\n"
            ]
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def decode_generated_text(self, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
        prompt_length = int(input_ids.shape[-1])
        completion_ids = generated_ids[0][prompt_length:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    def build_route_messages(self, question: str) -> list[dict[str, str]]:
        system_prompt = (
            "당신은 공정위 의결서 검색용 질문 라우터다. "
            "반드시 제공된 label 중 하나만 선택하고 JSON만 출력하라."
        )
        user_prompt = "\n\n".join(
            [
                build_keyword_hint_text(),
                f"입력 종류: question\n입력 텍스트:\n{question}",
                (
                    "출력 JSON 스키마:\n"
                    "{\n"
                    '  "theme": "label",\n'
                    '  "company_size": "label",\n'
                    '  "legal_role": "label",\n'
                    '  "industry": "label",\n'
                    '  "focus": "처분|위법성|사실관계|일반",\n'
                    '  "confidence": 0.0,\n'
                    '  "reason": "짧은 이유"\n'
                    "}"
                ),
            ]
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def normalize_route_label(self, value: str, allowed: set[str], default: str = OTHER) -> str:
        return value if value in allowed else default

    def route_question(self, question: str) -> RouteDecision:
        self.ensure_runtime()
        messages = self.build_route_messages(question)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=min(4096, int(getattr(self.tokenizer, "model_max_length", 4096))),
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=self.route_max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                )
            generated_text = self.decode_generated_text(encoded["input_ids"], generated_ids)
        finally:
            try:
                del generated_ids
            except UnboundLocalError:
                pass
            del encoded
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        match = JSON_OBJECT_PATTERN.search(generated_text)
        if match is None:
            raise ValueError(f"LLM route output does not contain JSON: {generated_text!r}")
        parsed = LLMRouteDecision(**json.loads(match.group(0)))
        return RouteDecision(
            theme=self.normalize_route_label(parsed.theme, set(THEME_KEYWORDS) | {OTHER}),
            company_size=self.normalize_route_label(parsed.company_size, set(ENTERPRISE_SIZE_KEYWORDS) | {OTHER}),
            legal_role=self.normalize_route_label(parsed.legal_role, set(LEGAL_ROLE_KEYWORDS) | {OTHER}),
            industry=self.normalize_route_label(parsed.industry, set(INDUSTRY_KEYWORDS) | {OTHER}),
            focus=parsed.focus if parsed.focus in FOCUS_LABELS else "일반",
            keywords=tokenize_text(question)[:20],
        )

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "질문과 직접 연결되는 근거 청크를 찾지 못했습니다."
        self.ensure_runtime()
        messages = self.build_messages(question, chunks)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=min(4096, int(getattr(self.tokenizer, "model_max_length", 4096))),
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                )
            answer = self.decode_generated_text(encoded["input_ids"], generated_ids)
        finally:
            try:
                del generated_ids
            except UnboundLocalError:
                pass
            del encoded
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if answer:
            return answer
        evidence_lines: list[str] = []
        for chunk in chunks[:self.max_evidence_items]:
            representative_line = select_representative_line(chunk.content)
            if not representative_line:
                continue
            evidence_lines.append(
                f"{representative_line} (근거: {chunk.chunk_id}, 문서: {chunk.doc_name})"
            )
        return "\n".join(evidence_lines).strip() or "생성 결과가 비어 있습니다."


def build_llm_backend() -> LLMBackend:
    return GroundedCausalLMBackend()


class GroundedGenerator(GroundedCausalLMBackend):
    pass
