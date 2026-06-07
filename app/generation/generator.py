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
from app.utils.config import resolve_llm_model_dir
from app.utils.schemas import LLMRouteDecision, RouteDecision
from app.utils.text import tokenize_text


FOCUS_LABELS = {"처분", "위법성", "사실관계", "일반"}
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def select_representative_line(content: str) -> str:
    # 본문에서 사람이 읽을 수 있는 첫 유효 문장을 추출합니다.
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return ""


class LLMBackend(Protocol):
    # 생성 백엔드가 서버에 제공해야 하는 최소 인터페이스입니다.
    def route_question(self, question: str) -> RouteDecision: ...

    def generate(self, question: str, chunks: list[Chunk]) -> str: ...


class GroundedCausalLMBackend:
    # causal LM 계열 생성 백엔드가 공유하는 프롬프트/추론 공통부입니다.
    def __init__(self, model_dir: Path | None = None, max_evidence_items: int = 3) -> None:
        # 생성에 사용할 모델 경로, 디바이스, 입력 길이 제한을 초기화합니다.
        self.max_evidence_items = max_evidence_items
        self.model_dir = Path(model_dir) if model_dir is not None else resolve_llm_model_dir()
        preferred_device = os.getenv("FAIRDATA_LLM_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        if preferred_device.startswith("cuda") and not torch.cuda.is_available():
            preferred_device = "cpu"
        self.device = torch.device(preferred_device)
        self.tokenizer = None
        self.model = None
        self.max_input_chars = int(os.getenv("FAIRDATA_GENERATION_MAX_INPUT_CHARS", "3500"))
        self.max_new_tokens = int(os.getenv("FAIRDATA_GENERATION_MAX_NEW_TOKENS", "160"))
        self.route_max_new_tokens = int(os.getenv("FAIRDATA_ROUTE_MAX_NEW_TOKENS", "64"))

    def ensure_runtime(self) -> None:
        # 첫 요청 시점에만 tokenizer/model을 로드해 서버 기동 비용을 줄입니다.
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                f"{self.__class__.__name__}를 사용하려면 transformers 패키지가 필요합니다."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        model_kwargs = {"local_files_only": True}
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)

    def build_context(self, chunks: list[Chunk]) -> str:
        # 상위 청크를 프롬프트에 넣기 좋은 텍스트 블록으로 직렬화합니다.
        sections: list[str] = []
        for chunk in chunks[: self.max_evidence_items]:
            section = "\n".join(
                part
                for part in [
                    f"[청크 ID] {chunk.chunk_id}",
                    f"[문서명] {chunk.doc_name}",
                    f"[헤더] {chunk.header}" if chunk.header else "",
                    f"[섹션] {chunk.section}" if chunk.section else "",
                    f"[본문] {chunk.content}",
                ]
                if part
            )
            sections.append(section)
        return "\n\n".join(sections).strip()[: self.max_input_chars]

    def build_messages(self, question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
        # 채팅 템플릿이 기대하는 system/user 메시지 배열을 구성합니다.
        context = self.build_context(chunks)
        system_prompt = (
            "당신은 공정거래위원회 의결서 질의응답 도우미다. "
            "반드시 제공된 근거 청크 안에서만 답하고, 근거가 부족하면 모른다고 답하라. "
            "최종 답변은 한국어로 작성하고, 핵심 사실만 간결하게 정리하라."
        )
        user_prompt = "\n\n".join(
            [
                f"질문:\n{question}",
                f"근거 청크:\n{context}" if context else "근거 청크:\n없음",
                (
                    "답변 형식:\n"
                    "1. 질문에 대한 직접 답변 2~4문장\n"
                ),
            ]
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def decode_generated_text(self, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
        # 프롬프트 이후에 새로 생성된 completion 부분만 잘라 텍스트로 복원합니다.
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
        # grounded generation 전체 흐름을 실행하고,
        # 생성이 비어 있으면 근거 문장 fallback 응답을 반환합니다.
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
        for chunk in chunks[: self.max_evidence_items]:
            representative_line = select_representative_line(chunk.content)
            if not representative_line:
                continue
            evidence_lines.append(
                f"{representative_line} (근거: {chunk.chunk_id}, 문서: {chunk.doc_name})"
            )
        return "\n".join(evidence_lines).strip() or "생성 결과가 비어 있습니다."


def build_llm_backend() -> LLMBackend:
    # 현재 선택된 로컬 모델 디렉터리를 그대로 사용하는 공통 causal LM backend를 반환합니다.
    return GroundedCausalLMBackend()


class GroundedGenerator(GroundedCausalLMBackend):
    # 기존 import 호환성을 위해 공통 grounded generator 이름을 유지합니다.
    pass
