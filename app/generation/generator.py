from __future__ import annotations

import os
from typing import Protocol

import torch

from app.preprocessing.corpus import Chunk
from app.utils.config import resolve_llm_backend_name, resolve_llm_model_dir


def select_representative_line(content: str) -> str:
    # 본문에서 사람이 읽을 수 있는 첫 유효 문장을 추출합니다.
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return ""


class LLMBackend(Protocol):
    # 생성 백엔드가 서버에 제공해야 하는 최소 인터페이스입니다.
    def generate(self, question: str, chunks: list[Chunk]) -> str: ...


class BaseGroundedCausalLMBackend:
    # causal LM 계열 생성 백엔드가 공유하는 프롬프트/추론 공통부입니다.
    def __init__(self, max_evidence_items: int = 3) -> None:
        # 생성에 사용할 모델 경로, 디바이스, 입력 길이 제한을 초기화합니다.
        self.max_evidence_items = max_evidence_items
        self.model_dir = resolve_llm_model_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.max_input_chars = int(os.getenv("FAIRDATA_GENERATION_MAX_INPUT_CHARS", "6000"))
        self.max_new_tokens = int(os.getenv("FAIRDATA_GENERATION_MAX_NEW_TOKENS", "256"))

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
                    "2. 마지막 줄에 '근거 청크: ...' 형식으로 사용한 chunk_id를 쉼표로 나열"
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


class QwenBackend(BaseGroundedCausalLMBackend):
    # 현재 제출 기준 기본 생성 백엔드입니다.
    pass


class ExaoneBackend(BaseGroundedCausalLMBackend):
    # EXAONE 계열 비교 실험용 생성 백엔드입니다.
    pass


class Llama3Backend(BaseGroundedCausalLMBackend):
    # Llama 계열 비교 실험용 생성 백엔드입니다.
    pass


def build_llm_backend() -> LLMBackend:
    # config에서 선택한 LLM backend 이름을 실제 구현 클래스로 매핑합니다.
    backend_name = resolve_llm_backend_name().strip().lower().replace("-", "").replace("_", "")
    if backend_name == "qwen":
        return QwenBackend()
    if backend_name == "exaone":
        return ExaoneBackend()
    if backend_name in {"llama3", "llama31"}:
        return Llama3Backend()
    raise ValueError(
        "Unsupported LLM backend: "
        f"{resolve_llm_backend_name()}. Supported: qwen, exaone, llama3"
    )


class GroundedGenerator(QwenBackend):
    # 기존 import 호환성을 위해 Qwen 기본 구현 이름을 유지합니다.
    pass
