from __future__ import annotations

import os

import torch

from app.preprocessing.corpus import Chunk
from app.utils.config import resolve_qwen_model_dir


# 본문에서 사람이 읽을 수 있는 첫 유효 문장을 추출합니다.
def select_representative_line(content: str) -> str:
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return ""


class GroundedGenerator:
    # 검색된 청크를 기반으로 근거 중심 답변 생성기를 초기화합니다.
    def __init__(self, max_evidence_items: int = 3) -> None:
        self.max_evidence_items = max_evidence_items
        self.model_dir = resolve_qwen_model_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.max_input_chars = int(os.getenv("FAIRDATA_GENERATION_MAX_INPUT_CHARS", "6000"))
        self.max_new_tokens = int(os.getenv("FAIRDATA_GENERATION_MAX_NEW_TOKENS", "256"))

    # 실제 생성이 호출될 때 Qwen 모델을 지연 초기화합니다.
    def ensure_runtime(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Qwen 생성 모델을 사용하려면 transformers 패키지가 필요합니다.") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)

        model_kwargs = {"local_files_only": True}
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)

    # 검색 청크를 Qwen 입력용 근거 텍스트로 정리합니다.
    def build_context(self, chunks: list[Chunk]) -> str:
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

        context = "\n\n".join(sections).strip()
        return context[: self.max_input_chars]

    # Qwen chat template용 메시지를 구성합니다.
    def build_messages(self, question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
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

    # 모델 출력에서 프롬프트 이후 생성 텍스트만 추출합니다.
    def decode_generated_text(self, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
        prompt_length = int(input_ids.shape[-1])
        completion_ids = generated_ids[0][prompt_length:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    # 상위 청크 근거를 바탕으로 간결한 답변 문장을 구성합니다.
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
