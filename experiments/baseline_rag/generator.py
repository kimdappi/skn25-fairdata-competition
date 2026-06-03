import os
import re
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline_rag.config import MODEL_DIR
from baseline_rag.retrieval_types import ChunkRecord
from baseline_rag.runtime import preferred_torch_dtype, require_runtime_device

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

DEFAULT_GUIDELINE_PATH = Path("from_uy/example/eval_guideline_260505.md")
DEFAULT_GENERATION_SCRIPT_PATH = Path("from_uy/example/generation_260505.py")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _extract_guideline_prompt(guideline_text: str) -> str | None:
    pattern = re.compile(r"```text\s*(.*?)```", re.DOTALL)
    for block in pattern.findall(guideline_text):
        if "[System]" in block and "Context:" in block and "3~5문장" in block:
            return block.strip()
    return None


def _extract_script_prompt(script_text: str) -> str | None:
    pattern = re.compile(r'prompt\s*=\s*f?"""(.*?)"""', re.DOTALL)
    for block in pattern.findall(script_text):
        if "[System]" in block and "Query:" in block and "Context:" in block:
            return block.strip()
    return None


def load_generation_instruction() -> str:
    guideline_text = _read_text(DEFAULT_GUIDELINE_PATH)
    script_text = _read_text(DEFAULT_GENERATION_SCRIPT_PATH)
    guideline_prompt = _extract_guideline_prompt(guideline_text)
    script_prompt = _extract_script_prompt(script_text)

    instruction_parts = [
        "당신은 공정거래위원회 심사관 역할의 한국어 답변 생성 모델입니다.",
        "반드시 제공된 Context에만 근거해서 답변하세요.",
        "질문에 직접 답하고, 핵심 위반 행위, 법리 판단, 시정조치나 제재를 우선적으로 요약하세요.",
        "답변은 3~5문장 내외의 간결한 한국어 문단으로 작성하세요.",
        "Context에 없는 사실은 추측하지 말고, 확인되지 않으면 문맥상 확인되지 않는다고 답하세요.",
    ]
    if guideline_prompt:
        instruction_parts.append("다음 가이드라인 프롬프트 취지를 따르세요:")
        instruction_parts.append(guideline_prompt)
    if script_prompt:
        instruction_parts.append("다음 예제 생성 스크립트의 프롬프트 취지도 반영하세요:")
        instruction_parts.append(script_prompt)
    return "\n\n".join(instruction_parts)


class LocalGenerator:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = (model_name or os.getenv("FAIRCOMP_GENERATION_MODEL", "Qwen3-8B")).strip()
        self.model_path = self._resolve_model_path(self.model_name)
        self.allow_cpu = os.getenv("FAIRCOMP_GENERATION_ALLOW_CPU", "false").lower() in {"1", "true", "yes", "on"}
        self.device = require_runtime_device()
        self.enabled = self.model_path is not None and (self.device.type == "cuda" or self.allow_cpu)
        self.max_context_chars = int(os.getenv("FAIRCOMP_GENERATION_CONTEXT_CHARS", "7000"))
        self.max_new_tokens = int(os.getenv("FAIRCOMP_GENERATION_MAX_NEW_TOKENS", "320"))
        self.temperature = float(os.getenv("FAIRCOMP_GENERATION_TEMPERATURE", "0"))
        self.system_instruction = load_generation_instruction()
        self.tokenizer = None
        self.model = None

        if self.enabled:
            self._load()

    def generate(self, question: str, chunks: Sequence[ChunkRecord]) -> str | None:
        if not self.enabled or self.model is None or self.tokenizer is None:
            return None

        context = self._build_context(chunks)
        messages = [
            {"role": "system", "content": self.system_instruction},
            {
                "role": "user",
                "content": (
                    f"Query: {question}\n\n"
                    f"Context:\n{context}\n\n"
                    "위 문맥만 근거로 한국어 답변을 작성하세요."
                ),
            },
        ]

        inputs = self._apply_chat_template(messages)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = 0.8
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            generated = self.model.generate(**inputs, **generate_kwargs)
        output_tokens = generated[0][inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        return answer or None

    def _apply_chat_template(self, messages: list[dict]) -> dict[str, torch.Tensor]:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer(rendered, return_tensors="pt")

    def _build_context(self, chunks: Sequence[ChunkRecord]) -> str:
        parts = []
        current_chars = 0
        for index, chunk in enumerate(chunks, start=1):
            header = chunk.header.strip()
            body = chunk.page_content.strip()
            part = f"[Chunk {index} | {chunk.chunk_id} | {header}]\n{body}".strip()
            if current_chars + len(part) > self.max_context_chars and parts:
                break
            parts.append(part)
            current_chars += len(part) + 2
        return "\n\n".join(parts)

    def _load(self) -> None:
        model_kwargs: dict[str, object] = {"local_files_only": True}
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = preferred_torch_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def _resolve_model_path(self, model_name: str) -> Path | None:
        candidates = [
            Path(model_name).expanduser(),
            MODEL_DIR / model_name,
            MODEL_DIR / model_name.replace("/", "--"),
            MODEL_DIR / model_name.replace("/", "_"),
            MODEL_DIR / model_name.replace(":", "-"),
            MODEL_DIR / "Qwen3-8B",
        ]
        seen = set()
        for candidate in candidates:
            normalized = str(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            if candidate.exists():
                return candidate
        return None
