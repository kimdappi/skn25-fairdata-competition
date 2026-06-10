from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.retrieval.keywords import (
    ENTERPRISE_SIZE_KEYWORDS,
    FACT_KEYWORDS,
    INDUSTRY_KEYWORDS,
    LAW_KEYWORDS,
    LEGAL_ROLE_KEYWORDS,
    ORDER_KEYWORDS,
    THEME_KEYWORDS,
)
from app.utils.config import is_llm_trust_remote_code_enabled, resolve_llm_model_dir
from app.utils.schemas import LLMRouteDecision


def _format_keyword_map(title: str, keyword_map: dict[str, list[str]], *, include_other: bool = True) -> str:
    lines = [f"[{title}]"]
    for label, keywords in keyword_map.items():
        lines.append(f"- {label}: {', '.join(keywords)}")
    if include_other:
        lines.append("- 기타")
    return "\n".join(lines)


def build_keyword_hint_text() -> str:
    focus_keywords = {
        "처분": ORDER_KEYWORDS,
        "위법성": LAW_KEYWORDS,
        "사실관계": FACT_KEYWORDS,
        "일반": [],
    }
    return "\n\n".join(
        [
            _format_keyword_map("theme", THEME_KEYWORDS),
            _format_keyword_map("company_size", ENTERPRISE_SIZE_KEYWORDS),
            _format_keyword_map("legal_role", LEGAL_ROLE_KEYWORDS),
            _format_keyword_map("industry", INDUSTRY_KEYWORDS),
            _format_keyword_map("focus", focus_keywords, include_other=False),
        ]
    )


class LangChainRouteTagger:
    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        max_new_tokens: int = 256,
        batch_size: int = 2,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else resolve_llm_model_dir()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.keyword_hints = build_keyword_hint_text()
        self.tokenizer = None
        self.model = None
        self.device = None
        self._ensure_runtime()

    def _ensure_runtime(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LLM 라우팅 태그 생성을 사용하려면 transformers, torch가 필요합니다."
            ) from exc

        trust_remote_code = is_llm_trust_remote_code_enabled()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
        }
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            model_kwargs["torch_dtype"] = torch.float16
        else:
            self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)

    def _build_messages(self, input_kind: str, text: str) -> list[dict[str, str]]:
        system_prompt = (
            "당신은 공정위 의결서 검색용 라우팅 태그 분류기다. "
            "반드시 제공된 label 중 하나만 선택하고 JSON만 출력하라."
        )
        user_prompt = "\n\n".join(
            [
                "아래 텍스트를 공정위 의결서 검색용 라우팅 태그로 분류하라.",
                "반드시 제공된 label 중 하나만 선택하라.",
                "모르면 기타 또는 일반을 선택하라.",
                self.keyword_hints,
                f"입력 종류: {input_kind}",
                f"입력 텍스트:\n{text}",
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
                    "}\n"
                    "JSON만 출력하라."
                ),
            ]
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _generate_one(self, input_kind: str, text: str) -> LLMRouteDecision:
        import torch

        messages = self._build_messages(input_kind, text)
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
            prompt_length = int(encoded["input_ids"].shape[-1])
            completion_ids = generated_ids[0][prompt_length:]
            generated_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        finally:
            try:
                del generated_ids
            except UnboundLocalError:
                pass
            del encoded
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        json_start = generated_text.find("{")
        json_end = generated_text.rfind("}")
        if json_start < 0 or json_end < json_start:
            raise ValueError(f"LLM route output does not contain JSON: {generated_text!r}")
        return LLMRouteDecision(**json.loads(generated_text[json_start : json_end + 1]))

    def tag_batch(
        self,
        inputs: list[dict[str, str]],
        *,
        max_concurrency: int,
    ) -> list[LLMRouteDecision | Exception]:
        del max_concurrency
        results: list[LLMRouteDecision | Exception] = []
        for item in inputs:
            try:
                results.append(self._generate_one(item["input_kind"], item["text"]))
            except Exception as exc:
                results.append(exc)
        return results
