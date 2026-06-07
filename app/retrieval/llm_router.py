from __future__ import annotations

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
from app.utils.config import resolve_llm_model_dir
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
        self.chain = self._build_chain()

    def _build_chain(self):
        try:
            import torch
            from langchain_core.output_parsers import PydanticOutputParser
            from langchain_core.prompts import PromptTemplate
            from langchain_huggingface import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "LLM 라우팅 태그 생성을 사용하려면 langchain-core, "
                "langchain-huggingface, transformers, torch가 필요합니다."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        model_kwargs: dict[str, Any] = {"local_files_only": True}
        device = 0 if torch.cuda.is_available() else -1
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(self.model_dir, **model_kwargs)
        text_generation = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        llm = HuggingFacePipeline(
            pipeline=text_generation,
            batch_size=max(1, self.batch_size),
        )
        parser = PydanticOutputParser(pydantic_object=LLMRouteDecision)
        prompt = PromptTemplate(
            template=(
                "아래 텍스트를 공정위 의결서 검색용 라우팅 태그로 분류하라.\n"
                "반드시 제공된 label 중 하나만 선택하라.\n"
                "모르면 기타 또는 일반을 선택하라.\n\n"
                "{keyword_hints}\n\n"
                "입력 종류: {input_kind}\n"
                "입력 텍스트:\n{text}\n\n"
                "{format_instructions}\n"
                "JSON만 출력하라."
            ),
            input_variables=["input_kind", "text"],
            partial_variables={
                "keyword_hints": build_keyword_hint_text(),
                "format_instructions": parser.get_format_instructions(),
            },
        )
        return prompt | llm | parser

    def tag_batch(
        self,
        inputs: list[dict[str, str]],
        *,
        max_concurrency: int,
    ) -> list[LLMRouteDecision | Exception]:
        return self.chain.batch(
            inputs,
            config={"max_concurrency": max(1, max_concurrency)},
            return_exceptions=True,
        )
