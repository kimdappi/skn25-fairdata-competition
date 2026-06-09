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


def build_lcel_route_instructions(prompt_variant: str) -> str:
    base_rules = (
        "분류 대상은 공정위 의결서 검색용 라우팅 태그다.\n"
        "각 필드는 반드시 제공된 label 중 하나만 선택하라.\n"
        "근거가 약하면 theme/company_size/legal_role/industry는 '기타', focus는 '일반'을 선택하라.\n"
        "추론한 내용을 새 label로 만들지 말고, JSON 외 텍스트를 출력하지 말라."
    )
    if prompt_variant != "lcel_prompt_boost":
        return base_rules

    return "\n\n".join(
        [
            base_rules,
            (
                "추가 규칙:\n"
                "1. 질문/문서의 핵심 위반 유형이 직접 드러나면 theme를 우선 결정하라.\n"
                "2. focus는 처분 관련 표현(시정명령, 과징금, 고발, 경고 등)이 있으면 '처분',\n"
                "   위법 여부/법조항/성립 판단이 핵심이면 '위법성',\n"
                "   사실 경위/행위 내용/배경 설명이 핵심이면 '사실관계',\n"
                "   그 외는 '일반'으로 분류하라.\n"
                "3. legal_role은 시장지배적 사업자, 사업자단체, 지주회사, 대규모유통업자 등\n"
                "   법적 지위가 명시될 때만 채우고 아니면 '기타'로 둬라.\n"
                "4. company_size는 대기업/중견기업/중소기업 근거가 명시적일 때만 채우고,\n"
                "   지주회사, 공시대상기업집단, 대규모유통업자 문맥은 대기업으로 볼 수 있다.\n"
                "5. industry는 업종이 명확히 드러날 때만 채우고, 회사명만으로 억지 추론하지 말라.\n"
                "6. 애매한 경우는 과잉분류보다 보수적으로 '기타'를 선택하라."
            ),
            (
                "검증 체크리스트:\n"
                "- 모든 필드가 허용 label 안에 있는가?\n"
                "- 입력 텍스트에 없는 내용을 상상해 넣지 않았는가?\n"
                "- 확신이 낮으면 '기타'/'일반'으로 낮췄는가?"
            ),
        ]
    )


class LangChainRouteTagger:
    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        prompt_variant: str = "lcel",
        max_new_tokens: int = 256,
        batch_size: int = 2,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else resolve_llm_model_dir()
        self.prompt_variant = prompt_variant
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
                "LCEL 라우팅을 사용하려면 langchain-core, "
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
                "{route_instructions}\n\n"
                "{keyword_hints}\n\n"
                "입력 종류: {input_kind}\n"
                "입력 텍스트:\n{text}\n\n"
                "{format_instructions}\n"
                "JSON만 출력하라."
            ),
            input_variables=["input_kind", "text"],
            partial_variables={
                "route_instructions": build_lcel_route_instructions(self.prompt_variant),
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

    def tag_text(
        self,
        *,
        input_kind: str,
        text: str,
    ) -> LLMRouteDecision:
        result = self.chain.invoke(
            {
                "input_kind": input_kind,
                "text": text,
            }
        )
        if isinstance(result, Exception):
            raise result
        return result
