from __future__ import annotations

import logging
from typing import Protocol

from app.retrieval.llm_router import LangChainRouteTagger
from app.retrieval.router_names import normalize_router_backend_name
from app.retrieval.route_tags import normalize_question_key
from app.retrieval.router import OTHER, QueryRouter
from app.utils.schemas import LLMRouteDecision, RouteDecision
from app.utils.text import tokenize_text


logger = logging.getLogger(__name__)


class RouteRuntime(Protocol):
    def route_from_text(self, text: str) -> RouteDecision: ...


FOCUS_LABELS = {"처분", "위법성", "사실관계", "일반"}


def normalize_lcel_route_decision(parsed: LLMRouteDecision, source_text: str) -> RouteDecision:
    fallback = QueryRouter()
    fallback_decision = fallback.route_from_text(source_text)
    return RouteDecision(
        theme=parsed.theme if parsed.theme in fallback.rules.theme_keywords or parsed.theme == OTHER else fallback_decision.theme,
        company_size=(
            parsed.company_size
            if parsed.company_size in fallback.rules.enterprise_size_keywords or parsed.company_size == OTHER
            else fallback_decision.company_size
        ),
        legal_role=(
            parsed.legal_role
            if parsed.legal_role in fallback.rules.legal_role_keywords or parsed.legal_role == OTHER
            else fallback_decision.legal_role
        ),
        industry=(
            parsed.industry
            if parsed.industry in fallback.rules.industry_keywords or parsed.industry == OTHER
            else fallback_decision.industry
        ),
        focus=parsed.focus if parsed.focus in FOCUS_LABELS else fallback_decision.focus,
        keywords=tokenize_text(source_text)[:20],
    )


class LCELQuestionRouter:
    def __init__(
        self,
        fallback: QueryRouter,
        *,
        prompt_variant: str,
    ) -> None:
        self.fallback = fallback
        self.prompt_variant = prompt_variant
        self.tagger: LangChainRouteTagger | None = None
        self.routes_by_key: dict[str, RouteDecision] = {}

    def ensure_tagger(self) -> LangChainRouteTagger:
        if self.tagger is None:
            self.tagger = LangChainRouteTagger(prompt_variant=self.prompt_variant)
        return self.tagger

    def route_from_text(self, text: str) -> RouteDecision:
        key = normalize_question_key(text)
        cached = self.routes_by_key.get(key)
        if cached is not None:
            return cached

        try:
            parsed = self.ensure_tagger().tag_text(input_kind="question", text=text)
            route = normalize_lcel_route_decision(parsed, text)
        except Exception as exc:
            logger.warning("LCEL question routing failed; using keyword fallback: %r", exc)
            route = self.fallback.route_from_text(text)

        self.routes_by_key[key] = route
        return route


def build_runtime_router(backend_name: str, *, fallback: QueryRouter | None = None) -> RouteRuntime:
    normalized = normalize_router_backend_name(backend_name)
    keyword_router = fallback or QueryRouter()
    if normalized == "keyword":
        return keyword_router
    if normalized == "lcel":
        return LCELQuestionRouter(keyword_router, prompt_variant="lcel")
    if normalized == "lcel_prompt_boost":
        return LCELQuestionRouter(keyword_router, prompt_variant="lcel_prompt_boost")
    raise ValueError(
        "Unsupported router backend "
        f"{backend_name!r}. Expected one of: keyword, lcel, lcel_prompt_boost."
    )
