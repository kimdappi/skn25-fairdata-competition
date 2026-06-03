from __future__ import annotations

from app.retrieval.keywords import (
    ENTERPRISE_SIZE_KEYWORDS,
    FACT_KEYWORDS,
    INDUSTRY_KEYWORDS,
    LAW_KEYWORDS,
    LEGAL_ROLE_KEYWORDS,
    ORDER_KEYWORDS,
    THEME_KEYWORDS,
)
from app.utils.schemas import RouteDecision
from app.utils.text import tokenize_text


OTHER = "기타"
FOCUS_GENERAL = "일반"
FOCUS_FACT = "사실관계"
FOCUS_LAW = "위법성"
FOCUS_ORDER = "처분"


class RoutingRules:
    # 규칙 기반 질의 분류에 필요한 키워드 사전을 초기화합니다.
    def __init__(self) -> None:
        self.theme_keywords = THEME_KEYWORDS
        self.enterprise_size_keywords = ENTERPRISE_SIZE_KEYWORDS
        self.legal_role_keywords = LEGAL_ROLE_KEYWORDS
        self.industry_keywords = INDUSTRY_KEYWORDS
        self.order_keywords = ORDER_KEYWORDS
        self.law_keywords = LAW_KEYWORDS
        self.fact_keywords = FACT_KEYWORDS

    # 키워드 사전을 이용해 텍스트를 가장 가까운 범주로 분류합니다.
    def classify_text(self, text: str, keyword_map: dict[str, list[str]], default: str = OTHER) -> str:
        for category, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                return category
        return default

    # 기업 규모가 직접 드러나지 않을 때 법적 지위와 문맥으로 보정합니다.
    def infer_company_size(self, text: str, current_size: str, legal_role: str) -> str:
        if current_size != OTHER:
            return current_size
        if legal_role == "시장지배적 사업자(독과점)":
            return "대기업"
        if any(keyword in text for keyword in ["대규모유통업자", "지주회사", "공시대상기업집단"]):
            return "대기업"
        return OTHER

    # 질문의 초점을 처분, 위법성, 사실관계, 일반 중 하나로 추정합니다.
    def extract_focus(self, text: str) -> str:
        if any(keyword in text for keyword in self.order_keywords):
            return FOCUS_ORDER
        if any(keyword in text for keyword in self.law_keywords):
            return FOCUS_LAW
        if any(keyword in text for keyword in self.fact_keywords):
            return FOCUS_FACT
        return FOCUS_GENERAL

    # 입력 텍스트를 질의 의도 메타데이터로 변환합니다.
    def build_route_decision(self, text: str) -> RouteDecision:
        legal_role = self.classify_text(text, self.legal_role_keywords)
        company_size = self.infer_company_size(
            text=text,
            current_size=self.classify_text(text, self.enterprise_size_keywords),
            legal_role=legal_role,
        )
        return RouteDecision(
            theme=self.classify_text(text, self.theme_keywords, OTHER),
            company_size=company_size,
            legal_role=legal_role,
            industry=self.classify_text(text, self.industry_keywords),
            focus=self.extract_focus(text),
            keywords=tokenize_text(text)[:20],
        )


class QueryRouter:
    # 규칙 객체를 감싼 질의 라우터 인터페이스를 초기화합니다.
    def __init__(self, rules: RoutingRules | None = None) -> None:
        self.rules = rules or RoutingRules()

    # 입력 텍스트를 질의 의도 메타데이터로 변환합니다.
    def route_from_text(self, text: str) -> RouteDecision:
        return self.rules.build_route_decision(text)
