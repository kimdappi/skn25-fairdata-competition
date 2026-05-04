import math
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np

from baseline_rag.keywords import (
    ENTERPRISE_SIZE_KEYWORDS,
    FACT_KEYWORDS,
    INDUSTRY_KEYWORDS,
    LAW_KEYWORDS,
    LEGAL_ROLE_KEYWORDS,
    ORDER_KEYWORDS,
    THEME_KEYWORDS,
)

OTHER = "기타"
FOCUS_ORDER = "처분"
FOCUS_LAW = "위법성"
FOCUS_FACT = "사실관계"
FOCUS_GENERAL = "일반"

ROLE_DOMINANT = "시장지배적 사업자(독과점)"
SIZE_LARGE = "대기업"

HEADER_ORDER = "주 문"
HEADER_FACT_WORDS = ["사실", "행위", "배경", "경위"]
HEADER_LAW_WORDS = ["판단", "위법", "관련", "적용"]


def tokenize_text(text: str) -> List[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
    return [token for token in tokens if len(token) > 1]


def classify_text(text: str, keyword_map: Dict[str, List[str]], default: str = OTHER) -> str:
    for category, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return category
    return default


def infer_company_size(full_content: str, current_size: str, legal_role: str) -> str:
    if current_size != OTHER:
        return current_size
    if legal_role == ROLE_DOMINANT:
        return SIZE_LARGE
    if "대규모유통업자" in full_content or "지주회사" in full_content:
        return SIZE_LARGE
    if "상호출자제한" in full_content or "공시대상기업집단" in full_content:
        return SIZE_LARGE
    if any(keyword in full_content for keyword in ["플랫폼 운영자", "통신판매중개업자"]):
        return SIZE_LARGE
    return OTHER


def extract_focus(text: str) -> str:
    if any(keyword in text for keyword in ORDER_KEYWORDS):
        return FOCUS_ORDER
    if any(keyword in text for keyword in LAW_KEYWORDS):
        return FOCUS_LAW
    if any(keyword in text for keyword in FACT_KEYWORDS):
        return FOCUS_FACT
    return FOCUS_GENERAL


def build_rule_route_fields(text: str) -> Dict[str, str | List[str]]:
    legal_role = classify_text(text, LEGAL_ROLE_KEYWORDS)
    return {
        "theme": classify_text(text, THEME_KEYWORDS),
        "company_size": infer_company_size(text, classify_text(text, ENTERPRISE_SIZE_KEYWORDS), legal_role),
        "legal_role": legal_role,
        "industry": classify_text(text, INDUSTRY_KEYWORDS),
        "focus": extract_focus(text),
        "keywords": tokenize_text(text)[:20],
    }


def normalize_name(text: str) -> str:
    text = re.sub(r"\((?:주)\)", "", text)
    text = re.sub(r"(?:주식회사|유한회사|재단법인|사단법인|\(주\))", "", text)
    return re.sub(r"[\s\-\(\)\[\],./\u00b7]", "", text).lower()


def min_max_scale(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    score_min = float(scores.min())
    score_max = float(scores.max())
    if math.isclose(score_min, score_max):
        return np.ones_like(scores)
    return (scores - score_min) / (score_max - score_min + 1e-9)


def reciprocal_rank_fusion(rankings: Dict[str, List[int]], weights: Dict[str, float], *, k: int) -> Dict[int, float]:
    fused: Dict[int, float] = defaultdict(float)
    for name, ranking in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += weight / (k + rank)
    return fused


def ranked_indices_from_scores(scores: np.ndarray | None, top_k: int) -> List[int]:
    if scores is None or scores.size == 0 or top_k <= 0:
        return []
    top_k = min(top_k, scores.shape[0])
    return np.argsort(scores)[::-1][:top_k].tolist()
