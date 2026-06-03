from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
NORMALIZE_PATTERN = re.compile(r"[\s\-\(\)\[\],./·]")
CORP_SUFFIX_PATTERN = re.compile(r"(주식회사|유한회사|재단법인|사단법인|\(주\)|㈜)")


# 한국어와 영문, 숫자를 기준으로 검색용 토큰을 추출합니다.
def tokenize_text(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1]


# 회사명과 문서명을 비교하기 쉽도록 불필요한 표기를 제거합니다.
def normalize_name(text: str) -> str:
    normalized = CORP_SUFFIX_PATTERN.sub("", text)
    normalized = NORMALIZE_PATTERN.sub("", normalized)
    return normalized.lower()
