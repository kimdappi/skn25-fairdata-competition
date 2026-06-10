from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
NORMALIZE_PATTERN = re.compile(r"[\s\-\(\)\[\],./·]")
CORP_SUFFIX_PATTERN = re.compile(r"(주식회사|유한회사|재단법인|사단법인|\(주\)|㈜)")


# 한국어와 영문, 숫자를 기준으로 검색용 토큰을 추출합니다.
def tokenize_text(text: str) -> list[str]:
    # 한 글자 토큰은 노이즈가 많아서 제외하고 소문자 기준으로 통일합니다.
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1]


# 회사명과 문서명을 비교하기 쉽도록 불필요한 표기를 제거합니다.
def normalize_name(text: str) -> str:
    # 법인 형태 표기와 구분 기호를 제거해 문자열 비교를 단순화합니다.
    normalized = CORP_SUFFIX_PATTERN.sub("", text)
    normalized = NORMALIZE_PATTERN.sub("", normalized)
    return normalized.lower()
