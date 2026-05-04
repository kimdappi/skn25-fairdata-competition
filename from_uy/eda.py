import os
import json
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. 경로 설정 (스크립트 위치 기준 절대 경로)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # eda 폴더 위치
PROJECT_ROOT = os.path.dirname(BASE_DIR)              # 프로젝트 루트 위치

METADATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Metadata')
HYBRID_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Hybrid')
OUTPUT_FILE = os.path.join(BASE_DIR, 'classification_results.csv')

# ---------------------------------------------------------
# 2. 분류 키워드 정의
# ---------------------------------------------------------
# 2) 핵심 화두 및 위반사항 (공정위 공식 분류 체계 기반)
THEME_KEYWORDS = {
    "1. 공정거래(독과점/담합)": ["담합", "공동행위", "시장지배적지위", "남용", "부당지원", "기업결합", "독점"],
    "2. 갑을관계(불공정거래)": ["하도급", "가맹", "프랜차이즈", "대리점", "유통업", "거래상 지위", "기술유용", "부당특약", "대규모유통"],
    "3. 소비자보호": ["표시광고", "전자상거래", "약관", "방문판매", "할부거래", "허위", "과장", "기만", "다크패턴"]
}

# 2) 기업 규모 및 주체 (분류 체계 보완)
ENTERPRISE_SIZE_KEYWORDS = {
    "대기업": ["대기업", "대규모", "글로벌 기업", "지주사", "대기업집단", "상호출자제한기업집단", "공시대상기업집단", "지주회사", "기업집단", "계열회사", "동일인"],
    "중견/중소/영세기업": ["중견기업", "중소기업", "소기업", "소상공인", "벤처기업", "영세사업자", "중소업체", "소규모", "개인사업자", "자영업자", "영세"],
    "공공기관/공기업": ["공사", "공단", "지방자치단체", "정부기관", "공공기관", "지자체"],
    "사업자단체/협동조합": ["협의회", "협회", "조합", "연합회", "사업자단체"]
}

# 3) 법적 지위 및 역할 (공정거래법 및 각 전문법상 공식 용어 반영)
LEGAL_ROLE_KEYWORDS = {
    "시장지배적 사업자(독과점)": ["시장지배적사업자", "독점사업자", "시장지배적지위", "독과점"],
    "거래상 우월적 지위자(갑)": [
        "원사업자", "가맹본부", "대규모유통업자", "공급업자", "발주자", "위탁자", 
        "플랫폼 운영자", "통신판매중개업자", "앱마켓 사업자", "배달앱", "오픈마켓", "검색포털", "운영사업자"
    ],
    "거래 상대방(을)": [
        "수급사업자", "가맹점사업자", "가맹점주", "납품업자", "대리점주", "매장임차인", 
        "수탁사업자", "신고인", "소비자", "이용자"
    ],
    "사업자단체": ["사업자단체", "협회", "조합", "협의회", "연합회", "친목회", "동호회"]
}

# 4) 기업 업종 (공정위 주요 감시 및 빈출 산업군 세분화)
INDUSTRY_KEYWORDS = {
    "제조/건설업": ["제조업", "건설업", "건축", "토목", "제조위탁", "건설위탁", "자동차부품", "기계", "전자부품"],
    "도소매/유통업": ["도소매업", "소매업", "도매업", "대형마트", "백화점", "편의점", "복합쇼핑몰", "아울렛", "면세점"],
    "디지털/플랫폼/IT": ["정보통신업", "소프트웨어", "소프트웨어 개발위탁", "플랫폼", "전자상거래", "통신판매업", "앱마켓", "배달앱", "오픈마켓", "온라인쇼핑몰", "포털"],
    "의료/제약업": ["제약업", "의료기기", "의약품", "병원", "의원", "도매상", "리베이트"],
    "운수/물류업": ["운수업", "물류업", "택배", "해운업", "항공업", "화물운송", "운송위탁"],
    "금융/보험업": ["금융업", "보험업", "은행", "카드사", "여신전문금융", "상호저축은행"],
    "서비스 및 기타": ["서비스업", "역무위탁", "프랜차이즈", "외식업", "교육서비스", "학원"]
}

def classify_text(text, keyword_map, default="기타"):
    for category, keywords in keyword_map.items():
        if any(kw in text for kw in keywords):
            return category
    return default

def infer_company_size(full_content, current_size, legal_role):
    """본문 문맥을 통한 기업 규모 유추 로직(Heuristics) 보완"""
    if current_size != "기타":
        return current_size
    
    # 1. 지위 기반 유추
    if legal_role == "시장지배적 사업자(독과점)":
        return "대기업"
    if "대규모유통업자" in full_content or "지주회사" in full_content:
        return "대기업"
    
    # 2. 법률 적용 기반 유추
    if "상호출자제한" in full_content or "공시대상기업집단" in full_content:
        return "대기업"
    
    # 3. 플랫폼 기업 등 신산업군 (대부분 중견 이상)
    if any(kw in full_content for kw in ["플랫폼 운영자", "통신판매중개업자"]):
        return "대기업" # 플랫폼 사업자는 대개 우월적 지위를 가진 중견/대기업
        
    return "기타"

def process_files(limit=500):
    files = [f for f in os.listdir(METADATA_DIR) if f.endswith('.json')]
    files = sorted(files)[:limit]
    
    results = []
    
    for filename in files:
        meta_path = os.path.join(METADATA_DIR, filename)
        hybrid_filename = filename.replace('_metadata.json', '_hybrid.json')
        hybrid_path = os.path.join(HYBRID_DIR, hybrid_filename)
        
        # 메타데이터 읽기
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        case_title = meta_data.get("의결서제목", "")
        
        # 하이브리드(본문) 데이터 읽기 (추출 범위 확대: 10개 청크)
        full_content = case_title
        if os.path.exists(hybrid_path):
            with open(hybrid_path, 'r', encoding='utf-8') as f:
                hybrid_data = json.load(f)
                full_content += " " + " ".join([chunk.get("page_content", "") for chunk in hybrid_data[:10]])
        
        # 분류 실행
        violation_cat = classify_text(full_content, THEME_KEYWORDS)
        legal_role = classify_text(full_content, LEGAL_ROLE_KEYWORDS)
        
        # 기업 규모 (키워드 매칭 + 유추 로직 적용)
        company_size = classify_text(full_content, ENTERPRISE_SIZE_KEYWORDS)
        company_size = infer_company_size(full_content, company_size, legal_role)
        
        industry = classify_text(full_content, INDUSTRY_KEYWORDS)
        
        results.append({
            "의결서번호": meta_data.get("의결서관리번호", ""),
            "의결서제목": case_title,
            "위반분야": violation_cat,
            "기업규모": company_size,
            "법적지위": legal_role,
            "업종": industry
        })
        
    return results
