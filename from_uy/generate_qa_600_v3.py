import os
import json
import random
import csv
import re

BASE_DIR = '/mnt/hgfs/muckja999/공모전/제2회 「공정위 AI·데이터」활용 공모전'
HYBRID_DIR = os.path.join(BASE_DIR, 'data/raw/Hybrid')
META_DIR = os.path.join(BASE_DIR, 'data/raw/Metadata')
EDA_CSV = os.path.join(BASE_DIR, 'eda/classification_results.csv')
OUT_JSON = os.path.join(BASE_DIR, 'example/qa_samples_600.json')

def has_jongseong(char):
    if not re.match(r'[가-힣]', char):
        return False
    return (ord(char) - 44032) % 28 > 0

def attach_particle(word, particle_type):
    if not word: return word
    last_char = word[-1]
    jong = has_jongseong(last_char)
    if particle_type == '이/가': return word + ('이' if jong else '가')
    if particle_type == '은/는': return word + ('은' if jong else '는')
    if particle_type == '을/를': return word + ('을' if jong else '를')
    if particle_type == '과/와': return word + ('과' if jong else '와')
    return word

def refine_company_name(name):
    name = re.sub(r'\(주\)|주식회사|\(유\)|\(합\)|주\)', '', name).strip()
    return name

def refine_violation(violation, chunks, title):
    if violation == '공정거래(독과점/담합)':
        text_content = " ".join([c.get('text', '') for c in chunks if c.get('metadata', {}).get('chunk_type') == 'text'])
        if '담합' in text_content or '부당한 공동행위' in text_content or '입찰' in text_content:
            return '담합'
        elif '독점' in text_content or '시장지배' in text_content:
            return '독과점'
        else:
            return random.choice(['독과점', '담합'])
    elif violation == '갑을관계(불공정거래)':
        text_content = " ".join([c.get('text', '') for c in chunks if c.get('metadata', {}).get('chunk_type') == 'text'])
        if '하도급' in text_content or '가맹' in text_content or '대리점' in text_content:
            return '갑을관계'
        else:
            return '불공정거래'
    elif violation == '기타':
        search_text = title + " ".join([c.get('text', '') for c in chunks[:10] if c.get('metadata', {}).get('chunk_type') == 'text'])
        if '방문판매' in search_text: return '방문판매'
        if '전자상거래' in search_text: return '전자상거래'
        if '할부거래' in search_text: return '할부거래'
        if '소비자' in search_text: return '소비자보호'
        if '약관' in search_text: return '불공정약관'
        return '기타 불공정행위'
    return violation.split('. ')[-1] if '. ' in violation else violation

def refine_size(size, chunks):
    if size == '기타':
        return '특정'
    return size

# 1. Load EDA data using built-in csv
eda_data = {}
with open(EDA_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        title = row.get('의결서제목', '')
        if title:
            eda_data[title] = row

single_qas = []
multi_qas = []

print("Loading metadata...")
docs = []
for f in sorted(os.listdir(META_DIR)):
    if not f.endswith('.json'): continue
    with open(os.path.join(META_DIR, f), 'r', encoding='utf-8') as fp:
        meta = json.load(fp)
        title = meta.get('의결서제목', '')
        hybrid_f = f.replace('_metadata.json', '_hybrid.json')
        hybrid_path = os.path.join(HYBRID_DIR, hybrid_f)
        
        if os.path.exists(hybrid_path):
            raw_company = "해당 사업자"
            if '관련' in title and '사업자의' in title:
                match = re.search(r'관련\s+(.*?사업자)의', title)
                if match: raw_company = match.group(1).strip()
            elif '의 ' in title:
                raw_company = title.split('의 ')[0].strip()
            
            company = refine_company_name(raw_company)
            
            # Use EDA metadata primarily
            if title in eda_data:
                row = eda_data[title]
                violation_raw = row.get('위반분야', '불공정거래행위')
                industry_raw = row.get('업종', '기타 업종')
                size_raw = row.get('기업규모', '일반 기업')
                legal_status_raw = row.get('법적지위', '사업자')
            else:
                violation_raw = "불공정거래행위"
                industry_raw = "기타 업종"
                size_raw = "일반 기업"
                legal_status_raw = "사업자"
                
            docs.append({
                'title': title,
                'hybrid_path': hybrid_path,
                'company': company,
                'violation_raw': violation_raw,
                'industry': industry_raw,
                'size_raw': size_raw,
                'legal_status': legal_status_raw
            })

print("Generating 500 single-doc QAs...")
random.seed(42)
for i, doc in enumerate(docs):
    with open(doc['hybrid_path'], 'r', encoding='utf-8') as fp:
        chunks = json.load(fp)
        
    doc['violation'] = refine_violation(doc['violation_raw'], chunks, doc['title'])
    doc['size'] = refine_size(doc['size_raw'], chunks)
    
    j_chunks = [c for c in chunks if c['metadata'].get('section') == '주문']
    w_chunks = [c for c in chunks if '위법성' in c['metadata'].get('Header2', '') and c['metadata'].get('chunk_type') == 'text']
    c_chunks = [c for c in chunks if '처분' in c['metadata'].get('Header2', '') and c['metadata'].get('chunk_type') == 'text']
    
    if not w_chunks:
        w_chunks = [c for c in chunks if c['metadata'].get('section') == '이유' and c['metadata'].get('chunk_type') == 'text']
    
    ans_chunks = []
    if j_chunks:
        ans_chunks.append({
            "chunk_id": j_chunks[0]['metadata']['chunk_id'],
            "answer_reason": f"주문 청크로, {doc['company']}의 핵심 위반 내용과 시정명령 요약 포함"
        })
    
    for c in w_chunks[:3]:
        ans_chunks.append({
            "chunk_id": c['metadata']['chunk_id'],
            "answer_reason": f"{doc['violation']}에 대한 위법성 판단 근거 및 구체적인 행위 사실 포함"
        })
        
    for c in c_chunks[:1]:
        ans_chunks.append({
            "chunk_id": c['metadata']['chunk_id'],
            "answer_reason": "위반 행위에 대한 구체적인 처분(시정조치/과징금 등) 내용 포함"
        })
        
    idx = 0
    while len(ans_chunks) < 5 and idx < len(chunks):
        cid = chunks[idx]['metadata']['chunk_id']
        if not any(a['chunk_id'] == cid for a in ans_chunks):
            ans_chunks.append({
                "chunk_id": cid,
                "answer_reason": "위반 사실 및 배경에 대한 추가 보충 설명"
            })
        idx += 1
        
    comp_ga = attach_particle(doc['company'], '이/가')
    comp_eun = attach_particle(doc['company'], '은/는')
    comp_ui = doc['company'] + '의'
    vio_reul = attach_particle(doc['violation'], '을/를')
    
    single_templates = [
        f"{comp_ga} {vio_reul} 한 구체적인 행위 사실과 그에 따른 처분 결과는 무엇인가?",
        f"{comp_ui} {doc['violation']} 사건에서 공정위가 내린 시정조치와 과징금 산정의 핵심 근거는 무엇인가?",
        f"{comp_ga} 위반한 {doc['violation']}의 사실관계 및 위법성 판단 내용을 설명하시오."
    ]
    
    query = random.choice(single_templates)
    
    single_qas.append({
        "id": f"Q-S-{i+1:03d}",
        "query": query,
        "source_doc": doc['title'],
        "answer_chunks": ans_chunks[:5]
    })

print("Generating 100 multi-doc QAs using varied templates...")
multi_count = 0
while multi_count < 100:
    d1, d2 = random.sample(docs, 2)
    
    with open(d1['hybrid_path'], 'r', encoding='utf-8') as f1:
        c1 = json.load(f1)
    with open(d2['hybrid_path'], 'r', encoding='utf-8') as f2:
        c2 = json.load(f2)
        
    v1 = refine_violation(d1['violation_raw'], c1, d1['title'])
    v2 = refine_violation(d2['violation_raw'], c2, d2['title'])
    s1 = refine_size(d1['size_raw'], c1)
    s2 = refine_size(d2['size_raw'], c2)
    
    comp1_wa = attach_particle(d1['company'], '과/와')
    comp2_ga = attach_particle(d2['company'], '이/가')
    comp1_ui = d1['company'] + '의'
    comp2_ui = d2['company'] + '의'
    
    query_templates = []
    
    if d1['industry'] == d2['industry']:
        query_templates.append(f"같은 {d1['industry']} 내에서 {comp1_wa} {comp2_ga} 벌인 위반 사례의 위법성을 대조하시오.")
    else:
        query_templates.append(f"서로 다른 업종({d1['industry']}, {d2['industry']})에 속한 두 기업 {d1['company']}, {d2['company']}의 사건을 비교하고 위법성 판단 근거를 설명하시오.")
        
    if s1 == s2 and s1 != '특정':
        query_templates.append(f"{s1} 규모인 {comp1_wa} {comp2_ui} {v1} 등 위반 사실을 비교하시오.")
        
    if d1['legal_status'] == d2['legal_status']:
        query_templates.append(f"공통적으로 {d1['legal_status']} 지위를 남용한 {comp1_wa} {comp2_ui} 행위 사실과 제재 수위를 비교하시오.")
        
    if d1['industry'] == d2['industry']:
        query_templates.append(f"최근 공정위가 제재한 {d1['industry']} 업종의 주요 불공정 행위 양상을 의결서 내용들을 바탕으로 분석하시오.")
    else:
        query_templates.append(f"최근 공정위가 제재한 {d1['industry']} 업종과 {d2['industry']} 업종의 주요 불공정 행위 양상을 의결서 내용들을 바탕으로 분석하시오.")
        
    if s1 == s2 and s1 != '특정':
        query_templates.append(f"{s1} 규모 기업들의 불공정 행위 제재 사례들을 바탕으로 핵심 법리 적용 과정을 요약하시오.")
    
    query_templates.append(f"공통된 법적 지위({d1['legal_status']})를 가진 사업자들이 저지른 사례들을 참고하여 공통점과 차이점은 무엇인지 서술하시오.")
    
    query = random.choice(query_templates)
    
    w1 = [c for c in c1 if '위법성' in c.get('metadata', {}).get('Header2', '') and c.get('metadata', {}).get('chunk_type') == 'text']
    w2 = [c for c in c2 if '위법성' in c.get('metadata', {}).get('Header2', '') and c.get('metadata', {}).get('chunk_type') == 'text']
    
    if not w1: w1 = c1
    if not w2: w2 = c2
    
    ans_chunks = []
    for c in w1[:3]:
        ans_chunks.append({
            "chunk_id": c['metadata']['chunk_id'],
            "answer_reason": f"{d1['company']}의 {v1}에 대한 위반 사실 및 법리 판단 근거"
        })
    for c in w2[:2]:
        ans_chunks.append({
            "chunk_id": c['metadata']['chunk_id'],
            "answer_reason": f"{d2['company']}의 {v2}에 대한 위반 사실 및 법리 판단 근거"
        })
        
    multi_qas.append({
        "id": f"Q-M-{multi_count+1:03d}",
        "query": query,
        "source_doc": f"1. {d1['title']}\n2. {d2['title']}",
        "answer_chunks": ans_chunks
    })
    multi_count += 1

all_qas = single_qas + multi_qas

print("Saving to JSON...")
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(all_qas, f, ensure_ascii=False, indent=2)

print(f"Done! Generated {len(single_qas)} single-doc QAs and {len(multi_qas)} multi-doc QAs. Total: {len(all_qas)}")
