"""
===============================================================================
공정위 AI·데이터 활용 공모전 - 통합 평가 데이터셋 생성 스크립트 (v4.1 - Async Optimized)
===============================================================================
"""

import os
import json
import random
import csv
import re
import time
import shutil
import asyncio
from google import genai

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

BASE_DIR = r'e:\muckja999\공모전\제2회 「공정위 AI·데이터」활용 공모전'
HYBRID_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'Hybrid')
META_DIR   = os.path.join(BASE_DIR, 'data', 'raw', 'Metadata')
EDA_CSV    = os.path.join(BASE_DIR, 'eda', 'classification_results.csv')
OUT_DATASET  = os.path.join(BASE_DIR, 'example', 'eval_dataset_260505.json')

api_key      = os.environ.get("GOOGLE_API_KEY", "")
client       = genai.Client(api_key=api_key) if api_key else None

# ─────────────────────────────────────────────────────────────────────────────
# 비동기 유틸 및 토큰 추적
# ─────────────────────────────────────────────────────────────────────────────

total_usage = {'prompt': 0, 'candidates': 0}

async def generate_reference_answer_via_llm_async(query, context, client):
    try:
        if client is None: return context
        prompt = f"""[System]
You are a professional legal investigator at the Korea Fair Trade Commission. 
Summarize the provided context into Korean in 3-5 sentences.
[User]
Query: {query}
Context: {context}"""
        response = await client.aio.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
        usage = response.usage_metadata
        if usage:
            total_usage['prompt'] += usage.prompt_token_count
            total_usage['candidates'] += usage.candidates_token_count
        return response.text.strip()
    except Exception as e:
        if "429" in str(e):
            await asyncio.sleep(2)
            return await generate_reference_answer_via_llm_async(query, context, client)
        return context

async def refine_violation_via_llm_async(title, chunks, client):
    try:
        if client is None: return "기타 불공정거래행위"
        text = ' '.join(c.get('page_content', '') for c in chunks[:5] if c.get('metadata', {}).get('chunk_type') == 'text')
        prompt = f"[System] Identify core 'Violation Category' in Korean noun phrase.\nTitle: {title}\nContent: {text[:1500]}\nCategory:"
        response = await client.aio.asyncio.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
        return response.text.strip().replace('Category:', '').strip()
    except: return "기타 불공정거래행위"

# ─────────────────────────────────────────────────────────────────────────────
# 기존 유틸 함수들
# ─────────────────────────────────────────────────────────────────────────────

def has_jongseong(char):
    if not re.match(r'[가-힣]', char): return False
    return (ord(char) - 44032) % 28 > 0

def attach_particle(word, particle_type):
    if not word: return word
    jong = has_jongseong(word[-1])
    if particle_type == '이/가':  return word + ('이' if jong else '가')
    if particle_type == '은/는':  return word + ('은' if jong else '는')
    if particle_type == '을/를':  return word + ('을' if jong else '를')
    if particle_type == '과/와':  return word + ('과' if jong else '와')
    return word

def refine_company_name(name):
    name = re.sub(r'\((?:주|사|재|유|합|유한|합자|학|의|복|영|특|농)\)', '', name)
    name = re.sub(r'주식회사|사단법인|재단법인|유한회사|합자회사|학교법인|의료법인|사회복지법인|영농조합법인', '', name)
    return name.strip()

def refine_industry_name(name):
    return name[:-2].strip() if name.endswith("업종") else name

def refine_violation(v, chunks, title, client):
    if v != '기타': return v
    return "기타 불공정거래행위" # 런타임 최적화를 위해 기본값 사용 (필요시 LLM 호출)

def refine_size(size, chunks):
    return '특정' if size == '기타' else size

def precompute_duplicate_map(docs):
    print("[STEP 2.5] Scanning for cross-document duplicate chunks...")
    dup_map = {}; chunk_to_content = {}
    for doc in tqdm(docs, desc="Duplicate Scan"):
        try:
            with open(doc['hybrid_path'], 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                for c in chunks:
                    if c.get('metadata', {}).get('chunk_type') != 'text': continue
                    text = c.get('page_content', '').strip()
                    if len(text) < 100: continue
                    cid = c['metadata']['chunk_id']
                    chunk_to_content[cid] = text
                    if text not in dup_map: dup_map[text] = []
                    dup_map[text].append((cid, doc['title']))
        except: pass
    dup_map = {k: v for k, v in dup_map.items() if len(v) > 1}
    return dup_map, chunk_to_content

def expand_duplicates(ans_chunks, dup_map, chunk_to_content):
    expanded = list(ans_chunks)
    existing_ids = set(a['chunk_id'] for a in ans_chunks)
    for a in ans_chunks:
        content = chunk_to_content.get(a['chunk_id'])
        if content and content in dup_map:
            for dup_id, doc_title in dup_map[content]:
                if dup_id not in existing_ids:
                    expanded.append({'chunk_id': dup_id, 'answer_reason': f"타 의결서({doc_title}) 중복 청크"})
                    existing_ids.add(dup_id)
    return expanded

def extract_anchor(chunk_list):
    # (추출 로직 생략되지 않도록 유지)
    text = ' '.join(c.get('page_content', '') for c in chunk_list)
    patterns = [
        ('amount', r'(\d+(?:\.\d+)?\s*억?\s*\d*\s*만?\s*원)'),
        ('period', r'(\d{4}년\s*(?:\d{1,2}월\s*)?(?:\d{1,2}일\s*)?(?:부터|~|및|-)\s*(?:\d{4}년\s*)?(?:\d{1,2}월\s*)?(?:\d{1,2}일\s*)?(?:까지|동안|내내)?)'),
    ]
    for a_type, pat in patterns:
        m = re.search(pat, text)
        if m: return m.group(1).strip(), a_type, None
    return None, None, None

def build_single_query(comp_ga, comp_ui, violation, vio_reul, anchor, a_type, a_subj, j_chunks, w_chunks, c_chunks):
    if anchor:
        return f"{comp_ga} {anchor} 관련 {violation} 행위로 인해 받은 시정조치와 법리적 근거를 설명해 주세요."
    return f"{comp_ui} 주요 위반 행위 사실과 그에 따른 공정위의 판단 내용을 설명해 주세요."

def build_multi_query(d1, d2, v1, v2, s1, s2, c1_wa, c2_ga, c1_ui, c2_ui, a1, at1, as1, a2, at2, as2, has_shared_content=False):
    if has_shared_content:
        return f"{d1['company']}와 {d2['company']}의 의결서에서 공통적으로 기재된 위반 행위의 배경과 위법성 판단 근거를 설명해 주세요."
    return f"{d1['company']}와 {d2['company']}의 위반 행위 사실을 대조하고 차이점을 설명해 주세요."

async def main():
    print("[STEP 0] Loading EDA CSV...")
    eda_data = {}
    with open(EDA_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('의결서제목'): eda_data[row['의결서제목']] = row

    print("[STEP 1] Loading metadata...")
    docs = []
    for fname in sorted(os.listdir(META_DIR)):
        if not fname.endswith('.json'): continue
        with open(os.path.join(META_DIR, fname), 'r', encoding='utf-8') as fp:
            meta = json.load(fp)
        title = meta.get('의결서제목', '')
        hybrid_path = os.path.join(HYBRID_DIR, fname.replace('_metadata.json', '_hybrid.json'))
        if not os.path.exists(hybrid_path): continue
        raw_company = title.split('의 ')[0].strip() if '의 ' in title else '해당 사업자'
        docs.append({
            'title': title, 'hybrid_path': hybrid_path, 'company': refine_company_name(raw_company),
            'violation_raw': eda_data.get(title, {}).get('위반분야', '불공정거래행위'),
            'industry': eda_data.get(title, {}).get('업종', '기타 업종'),
            'size_raw': eda_data.get(title, {}).get('기업규모', '일반 기업')
        })

    print("[STEP 2] Building chunk_index...")
    chunk_index = {}
    for doc in docs:
        with open(doc['hybrid_path'], 'r', encoding='utf-8') as f:
            for c in json.load(f):
                cid = c.get('metadata', {}).get('chunk_id')
                if cid: chunk_index[cid] = c.get('page_content', '')

    dup_map, chunk_to_content = precompute_duplicate_map(docs)
    sem = asyncio.Semaphore(10)

    # STEP 3: Single QA
    print("[STEP 3] Generating 500 single-doc QAs (Async)...")
    single_qas = []
    pbar = tqdm(total=len(docs), desc="Single QA")
    
    async def make_single(i, doc):
        async with sem:
            with open(doc['hybrid_path'], 'r', encoding='utf-8') as f: chunks = json.load(f)
            v = refine_violation(doc['violation_raw'], chunks, doc['title'], client)
            w_chunks = [c for c in chunks if '위법성' in c['metadata'].get('Header2', '')][:5]
            anchor, a_type, a_subj = extract_anchor(w_chunks)
            query = build_single_query(attach_particle(doc['company'], '이/가'), doc['company']+'의', v, attach_particle(v, '을/를'), anchor, a_type, a_subj, [], w_chunks, [])
            ans_chunks = [{'chunk_id': c['metadata']['chunk_id'], 'answer_reason': '위법성 판단 근거'} for c in w_chunks]
            ans_chunks = expand_duplicates(ans_chunks, dup_map, chunk_to_content)
            pbar.update(1)
            return {'id': f"Q-S-{i+1:03d}", 'query': query, 'source_doc': doc['title'], 'answer_chunks': ans_chunks[:10]}

    single_qas = await asyncio.gather(*[make_single(i, d) for i, d in enumerate(docs)])
    pbar.close()

    # STEP 4: Multi QA
    print("[STEP 4] Generating 100 multi-doc QAs (Async)...")
    multi_qas = []
    pbar = tqdm(total=100, desc="Multi QA")
    
    async def make_multi(idx):
        async with sem:
            d1, d2 = random.sample(docs, 2)
            query = build_multi_query(d1, d2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, False)
            pbar.update(1)
            return {'id': f"Q-M-{idx+1:03d}", 'query': query, 'source_doc': [d1['title'], d2['title']], 'answer_chunks': []}

    multi_qas = await asyncio.gather(*[make_multi(i) for i in range(100)])
    pbar.close()

    # STEP 5: Reference Answers
    print("[STEP 5] Generating Reference Answers (Async)...")
    existing_data = []
    if os.path.exists(OUT_DATASET):
        with open(OUT_DATASET, 'r', encoding='utf-8') as f: existing_data = json.load(f)
    
    processed_ids = {item['id'] for item in existing_data if item.get('reference_answer')}
    all_tasks = [item for item in (single_qas + multi_qas) if item['id'] not in processed_ids]
    generation_eval = existing_data
    missing_chunks = 0

    async def make_ref(item, pbar_ref):
        nonlocal missing_chunks
        async with sem:
            texts = [chunk_index.get(ac['chunk_id'], '') for ac in item['answer_chunks']]
            ref_context = '\n\n'.join(t for t in texts if t)
            ref_answer = await generate_reference_answer_via_llm_async(item['query'], ref_context, client)
            pbar_ref.update(1)
            pbar_ref.set_description(f"Gen | Tokens: {total_usage['prompt'] + total_usage['candidates']:,}")
            return {**item, 'reference_context': ref_context, 'reference_answer': ref_answer}

    if all_tasks:
        pbar_ref = tqdm(total=len(all_tasks), desc="Generating")
        batch_size = 20
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i+batch_size]
            results = await asyncio.gather(*[make_ref(item, pbar_ref) for item in batch])
            for res in results: generation_eval.append(res)
            with open(OUT_DATASET + ".tmp", 'w', encoding='utf-8') as f:
                json.dump(generation_eval, f, ensure_ascii=False, indent=2)
            shutil.move(OUT_DATASET + ".tmp", OUT_DATASET)
        pbar_ref.close()

    print(f"완료! 통합 데이터셋: {OUT_DATASET}")

if __name__ == "__main__":
    asyncio.run(main())
