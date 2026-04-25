"""
================================================================================
[공정위 AI 데이터 활용 공모전 - RAG 파이프라인 Step 2]
오프라인 벡터 데이터베이스(FAISS, BM25) 사전 구축 스크립트
================================================================================

이 스크립트는 RAG 시스템의 '데이터 수집 및 인덱싱(Data Ingestion & Indexing)'이라는
가장 기본적이고 핵심적인 역할을 담당하는 1회성 배치(Batch) 프로그램입니다.

[주요 역할 및 목적]
1. 30초 타임아웃 방지 (사전 인덱싱):
   - 대회 규정상 30초 이내에 검색 및 생성을 마쳐야 합니다. 만약 질문이 들어올 때마다 수천 개의 
     원본 JSON 파일을 파싱하고 벡터화한다면 무조건 타임아웃(0점)이 발생합니다.
   - 따라서 이 코드는 제출 전 로컬 환경에서 '단 한 번' 실행되어, AI가 즉시 유사도 연산을 
     수행할 수 있도록 데이터를 고속 인덱스 파일 형태로 미리 컴파일하고 영구 저장해 둡니다.

2. 메타데이터 융합 텍스트 전처리 (Metadata Enrichment):
   - _hybrid.json의 단순 본문(page_content)만 벡터화하면 "어떤 사건의 내용인지" 맥락을 잃기 쉽습니다.
   - 이를 방지하기 위해 _metadata.json의 핵심 정보(의결서제목, 피심인기업명, 위반유형)를 읽어와 
     본문 앞에 강제로 문자열 결합을 수행함으로써, 검색(Retrieval) 성능을 극대화합니다.

3. 하이브리드 앙상블 검색 기반 마련:
   - Dense Index (FAISS): 문맥 의미 기반 검색을 위한 고차원 벡터 인덱싱 (SentenceTransformer 모델 활용)
   - Sparse Index (BM25): 정확한 고유명사, 조항 번호 매칭을 위해 형태소 분석기(Kiwi)를 
     거친 키워드 토큰 기반 인덱싱

[출력 결과물]
실행 완료 시 지정된 `vector_db` 디렉토리에 아래 3개의 파일이 생성되며, 이 파일들은 
이후 Step 3의 검색(Retriever) 모듈 로드 시 사용되며, 최종 제출 시 도커 컨테이너 내부에 포함되어야 합니다.
- faiss_index.bin
- faiss_metadata.pkl
- bm25_index.pkl
================================================================================
"""
import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi # 한국어 형태소 분석기

def build_vector_database(raw_data_dir, vector_db_dir, embedding_model_path):
    hybrid_dir = os.path.join(raw_data_dir, 'Hybrid')
    metadata_dir = os.path.join(raw_data_dir, 'Metadata')
    
    chunks_data = [] # 전체 청크 정보 저장 리스트
    corpus_for_dense = [] # FAISS 임베딩용 텍스트
    corpus_for_sparse = [] # BM25용 형태소 토큰 리스트
    
    kiwi = Kiwi()
    
    # 1. 데이터 로드 및 메타데이터 병합
    print("데이터 로딩 및 전처리 시작...")
    for filename in os.listdir(hybrid_dir):
        if not filename.endswith('_hybrid.json'): continue
        
        # 파일명 매칭을 통해 짝이 되는 metadata.json 찾기
        base_name = filename.replace('_hybrid.json', '')
        meta_filepath = os.path.join(metadata_dir, f"{base_name}_metadata.json")
        hybrid_filepath = os.path.join(hybrid_dir, filename)
        
        try:
            with open(meta_filepath, 'r', encoding='utf-8') as f:
                meta_json = json.load(f)
            with open(hybrid_filepath, 'r', encoding='utf-8') as f:
                hybrid_json = json.load(f)
        except Exception as e:
            print(f"파일 읽기 오류 ({base_name}): {e}")
            continue
            
        # 메타데이터 정보 추출
        doc_title = meta_json.get('의결서제목', '')
        company_name = meta_json.get('피심인기업명', '')
        violation_type = meta_json.get('위반유형', '')
        
        # 각 청크별로 병합 진행
        for chunk in hybrid_json:
            content = chunk.get('page_content', '')
            chunk_id = chunk.get('metadata', {}).get('chunk_id', '')
            
            if not content or not chunk_id: continue
            
            # [전략 1] 메타데이터 융합 텍스트 생성
            enriched_text = f"[문서제목: {doc_title}, 피심인: {company_name}, 위반유형: {violation_type}] \n {content}"
            
            # [전략 2를 위한 준비] E5 모델 권장 Prefix 추가
            dense_text = f"passage: {enriched_text}"
            
            # [전략 3을 위한 준비] 형태소 분석을 통한 토큰화
            tokens = [t.form for t in kiwi.tokenize(enriched_text)]
            
            chunks_data.append({
                "chunk_id": chunk_id,
                "original_content": content,
                "enriched_text": enriched_text
            })
            corpus_for_dense.append(dense_text)
            corpus_for_sparse.append(tokens)

    print(f"총 {len(chunks_data)}개의 청크 전처리 완료.")

    # 2. FAISS Dense Index 구축
    print("FAISS 벡터 임베딩 생성 중... (시간이 소요됩니다)")
    # 오프라인 환경을 위해 로컬에 다운받은 모델 경로 지정 권장
    embedder = SentenceTransformer(embedding_model_path) 
    embeddings = embedder.encode(corpus_for_dense, show_progress_bar=True, convert_to_numpy=True)
    
    # 임베딩 차원 확인 후 FAISS 인덱스 생성 (L2 거리 기준)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    
    # 3. BM25 Sparse Index 구축
    print("BM25 인덱스 생성 중...")
    bm25_index = BM25Okapi(corpus_for_sparse)
    
    # 4. 생성된 인덱스 및 메타데이터 파일로 저장
    print("인덱스 파일 저장 중...")
    os.makedirs(vector_db_dir, exist_ok=True)
    
    faiss.write_index(faiss_index, os.path.join(vector_db_dir, 'faiss_index.bin'))
    
    with open(os.path.join(vector_db_dir, 'faiss_metadata.pkl'), 'wb') as f:
        pickle.dump(chunks_data, f)
        
    with open(os.path.join(vector_db_dir, 'bm25_index.pkl'), 'wb') as f:
        pickle.dump(bm25_index, f)
        
    print("✅ 벡터 데이터베이스 구축이 완벽하게 끝났습니다!")

if __name__ == "__main__":
    RAW_DATA_DIR = '../data/raw'
    VECTOR_DB_DIR = '../models/vector_db'
    EMBEDDING_MODEL = 'intfloat/multilingual-e5-large' 
    build_vector_database(RAW_DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL)