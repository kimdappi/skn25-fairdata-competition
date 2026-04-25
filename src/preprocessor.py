"""
================================================================================
[공정위 AI 데이터 활용 공모전 - RAG 파이프라인]
데이터 전처리 및 메타데이터 융합 모듈 (Preprocessor)
================================================================================

이 모듈은 벡터 데이터베이스(FAISS, BM25)를 구축하기 전, 데이터의 '맥락(Context)'을 
보존하고 검색 성능(Recall@5, MRR)을 극대화하기 위한 '메타데이터 융합(Metadata Enrichment)'을 수행합니다.

[주요 역할 및 목적]
1. 청크의 문맥 단절(Context Loss) 문제 해결:
   - 수십 페이지에 달하는 긴 공정위 의결서 문서를 여러 개의 작은 청크(Chunk)로 쪼개게 되면, 
     중간에 위치한 본문 청크들은 자신이 '어떤 기업의 사건'인지, '어떤 위반 유형'인지에 대한 
     핵심 문맥을 잃어버리게 됩니다.
   - 이 모듈은 짝이 맞는 `_metadata.json`의 핵심 요약 정보(문서제목, 기업명, 위반유형)를 읽어와 
     모든 청크의 맨 앞에 접두사(Prefix) 형태로 강제 결합합니다.
   - 변환 예: "[문서제목: OO, 피심인: OO, 위반유형: OO] \n 1. 피심인은 임대단가를..."

2. 모델 특화 프롬프트 추가 (Task-specific Prefixing):
   - Dense 검색에 추천되는 다국어 임베딩 모델(예: multilingual-e5-large)은 
     코퍼스(문서)를 임베딩할 때 'passage: '라는 텍스트를 앞에 붙이는 것을 강하게 권장합니다. 
     이러한 임베딩 모델 맞춤형 텍스트를 따로 분리하여 생성합니다.

3. 파이프라인 모듈 간의 책임 분리 (Decoupling):
   - `data_loader.py`가 단순히 파일을 읽어오는 역할만 한다면, 이 모듈은 데이터를 
     검색 엔진이 가장 좋아하는 형태로 '요리'하는 역할을 합니다. 
   - 전처리가 완료된 정제 데이터만 `build_index.py`로 넘겨 코드를 모듈화하고 유지보수를 쉽게 합니다.
================================================================================
"""

def enrich_chunk_with_metadata(chunk_dict, meta_dict):
    """
    단일 청크 데이터에 메타데이터를 융합하여 검색에 최적화된 텍스트들을 생성합니다.
    
    Args:
        chunk_dict (dict): _hybrid.json에서 파싱된 단일 청크 정보
        meta_dict (dict): 짝이 맞는 _metadata.json의 전체 정보
        
    Returns:
        dict: 전처리가 완료된 다목적 청크 데이터 딕셔너리
    """
    original_content = chunk_dict.get('page_content', '').strip()
    chunk_id = chunk_dict.get('metadata', {}).get('chunk_id', '')
    
    # 메타데이터 핵심 정보 추출 (값이 없을 경우를 대비한 기본값 처리)
    doc_title = meta_dict.get('의결서제목', '제목없음')
    company_name = meta_dict.get('피심인기업명', '알수없음')
    violation_type = meta_dict.get('위반유형', '분류없음')
    
    # 1. 하이브리드 검색(BM25) 및 Reranker에 활용할 메타데이터 융합 텍스트
    enriched_text = f"[문서제목: {doc_title}, 피심인: {company_name}, 위반유형: {violation_type}]\n{original_content}"
    
    # 2. Dense 임베딩(E5 모델 등)을 위한 Prefix 추가 텍스트
    # E5 계열 모델은 검색 대상 문서에는 'passage: ', 질문에는 'query: '를 붙여야 성능이 극대화됩니다.
    dense_text = f"passage: {enriched_text}"
    
    return {
        "chunk_id": chunk_id,
        "original_content": original_content,  # 추후 답변 생성(Generation) 시 프롬프트에 넣을 원본
        "enriched_text": enriched_text,        # 형태소 분석기(Kiwi) 및 Reranker에 사용할 텍스트
        "dense_text": dense_text               # FAISS 임베딩 모델 연산에 집어넣을 텍스트
    }


def preprocess_documents(data_generator):
    """
    data_loader의 제너레이터를 통째로 받아, 전체 문서의 청크들을 한 번에 전처리합니다.
    
    Args:
        data_generator (generator): data_loader.load_paired_json_data()의 반환값
        
    Returns:
        list: 전처리가 완료된 모든 청크 딕셔너리들의 리스트
    """
    processed_chunks = []
    
    print("데이터 전처리(메타데이터 융합) 진행 중...")
    
    for base_name, meta_json, hybrid_json in data_generator:
        for chunk in hybrid_json:
            # 내용이 비어있거나 chunk_id가 없는 쓰레기 데이터(Noise)는 사전 차단(Skip)
            if not chunk.get('page_content') or not chunk.get('metadata', {}).get('chunk_id'):
                continue
                
            processed_chunk = enrich_chunk_with_metadata(chunk, meta_json)
            processed_chunks.append(processed_chunk)
            
    print(f"총 {len(processed_chunks)}개의 청크 메타데이터 융합 완료.")
    return processed_chunks

if __name__ == "__main__":
    # 단독 테스트 로직 (data_loader가 필요함)
    import sys
    import os
    
    # 임시로 모듈 경로 추가
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from data_loader import load_paired_json_data
        
        RAW_DATA_DIR = '../data/raw'
        if os.path.exists(RAW_DATA_DIR):
            print("\n--- 전처리 모듈 테스트 ---")
            loader = load_paired_json_data(RAW_DATA_DIR)
            processed_data = preprocess_documents(loader)
            
            if processed_data:
                print("\n[전처리 완료된 첫 번째 청크 샘플 확인]")
                sample = processed_data[0]
                print(f"Chunk ID: {sample['chunk_id']}")
                print(f"Dense 임베딩용 텍스트 (앞부분 100자):\n{sample['dense_text'][:100]}...")
    except ImportError:
        print("테스트를 실행하려면 data_loader.py 파일이 같은 폴더 내에 있어야 합니다.")