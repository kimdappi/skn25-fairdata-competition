"""
================================================================================
[공정위 AI 데이터 활용 공모전 - RAG 파이프라인]
데이터 로드 및 파싱 모듈 (Data Loader)
================================================================================

이 모듈은 전체 RAG 파이프라인의 가장 첫 단계인 '원시 데이터 수집 및 적재(Data Ingestion)' 
역할을 전담하는 핵심 유틸리티입니다.

[주요 역할 및 목적]
1. 원본 ZIP 파일 안전 해제 및 정규화 (인코딩 복원):
   - Windows 환경에서 압축된 한글 파일명이 Linux(Docker/RunPod) 환경에서 풀릴 때 
     발생하는 'File name too long' 에러와 외계어 깨짐 현상을 강제 인코딩 변환
     (cp437 -> cp949)을 통해 안전하게 복원합니다.

2. RAG 파이프라인 맞춤형 데이터 분류:
   - 주최측이 제공한 수천 개의 파일 중 불필요한 파일(PDF 등)은 과감히 건너뛰고, 
     오직 벡터 DB 구축에 필요한 `_hybrid.json`과 `_metadata.json` 파일만 추려내어 
     각각의 지정된 디렉토리에 깔끔하게 분류하여 저장합니다.

3. 데이터 파싱 및 1:1 매핑 스트림 제공 (Generator):
   - 로컬 디렉토리에 흩어진 하이브리드(본문 청크) 파일과 메타데이터(문서 요약) 파일을 
     '파일명'을 기준으로 완벽하게 1:1 매핑하여 파이썬 딕셔너리(Dict) 객체로 읽어옵니다.
   - 이 모듈을 통해 `build_index.py`나 `preprocessor.py` 같은 후속 모듈들은 
     복잡한 파일 입출력(I/O)이나 경로 처리에 신경 쓸 필요 없이, 정제된 데이터만을 
     손쉽게 공급받아 핵심 알고리즘에만 집중할 수 있게 됩니다.
================================================================================
"""
import os
import json
import zipfile
import shutil

def extract_and_classify_zip(zip_filepath, base_output_dir):
    """
    ZIP 파일을 압축 해제하고 파일명 패턴에 따라 폴더별로 분류하여 저장합니다.
    (한글 파일명 깨짐 방지 및 불필요 파일 필터링 처리 포함)
    """
    hybrid_dir = os.path.join(base_output_dir, 'Hybrid')
    metadata_dir = os.path.join(base_output_dir, 'Metadata')

    for folder in [hybrid_dir, metadata_dir]:
        os.makedirs(folder, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.is_dir():
                    continue

                # 💡 핵심 수정 부분: 한글 인코딩(cp949) 복원 처리 (오류 무시 옵션 추가)
                original_name = file_info.filename
                try:
                    # zipfile이 기본(cp437)으로 읽은 문자열을 바이트로 되돌린 후, cp949로 디코딩
                    decoded_name = original_name.encode('cp437').decode('cp949', 'ignore')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    decoded_name = original_name

                # 디코딩된 경로에서 순수 파일명만 추출
                filename = os.path.basename(decoded_name)
                if not filename:
                    continue

                filename_lower = filename.lower()
                
                # 패턴에 따른 경로 지정 (PDF 등 불필요 파일은 continue로 패스)
                if filename_lower.endswith('_hybrid.json'):
                    target_path = os.path.join(hybrid_dir, filename)
                elif filename_lower.endswith('_metadata.json'):
                    target_path = os.path.join(metadata_dir, filename)
                else:
                    continue 

                # 파일 추출 및 저장
                with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                    
        print(f"✅ 압축 해제 및 분류가 완료되었습니다. (저장 위치: {base_output_dir})")

    except FileNotFoundError:
        print(f"❌ 오류: '{zip_filepath}' 파일을 찾을 수 없습니다.")
    except zipfile.BadZipFile:
        print(f"❌ 오류: '{zip_filepath}' 파일이 손상되었거나 올바른 ZIP 파일이 아닙니다.")


def load_paired_json_data(raw_data_dir):
    """
    추출된 Hybrid 및 Metadata 폴더를 순회하며 짝이 맞는 JSON 데이터를 파싱하여 반환하는 Generator.
    (이 함수를 호출하면 메모리를 절약하면서 파일들을 하나씩 처리할 수 있습니다.)
    
    Yields:
        tuple: (base_name, metadata_json_dict, hybrid_json_list)
    """
    hybrid_dir = os.path.join(raw_data_dir, 'Hybrid')
    metadata_dir = os.path.join(raw_data_dir, 'Metadata')
    
    if not os.path.exists(hybrid_dir) or not os.path.exists(metadata_dir):
        print("❌ 오류: 원본 데이터 폴더(Hybrid 또는 Metadata)가 존재하지 않습니다.")
        return

    # Hybrid 폴더에 있는 파일들을 기준으로 짝(Metadata)을 찾습니다.
    for filename in os.listdir(hybrid_dir):
        if not filename.endswith('_hybrid.json'):
            continue
            
        base_name = filename.replace('_hybrid.json', '')
        meta_filepath = os.path.join(metadata_dir, f"{base_name}_metadata.json")
        hybrid_filepath = os.path.join(hybrid_dir, filename)
        
        # 짝이 되는 메타데이터 파일이 존재하는지 확인
        if not os.path.exists(meta_filepath):
            print(f"⚠️ 경고: '{base_name}'에 대한 메타데이터 파일이 없습니다. 건너뜁니다.")
            continue
            
        try:
            with open(meta_filepath, 'r', encoding='utf-8') as f:
                meta_json = json.load(f)
            with open(hybrid_filepath, 'r', encoding='utf-8') as f:
                hybrid_json = json.load(f)
                
            # 파일 이름 기준(base_name)과 파싱된 데이터를 튜플 형태로 전달
            yield base_name, meta_json, hybrid_json
            
        except json.JSONDecodeError:
            print(f"❌ JSON 파싱 오류: '{base_name}' 파일의 형식이 올바르지 않습니다.")
        except Exception as e:
            print(f"❌ 파일 읽기 오류 ({base_name}): {e}")


def save_mapped_data(raw_data_dir, processed_dir):
    """
    로드 및 매핑된 데이터를 눈으로 확인하거나 디버깅 목적으로 쓰기 위해 
    물리적인 파일(JSON)로 병합하여 저장하는 기능입니다.
    """
    if not os.path.exists(raw_data_dir):
        print("❌ 오류: 원본 데이터 폴더가 존재하지 않아 저장할 수 없습니다.")
        return 0
        
    os.makedirs(processed_dir, exist_ok=True)
    loader = load_paired_json_data(raw_data_dir)
    
    saved_count = 0
    for base_name, meta_data, hybrid_data in loader:
        # 문서 요약(meta)과 본문 청크(hybrid)를 하나의 딕셔너리로 깔끔하게 묶음
        merged_data = {
            "metadata": meta_data,
            "chunks": hybrid_data
        }
        
        save_path = os.path.join(processed_dir, f"{base_name}_mapped.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 옵션으로 한글이 \uXXXX 형태로 깨지지 않게 저장
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
        saved_count += 1
        
    print(f"✅ 물리적 저장 완료: 총 {saved_count}개의 병합 파일이 '{processed_dir}' 폴더에 저장되었습니다.")
    return saved_count


if __name__ == "__main__":
    # 단독으로 이 스크립트를 실행할 때의 테스트 로직
    # 1. 압축 해제 테스트
    ZIP_FILE = '../data/공개본 의결서.zip' 
    OUTPUT_DIR = '../data/raw' 
    PROCESSED_DIR = '../data/processed' # 저장할 폴더 경로 추가
    
    if os.path.exists(ZIP_FILE):
        extract_and_classify_zip(ZIP_FILE, OUTPUT_DIR)
        
    # 2. 로드 기능 테스트 (전체 문서 모두 출력하도록 수정됨)
    if os.path.exists(OUTPUT_DIR):
        print("\n--- 데이터 로드 테스트 (전체) ---")
        loader = load_paired_json_data(OUTPUT_DIR)
        
        total_docs = 0
        total_chunks = 0
        
        # Generator를 순회하며 모든 파일의 데이터를 출력합니다.
        for base_name, meta_data, hybrid_data in loader:
            total_docs += 1
            chunk_count = len(hybrid_data)
            total_chunks += chunk_count
            
            print(f"[{total_docs}] 로드 완료: {base_name}")
            print(f"   -> 메타데이터 항목 수: {len(meta_data)} / 하이브리드 청크 수: {chunk_count}")
            
        if total_docs == 0:
            print("처리할 파일이 없습니다.")
        else:
            print(f"\n✅ 요약: 총 {total_docs}개의 문서와 {total_chunks}개의 청크가 성공적으로 로드 및 매핑되었습니다.")
            
        # 3. 매핑된 데이터를 실제 파일로 저장하는 테스트 추가
        print("\n--- 데이터 물리적 저장 테스트 ---")
        save_mapped_data(OUTPUT_DIR, PROCESSED_DIR)