"""
================================================================================
[공정위 AI 데이터 활용 공모전 - RAG 파이프라인]
마스터 컨트롤 스크립트 (Project Orchestrator / Entry Point)
================================================================================

이 스크립트는 거대한 RAG 파이프라인 전체를 지휘하는 '마스터 사령탑(Orchestrator)'입니다.
프로젝트의 1단계(사전 공사)와 2단계(서버 구동)를 분리하거나, 혹은 한 번에 연속으로 
실행할 수 있도록 제어하는 통합 진입점(Entry Point) 역할을 합니다.

[전체 프로젝트에서의 핵심 역할]
1. Phase 1 (사전 공사) 자동화:
   - 제공된 원본 ZIP 파일을 압축 해제하고 정규화합니다. (`src.data_loader`)
   - 3만 개의 청크에 메타데이터를 결합하고 FAISS 및 BM25 인덱스를 굽습니다. (`src.build_index`)
   - 이 과정은 시간이 오래 걸리므로, 평소에는 생략하고 데이터가 변경되었을 때만 
     선택적으로 실행할 수 있게 분리해 두었습니다.

2. Phase 2 (실전 영업 / 서버 구동) 통제:
   - 구축된 데이터베이스와 AI 모델을 기반으로 FastAPI 서버를 구동합니다. (`server.py`)
   - 주최측 평가 서버와 통신할 준비를 마칩니다.

3. 통합 CLI (Command Line Interface) 제공:
   - 터미널에서 파이썬 파일들을 일일이 찾아다니며 실행할 필요 없이, 
     `python main.py --help`를 통해 원하는 파이프라인만 골라서 실행할 수 있는 
     직관적이고 프로페셔널한 인터페이스를 제공합니다.
================================================================================
"""

import os
import argparse
import uvicorn
import time

# 내부 모듈 임포트
from src.data_loader import extract_and_classify_zip
from src.build_index import build_vector_database

# ---------------------------------------------------------
# [환경 설정] 글로벌 경로 상수 (디렉토리 구조에 맞게 설정)
# ---------------------------------------------------------
ZIP_FILE_PATH = './data/공개본 의결서.zip'     # 원본 압축 파일 경로
RAW_DATA_DIR = './data/raw'                 # 압축 해제될 폴더 경로
VECTOR_DB_DIR = './models/vector_db'               # 인덱스 파일이 저장될 폴더 경로
EMBEDDING_MODEL_PATH = './models/local_embedding' # 로컬 임베딩 모델 경로

def run_phase_1_build():
    """
    [Phase 1: 사전 공사] 
    데이터 압축 해제 및 벡터/키워드 데이터베이스 구축을 수행합니다.
    """
    print("\n" + "="*60)
    print("🚀 [Phase 1] 데이터베이스 사전 구축 파이프라인 시작")
    print("="*60)
    
    start_time = time.time()
    
    # 1. 압축 해제 및 분류
    if os.path.exists(ZIP_FILE_PATH):
        print("\n▶ 1단계: 원본 ZIP 파일 압축 해제 및 분류")
        extract_and_classify_zip(ZIP_FILE_PATH, RAW_DATA_DIR)
    else:
        print(f"\n⚠️ 경고: '{ZIP_FILE_PATH}' 파일이 없습니다. 이미 해제되었거나 경로를 확인하세요.")
        
    # 2. 벡터 DB 및 BM25 인덱스 생성
    if os.path.exists(os.path.join(RAW_DATA_DIR, 'Hybrid')):
        print("\n▶ 2단계: FAISS 및 BM25 인덱스 파일 생성 (시간 소요)")
        build_vector_database(RAW_DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL_PATH)
    else:
        print(f"\n❌ 오류: '{RAW_DATA_DIR}' 폴더에 처리할 데이터가 없습니다.")
        return False
        
    elapsed_time = time.time() - start_time
    print(f"\n✅ [Phase 1 완료] 총 소요 시간: {elapsed_time:.2f}초")
    return True

def run_phase_2_serve():
    """
    [Phase 2: 실전 영업] 
    구축된 DB와 모델을 바탕으로 FastAPI 서버를 구동합니다.
    """
    print("\n" + "="*60)
    print("🚀 [Phase 2] FastAPI 추론 서버 구동 시작")
    print("="*60)
    
    # 필수 파일 존재 여부 사전 점검
    required_files = [
        os.path.join(VECTOR_DB_DIR, 'faiss_index.bin'),
        os.path.join(VECTOR_DB_DIR, 'bm25_index.pkl')
    ]
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"❌ 치명적 오류: 데이터베이스 파일('{filepath}')이 없습니다.")
            print("💡 해결 방법: 'python main.py --build' 명령어를 먼저 실행하여 DB를 생성하세요.")
            return

    # uvicorn을 사용하여 server.py 안의 app 객체를 실행
    print("\n▶ 서버 부팅 중... (GPU 모델 적재로 인해 1~2분 소요될 수 있습니다.)")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    # 터미널 명령어(CLI) 파서 설정
    parser = argparse.ArgumentParser(description="공정위 RAG 프로젝트 마스터 컨트롤러")
    
    parser.add_argument('--build', action='store_true', help="Phase 1만 실행: 원본 데이터 추출 및 벡터 DB 구축")
    parser.add_argument('--serve', action='store_true', help="Phase 2만 실행: API 서버 구동")
    parser.add_argument('--all', action='store_true', help="Phase 1 & 2 모두 실행: DB 구축 후 즉시 서버 구동")
    
    args = parser.parse_args()
    
    # 아무 옵션도 주지 않고 'python main.py'만 쳤을 때의 기본 동작 안내
    if not (args.build or args.serve or args.all):
        parser.print_help()
        print("\n💡 팁: 처음 실행하신다면 'python main.py --all'을 권장합니다.")
        exit(0)
        
    # 옵션에 따른 실행 분기
    if args.all:
        success = run_phase_1_build()
        if success:
            run_phase_2_serve()
    else:
        if args.build:
            run_phase_1_build()
        if args.serve:
            run_phase_2_serve()