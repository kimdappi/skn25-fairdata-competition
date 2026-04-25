"""
================================================================================
[공정위 AI 데이터 활용 공모전 - RAG 파이프라인 Step 5]
FastAPI 기반 최종 제출용 API 서버 본체 (Server)
================================================================================

이 스크립트는 거대한 RAG 파이프라인의 '조종실(Controller)'이자 주최측 평가 서버와
소통하는 '유일한 출입구(API Endpoint)' 역할을 합니다.

[전체 프로젝트에서의 핵심 역할]
1. 모듈 통합 및 메모리 적재 (Initialization):
   - 서버가 부팅될 때(앱 실행 시), 앞서 Step 3와 Step 4에서 만든 `HybridRetriever`와 
     `RAGGenerator`를 호출하여 무거운 인덱스 파일(FAISS, BM25)과 AI 모델 가중치(ONNX, vLLM)를 
     A100 GPU 및 시스템 메모리에 단 한 번 적재합니다. 
   - 이 과정이 끝나야 비로소 '/health' 엔드포인트가 "ok"를 반환하며 평가받을 준비를 마칩니다.

2. 대회 규격에 맞춘 통신 인터페이스 (API Routing):
   - 주최측 평가 서버가 요구하는 엄격한 API 규격을 준수합니다.
   - [GET /health]: 서버와 AI 모델이 모두 성공적으로 준비되었는지 확인하는 상태 체크.
   - [POST /predict]: 실제 평가가 이루어지는 핵심 엔드포인트. 질문을 수신하고 답변을 반환.

3. 30초 타임아웃 방어 및 파이프라인 오케스트레이션 (Orchestration):
   - 질문(Query)이 들어오면 다음의 흐름을 지휘합니다:
     ① Retriever에 질문을 던져 5개의 정답 청크 ID와 본문을 1초 내로 받아옴
     ② Generator에 질문과 본문을 넘겨 vLLM 엔진으로 답변을 20여 초 내로 초고속 생성
     ③ 최종적으로 대회 규격에 맞춰 JSON 형태로 조립하여 30초 내에 신속히 반환
================================================================================
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os

# 우리가 작성한 RAG 파이프라인 모듈들 임포트
from src.retriever import HybridRetriever
from src.generator import RAGGenerator

# 1. 경로 설정 (오프라인 환경 기준 상대 경로)
# ---------------------------------------------------------
VECTOR_DB_DIR = './models/vector_db'
EMBEDDING_MODEL = './models/local_embedding'
ONNX_RERANKER = './models/onnx_reranker'
LLM_MODEL = './models/local_llm'

# ---------------------------------------------------------
# 2. FastAPI 앱 초기화 및 전역 변수 설정
# ---------------------------------------------------------
app = FastAPI(title="공정위 RAG API 서버", version="1.0")

# 글로벌 객체 선언 (서버 시작 시 한 번만 로드)
retriever = None
generator = None

# 평가 서버가 보내는 질문 포맷 규격
class PredictRequest(BaseModel):
    question: str

# 우리가 평가 서버로 반환해야 할 답변 포맷 규격
class PredictResponse(BaseModel):
    chunk_ids: List[str]
    answer: str

# ---------------------------------------------------------
# 3. 서버 시작 이벤트 (모델 및 DB 메모리 적재)
# ---------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """서버 부팅 시 가장 먼저 실행되어 무거운 모델들을 로드합니다."""
    global retriever, generator
    print("🚀 API 서버 부팅을 시작합니다. 모델 및 인덱스를 메모리에 적재 중...")
    
    try:
        # Step 3: 하이브리드 검색기 로드 (FAISS, BM25, ONNX Reranker)
        retriever = HybridRetriever(VECTOR_DB_DIR, EMBEDDING_MODEL, ONNX_RERANKER)
        
        # Step 4: vLLM 기반 생성기 로드
        generator = RAGGenerator(LLM_MODEL, gpu_utilization=0.85)
        
        print("✅ 모든 모델과 데이터베이스가 성공적으로 로드되었습니다!")
    except Exception as e:
        print(f"❌ 서버 초기화 중 오류 발생: {e}")
        # 오류 발생 시 서버가 비정상 상태임을 알 수 있도록 처리 필요
        raise RuntimeError("서버 초기화에 실패했습니다. 경로 및 파일을 확인하세요.")

# ---------------------------------------------------------
# 4. API 엔드포인트 구현
# ---------------------------------------------------------
@app.get("/health")
async def health_check():
    """
    주최측 평가 서버가 시스템의 준비 상태를 확인할 때 호출합니다.
    이 엔드포인트가 정상적으로 "ok"를 반환해야만 본격적인 predict 평가가 시작됩니다.
    """
    if retriever is not None and generator is not None:
        return {"status": "ok"}
    else:
        # 아직 로드가 덜 되었거나 실패한 경우 503 에러 반환
        raise HTTPException(status_code=503, detail="Server is not ready yet.")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    핵심 평가 엔드포인트입니다. 
    질문(Query)을 받아 검색(Retrieval)과 생성(Generation)을 거쳐 최종 결과를 반환합니다.
    """
    query = request.question.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="질문(question)이 비어있습니다.")
        
    print(f"\n[새로운 질문 수신]: {query}")
    
    try:
        # ① Retriever를 통해 검색 및 재정렬 수행 (Top 5 Chunk 추출)
        # 반드시 5개의 chunk_id가 반환되도록 retriever 내부에 Fallback 로직이 적용되어 있음
        chunk_ids, context = retriever.get_context_and_ids(query)
        
        # ② Generator를 통해 답변 생성
        answer = generator.generate_answer(query, context)
        
        print(f"[답변 생성 완료] chunk_ids: {chunk_ids}")
        
        # ③ 최종 JSON 포맷으로 반환
        return PredictResponse(
            chunk_ids=chunk_ids,
            answer=answer
        )
        
    except Exception as e:
        print(f"❌ 예측 수행 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 오류로 답변을 생성할 수 없습니다.")

# ---------------------------------------------------------
# 5. 로컬 실행용 엔트리 포인트
# ---------------------------------------------------------
if __name__ == "__main__":
    # 이 스크립트를 직접 실행할 경우 uvicorn을 통해 8000번 포트에서 서버가 열립니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)