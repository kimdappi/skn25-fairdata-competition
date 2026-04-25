# ==============================================================================
# 공정위 AI 데이터 활용 공모전 - 최종 제출용 Dockerfile
# ==============================================================================

# 1. 베이스 이미지 설정: PyTorch 및 CUDA 12.1이 기본 탑재된 공식 이미지
# (vLLM과 ONNX Runtime-GPU가 완벽하게 호환되는 안정적인 환경입니다)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시간대 설정 (서울) 및 시스템 패키지 업데이트
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Seoul

# 4. 파이썬 라이브러리 설치
# (의존성 충돌 방지를 위해 vLLM을 먼저 독립적으로 설치하는 것을 권장합니다)
COPY requirements.txt ./
RUN pip install --no-cache-dir vllm>=0.6.0 --no-deps
RUN pip install --no-cache-dir -r requirements.txt

# 5. RAG 파이프라인 소스코드 복사
COPY src/ ./src/
COPY server.py ./

# 6. [매우 중요] 오프라인 인덱스 및 모델 가중치 복사
# 오프라인망에서 동작해야 하므로, 로컬에 구축된 모든 DB와 모델이 컨테이너 내부로 들어와야 합니다.
COPY models/ ./models/

# 7. 평가 서버 통신을 위한 포트 개방
EXPOSE 8000

# 8. 컨테이너 실행 시 FastAPI 서버 구동
CMD ["python", "server.py"]