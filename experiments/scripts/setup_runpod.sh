#!/bin/bash
# ==============================================================================
# RunPod GPU 환경 초기화 스크립트
# 공정위 AI 공모전 - 3개 모형 600-query 벤치마크
# ==============================================================================
#
# RunPod 템플릿 권장: RunPod PyTorch 2.4+ / CUDA 12.4+
# GPU: RTX 3090 or A4000 (24GB VRAM)
# Disk: 50GB+ (models = ~18GB)
#
# 사용법:
#   bash setup_runpod.sh
#   source ~/.bashrc
#   bash run_benchmark.sh
# ==============================================================================

set -euo pipefail
echo "🚀 RunPod Benchmark Setup 시작 — $(date)"

# ── 1. 시스템 의존성 ──
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake 2>&1 | tail -1

# ── 2. Python 환경 ──
echo "[2/6] Setting up Python environment..."
python -m pip install --upgrade pip setuptools wheel 2>&1 | tail -1

# ── 3. 프로젝트 클론 또는 복사 ──
echo "[3/6] Preparing project..."
WORKSPACE="${WORKSPACE:-/workspace}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE/skn25-fairdata-competition}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "  Cloning project..."
    git clone https://github.com/kimdappi/skn25-fairdata-competition.git "$PROJECT_DIR" 2>/dev/null || {
        echo "  ⚠️ git clone 실패 — 수동으로 프로젝트를 $PROJECT_DIR 에 복사하세요"
        mkdir -p "$PROJECT_DIR"
    }
fi
cd "$PROJECT_DIR"

# ── 4. Python 패키지 설치 ──
echo "[4/6] Installing Python packages..."
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -3

# 추가 의존성
pip install --no-cache-dir \
    fastapi uvicorn \
    onnxruntime-gpu \
    accelerate \
    bitsandbytes 2>&1 | tail -2

# langgraph는 requirements에 있음 — 확인
pip install langgraph==1.0.10 2>&1 | tail -1

# ── 5. 모델 다운로드 ──
echo "[5/6] Downloading models (BGE-M3 + Qwen2.5-7B)..."
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
mkdir -p "$MODEL_DIR"

python3 -c "
from huggingface_hub import snapshot_download
import os

model_dir = os.environ.get('MODEL_DIR', './models')
print(f'  Models will be saved to: {model_dir}')

# BGE-M3 임베딩 모델 (2.6 GB)
print('  Downloading BAAI/bge-m3...')
snapshot_download(
    'BAAI/bge-m3',
    local_dir=f'{model_dir}/BAAI/bge-m3',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('  ✅ BGE-M3 downloaded')

# Qwen2.5-7B-Instruct 라우터 모델 (15 GB)
print('  Downloading Qwen/Qwen2.5-7B-Instruct...')
snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct',
    local_dir=f'{model_dir}/Qwen2.5-7B-Instruct',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('  ✅ Qwen2.5-7B-Instruct downloaded')
"
echo "[5/6] ✅ Models downloaded"

# ── 6. 환경변수 설정 ──
echo "[6/6] Configuring environment..."

ENV_FILE="$WORKSPACE/.env_benchmark"
cat > "$ENV_FILE" << 'ENVEOF'
# 공정위 공모전 벤치마크 환경변수
export FAIRCOMP_PROJECT_DIR="/workspace/skn25-fairdata-competition"
export FAIRCOMP_MODEL_DIR="/workspace/skn25-fairdata-competition/models"
export FAIRCOMP_DATA_DIR="/workspace/skn25-fairdata-competition/search_raw_data"
export FAIRCOMP_METRICS_PATH="/workspace/skn25-fairdata-competition/metrics.json"
export FAIRCOMP_RESULT_DIR="/workspace/skn25-fairdata-competition/results"

# Router 설정 (Ollama → local HF)
export ROUTER_BACKEND="hf"
export HF_ROUTER_MODEL="Qwen2.5-7B-Instruct"
export HF_ROUTER_LOCAL_FILES_ONLY="true"

# GPU 설정
export FAIRCOMP_REQUIRE_CUDA="true"
export STAGE_LOG_ENABLED="true"
export QUERY_TIMING_LOG_ENABLED="true"

# PyTorch 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENVEOF

# ~/.bashrc에 추가
grep -q "FAIRCOMP_" ~/.bashrc 2>/dev/null || {
    echo "" >> ~/.bashrc
    echo "# 공정위 공모전 벤치마크 설정" >> ~/.bashrc
    echo "source $ENV_FILE" >> ~/.bashrc
}
source "$ENV_FILE"

echo ""
echo "============================================"
echo "✅ RunPod 설정 완료!"
echo ""
echo "실행 명령어:"
echo "  source $ENV_FILE"
echo "  cd $PROJECT_DIR"
echo ""
echo "  # 600 query 전체 벤치마크"
echo "  python benchmark_three_models.py --queries 600"
echo ""
echo "  # 50 query 빠른 테스트"
echo "  python benchmark_three_models.py --queries 50"
echo ""
echo "  # 모형 3만 테스트"
echo "  python -c '...'  # README 참조"
echo "============================================"
