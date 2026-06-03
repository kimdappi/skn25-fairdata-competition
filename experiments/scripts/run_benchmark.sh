#!/bin/bash
# ==============================================================================
# RunPod 벤치마크 실행 스크립트
# ==============================================================================
# 전제: setup_runpod.sh 완료 + source .env_benchmark
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${FAIRCOMP_PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
cd "$PROJECT_DIR"

echo "============================================"
echo "🏃 3-Model Benchmark 시작"
echo "   프로젝트: $PROJECT_DIR"
echo "   모델 디렉토리: ${FAIRCOMP_MODEL_DIR:-not set}"
echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "   메모리: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "   시간: $(date)"
echo "============================================"

# GPU 상태 확인
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv 2>/dev/null | tail -1

# 실행
python benchmark_three_models.py --queries 600

echo ""
echo "✅ 벤치마크 완료!"
echo "결과: $PROJECT_DIR/results/benchmarks/comparison_*.{json,csv}"
ls -lh "$PROJECT_DIR/results/benchmarks/" 2>/dev/null
