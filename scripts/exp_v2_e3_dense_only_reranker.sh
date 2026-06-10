#!/usr/bin/env bash
set -euo pipefail
cd /workspace/skn25-fairdata-competition

PY="${PYTHON_BIN:-/root/venvs/fairdata-a100/bin/python}"

# HTML 기준 E3:
# jina 또는 gte + dense only + reranker
# 기본 backend는 jina_v4, 필요 시 첫 번째 인자로 gte / gte_multilingual / jina / jina_v4 전달
DENSE_BACKEND_INPUT="${1:-jina_v4}"
case "$DENSE_BACKEND_INPUT" in
  jina|jina_v4|jina_embeddings_v4)
    DENSE_BACKEND="jina_v4"
    EXP_SUFFIX="JINA"
    PORT=8003
    ;;
  gte|gte_multilingual|gte_multilingual_base)
    DENSE_BACKEND="gte_multilingual"
    EXP_SUFFIX="GTE"
    PORT=8005
    ;;
  *)
    echo "[E3] unsupported dense backend: $DENSE_BACKEND_INPUT" >&2
    echo "[E3] allowed: jina_v4|jina|gte_multilingual|gte" >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export FAIRDATA_BERTSCORE_DEVICE=cpu
export FAIRDATA_DENSE_DEVICE="${FAIRDATA_DENSE_DEVICE:-cuda:0}"
export FAIRDATA_ENABLE_DENSE=1
export FAIRDATA_ENABLE_SPARSE=0
export FAIRDATA_ENABLE_MULTIVECTOR=0
export FAIRDATA_DENSE_BACKEND="$DENSE_BACKEND"
export FAIRDATA_LLM_DEVICE="${FAIRDATA_LLM_DEVICE:-cuda:0}"
export FAIRDATA_SKIP_LLM_WARMUP="${FAIRDATA_SKIP_LLM_WARMUP:-1}"
export FAIRDATA_EXPERIMENT_TAG="V2-E3-${EXP_SUFFIX}-DENSE"
RESULTS_DIR="results/${FAIRDATA_EXPERIMENT_TAG}"
LOG="/tmp/${FAIRDATA_EXPERIMENT_TAG,,}_server.log"
EVAL_LOG="/tmp/${FAIRDATA_EXPERIMENT_TAG,,}_eval.log"
PID_FILE="/tmp/${FAIRDATA_EXPERIMENT_TAG,,}_server.pid"

mkdir -p "$RESULTS_DIR"
rm -f "$LOG" "$EVAL_LOG" "$PID_FILE"
fuser -k ${PORT}/tcp 2>/dev/null || true

"$PY" -u scripts/build_indexes.py | tee "$RESULTS_DIR/build.log"
nohup "$PY" -u -m uvicorn server:app --host 0.0.0.0 --port ${PORT} > "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/health" > "$RESULTS_DIR/health.json"

"$PY" -u scripts/evaluate_local.py \
  --eval-file ./data/test/eval_dataset_260505.json \
  --base-url "http://127.0.0.1:${PORT}" \
  --results-dir "$RESULTS_DIR" \
  --experiment-tag "$FAIRDATA_EXPERIMENT_TAG" | tee "$EVAL_LOG"
