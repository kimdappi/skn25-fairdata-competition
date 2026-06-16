#!/usr/bin/env bash
set -euo pipefail
cd /workspace/skn25-fairdata-competition

PY="${PYTHON_BIN:-/root/venvs/fairdata-a100/bin/python}"

# HTML 기준 E4:
# dense model(e5) + BM25 + reranker
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export FAIRDATA_BERTSCORE_DEVICE=cpu
export FAIRDATA_DENSE_DEVICE="${FAIRDATA_DENSE_DEVICE:-cuda:0}"
export FAIRDATA_ENABLE_DENSE=1
export FAIRDATA_ENABLE_SPARSE=1
export FAIRDATA_ENABLE_MULTIVECTOR=0
export FAIRDATA_DENSE_BACKEND=e5
export FAIRDATA_SPARSE_BACKEND=bm25
export FAIRDATA_LLM_DEVICE="${FAIRDATA_LLM_DEVICE:-cuda:0}"
export FAIRDATA_SKIP_LLM_WARMUP="${FAIRDATA_SKIP_LLM_WARMUP:-1}"
export FAIRDATA_EXPERIMENT_TAG=V2-E4-E5-BM25
PORT=8004
RESULTS_DIR=results/V2-E4-E5-BM25
LOG=/tmp/v2_e4_e5_bm25_server.log
EVAL_LOG=/tmp/v2_e4_e5_bm25_eval.log
PID_FILE=/tmp/v2_e4_e5_bm25_server.pid

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
  --experiment-tag V2-E4-E5-BM25 | tee "$EVAL_LOG"
