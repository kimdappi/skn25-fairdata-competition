#!/usr/bin/env bash
set -euo pipefail
cd /workspace/skn25-fairdata-competition

# HTML 기준 E2:
# bgem3 + dense+sparse(learned sparse) + reranker
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export FAIRDATA_BERTSCORE_DEVICE=cpu
export FAIRDATA_ENABLE_DENSE=1
export FAIRDATA_ENABLE_SPARSE=1
export FAIRDATA_ENABLE_MULTIVECTOR=0
export FAIRDATA_DENSE_BACKEND=bgem3
export FAIRDATA_SPARSE_BACKEND=bgem3
export FAIRDATA_EXPERIMENT_TAG=V2-E2-BGEM3-DS
PORT=8102
RESULTS_DIR=results/V2-E2-BGEM3-DS
LOG=/tmp/v2_e2_bgem3_ds_server.log
EVAL_LOG=/tmp/v2_e2_bgem3_ds_eval.log
PID_FILE=/tmp/v2_e2_bgem3_ds_server.pid

mkdir -p "$RESULTS_DIR"
rm -f "$LOG" "$EVAL_LOG" "$PID_FILE"
fuser -k ${PORT}/tcp 2>/dev/null || true

python3 -u scripts/build_indexes.py | tee "$RESULTS_DIR/build.log"
nohup python3 -u -m uvicorn server:app --host 0.0.0.0 --port ${PORT} > "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/health" > "$RESULTS_DIR/health.json"

python3 -u scripts/evaluate_local.py \
  --eval-file ./data/test/eval_dataset_260505.json \
  --base-url "http://127.0.0.1:${PORT}" \
  --results-dir "$RESULTS_DIR" \
  --experiment-tag V2-E2-BGEM3-DS | tee "$EVAL_LOG"
