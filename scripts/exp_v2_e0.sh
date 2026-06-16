#!/usr/bin/env bash
set -euo pipefail
cd /workspace/skn25-fairdata-competition

PY=/root/venvs/fairdata-a100/bin/python
export PATH=/root/venvs/fairdata-a100/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export FAIRDATA_ENABLE_DENSE=1
export FAIRDATA_ENABLE_SPARSE=1
export FAIRDATA_ENABLE_MULTIVECTOR=1
export FAIRDATA_DENSE_BACKEND=bgem3
export FAIRDATA_SPARSE_BACKEND=bgem3
export FAIRDATA_MULTIVECTOR_BACKEND=bgem3
export FAIRDATA_MULTIVECTOR_BATCH_SIZE=${FAIRDATA_MULTIVECTOR_BATCH_SIZE:-16}
export FAIRDATA_MULTIVECTOR_SHARD_SIZE=${FAIRDATA_MULTIVECTOR_SHARD_SIZE:-128}
export FAIRDATA_RERANK_BACKEND=${FAIRDATA_RERANK_BACKEND:-bge_reranker}
export FAIRDATA_EXPERIMENT_TAG=V2-E0
PORT=8000
RESULTS_DIR=results/V2-E0
LOG=/tmp/v2_e0_server.log
EVAL_LOG=/tmp/v2_e0_eval.log

mkdir -p "$RESULTS_DIR"
rm -f "$LOG" "$EVAL_LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true

$PY -u scripts/build_indexes.py | tee "$RESULTS_DIR/build.log"
nohup $PY -u -m uvicorn server:app --host 0.0.0.0 --port ${PORT} > "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > /tmp/v2_e0_server.pid

for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/health" > "$RESULTS_DIR/health.json"

$PY -u scripts/evaluate_local.py \
  --eval-file ./data/test/eval_dataset_260505.json \
  --base-url "http://127.0.0.1:${PORT}" \
  --results-dir "$RESULTS_DIR" \
  --experiment-tag V2-E0 | tee "$EVAL_LOG"
