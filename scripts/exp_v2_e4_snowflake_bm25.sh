#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# HTML 기준 E4:
# snowflake dense + BM25 + reranker + EXAONE generation
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export FAIRDATA_BERTSCORE_DEVICE="${FAIRDATA_BERTSCORE_DEVICE:-cuda}"
  RUNTIME_DEVICE="gpu"
else
  unset CUDA_VISIBLE_DEVICES
  export FAIRDATA_BERTSCORE_DEVICE="${FAIRDATA_BERTSCORE_DEVICE:-cpu}"
  RUNTIME_DEVICE="cpu"
fi

export PYTHONUNBUFFERED=1
export FAIRDATA_ENABLE_DENSE=1
export FAIRDATA_ENABLE_SPARSE=1
export FAIRDATA_ENABLE_MULTIVECTOR=0
export FAIRDATA_DENSE_BACKEND=snowflake_ko
export FAIRDATA_SPARSE_BACKEND=bm25
export FAIRDATA_LLM_BACKEND=exaone-3.5-7.8b-instruct
export FAIRDATA_EXPERIMENT_TAG=SNOWFLAKE-BM25-EXAONE
PORT=8004
RESULTS_DIR="results/${FAIRDATA_EXPERIMENT_TAG}"
LOG="/tmp/${FAIRDATA_EXPERIMENT_TAG,,}_server.log"
EVAL_LOG="/tmp/${FAIRDATA_EXPERIMENT_TAG,,}_eval.log"
PID_FILE="/tmp/${FAIRDATA_EXPERIMENT_TAG,,}_server.pid"

mkdir -p "$RESULTS_DIR"
rm -f "$LOG" "$EVAL_LOG" "$PID_FILE"
fuser -k ${PORT}/tcp 2>/dev/null || true

echo "[exp_v2_e4_snowflake_bm25] repo_root=${REPO_ROOT}"
echo "[exp_v2_e4_snowflake_bm25] runtime_device=${RUNTIME_DEVICE}"
echo "[exp_v2_e4_snowflake_bm25] llm_backend=${FAIRDATA_LLM_BACKEND}"

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
  --experiment-tag "$FAIRDATA_EXPERIMENT_TAG" | tee "$EVAL_LOG"
