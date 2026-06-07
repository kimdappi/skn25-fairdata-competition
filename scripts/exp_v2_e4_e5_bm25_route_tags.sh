#!/usr/bin/env bash
set -euo pipefail
cd /workspace/skn25-fairdata-competition

# HTML 기준 E4 + LLM route tags:
# dense model(e5) + BM25 + reranker + route-tag candidate filtering
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export FAIRDATA_BERTSCORE_DEVICE=cpu
export FAIRDATA_ENABLE_DENSE=1
export FAIRDATA_ENABLE_SPARSE=1
export FAIRDATA_ENABLE_MULTIVECTOR=0
export FAIRDATA_DENSE_BACKEND=e5
export FAIRDATA_SPARSE_BACKEND=bm25
export FAIRDATA_ROUTE_FILTER=1
export FAIRDATA_ROUTE_MIN_CANDIDATE_DOCS=20
export FAIRDATA_ROUTE_MAX_CONCURRENCY=4
export FAIRDATA_ROUTE_MAX_NEW_TOKENS=64
export FAIRDATA_GENERATION_MAX_INPUT_CHARS=3500
export FAIRDATA_GENERATION_MAX_NEW_TOKENS=160
export FAIRDATA_LLM_DEVICE=cuda:0
export FAIRDATA_RERANK_DEVICE=cuda:1
export FAIRDATA_ROUTE_TAGS_PATH=cache/routes/V2-E4-E5-BM25-ROUTE/route_tags.json
export FAIRDATA_INDEX_NAMESPACE=v2_e4_e5_bm25_route_tags
export FAIRDATA_EXPERIMENT_TAG=V2-E4-E5-BM25-ROUTE
PORT=8014
RESULTS_DIR=results/V2-E4-E5-BM25-ROUTE
LOG=/tmp/v2_e4_e5_bm25_route_server.log
EVAL_LOG=/tmp/v2_e4_e5_bm25_route_eval.log
PID_FILE=/tmp/v2_e4_e5_bm25_route_server.pid

mkdir -p "$RESULTS_DIR"
mkdir -p "$(dirname "$FAIRDATA_ROUTE_TAGS_PATH")"
rm -f "$LOG" "$EVAL_LOG" "$PID_FILE"
fuser -k ${PORT}/tcp 2>/dev/null || true

python3 -u scripts/build_route_tags.py \
  --data-dir ./data/raw \
  --eval-file ./data/test/eval_dataset_260505.json \
  --output-file "$FAIRDATA_ROUTE_TAGS_PATH" \
  --router-backend llm \
  --documents-only \
  --max-concurrency "$FAIRDATA_ROUTE_MAX_CONCURRENCY" \
  | tee "$RESULTS_DIR/route_tags.log"

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
  --experiment-tag V2-E4-E5-BM25-ROUTE | tee "$EVAL_LOG"
