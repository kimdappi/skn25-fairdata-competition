#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <router_backend> <experiment_tag> <port> <cuda_visible_devices>" >&2
  exit 1
fi

ROUTER_BACKEND="$1"
EXPERIMENT_TAG="$2"
PORT="$3"
CUDA_DEVICES="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
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
export FAIRDATA_QUESTION_ROUTER_BACKEND="${ROUTER_BACKEND}"
export FAIRDATA_ROUTE_TAGS_PATH="cache/routes/${EXPERIMENT_TAG}/route_tags.json"
export FAIRDATA_INDEX_NAMESPACE="$(echo "${EXPERIMENT_TAG}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
export FAIRDATA_EXPERIMENT_TAG="${EXPERIMENT_TAG}"

RESULTS_DIR="results/${EXPERIMENT_TAG}"
LOG="/tmp/${FAIRDATA_INDEX_NAMESPACE}_server.log"
EVAL_LOG="/tmp/${FAIRDATA_INDEX_NAMESPACE}_eval.log"
PID_FILE="/tmp/${FAIRDATA_INDEX_NAMESPACE}_server.pid"

mkdir -p "${RESULTS_DIR}"
mkdir -p "$(dirname "${FAIRDATA_ROUTE_TAGS_PATH}")"
rm -f "${LOG}" "${EVAL_LOG}" "${PID_FILE}"
fuser -k "${PORT}/tcp" 2>/dev/null || true

python3 -u scripts/build_route_tags.py \
  --data-dir ./data/raw \
  --eval-file ./data/test/eval_dataset_260505.json \
  --output-file "${FAIRDATA_ROUTE_TAGS_PATH}" \
  --router-backend "${ROUTER_BACKEND}" \
  --documents-only \
  --max-concurrency "${FAIRDATA_ROUTE_MAX_CONCURRENCY}" \
  | tee "${RESULTS_DIR}/route_tags.log"

python3 -u scripts/build_indexes.py | tee "${RESULTS_DIR}/build.log"
nohup python3 -u -m uvicorn server:app --host 0.0.0.0 --port "${PORT}" > "${LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/health" > "${RESULTS_DIR}/health.json"

python3 -u scripts/evaluate_local.py \
  --eval-file ./data/test/eval_dataset_260505.json \
  --base-url "http://127.0.0.1:${PORT}" \
  --results-dir "${RESULTS_DIR}" \
  --experiment-tag "${EXPERIMENT_TAG}" | tee "${EVAL_LOG}"
