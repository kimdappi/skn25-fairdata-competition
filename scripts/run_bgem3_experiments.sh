#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_EVAL_FILE="${PROJECT_DIR}/data/test/eval_dataset_260505.json"
DEFAULT_RESULTS_DIR="${PROJECT_DIR}/results/bgem3_experiments"
DEFAULT_BASE_URL_HOST="127.0.0.1"
DEFAULT_BASE_PORT="8000"
DEFAULT_UVICORN_HOST="0.0.0.0"
DEFAULT_TIMEOUT="30"

EVAL_FILE="${DEFAULT_EVAL_FILE}"
RESULTS_DIR="${DEFAULT_RESULTS_DIR}"
BASE_URL_HOST="${DEFAULT_BASE_URL_HOST}"
BASE_PORT="${DEFAULT_BASE_PORT}"
UVICORN_HOST="${DEFAULT_UVICORN_HOST}"
TIMEOUT="${DEFAULT_TIMEOUT}"
LIMIT="0"
OFFSET="0"
RUN_MODE="all"
EXPERIMENT_LIST=""

SERVER_PID=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_bgem3_experiments.sh [options]

Options:
  --eval-file PATH         평가셋 경로. 기본값: data/test/eval_dataset_260505.json
  --results-dir PATH       결과 저장 루트. 기본값: results/bgem3_experiments
  --base-url-host HOST     evaluate_local.py가 호출할 호스트. 기본값: 127.0.0.1
  --port PORT              서버 포트. 기본값: 8000
  --uvicorn-host HOST      uvicorn 바인딩 호스트. 기본값: 0.0.0.0
  --timeout SECONDS        evaluate_local 요청 타임아웃. 기본값: 30
  --limit N                평가 문항 제한. 0이면 전체
  --offset N               평가 시작 오프셋
  --mode MODE              core 또는 all. 기본값: all
  --experiments LIST       쉼표 구분 실험명 직접 지정. 예: full_hybrid,dense_only
  -h, --help               도움말 출력

Supported experiments:
  full_hybrid       dense+sparse+multivector
  dense_only        dense만 사용
  dense_sparse      dense+learned sparse
  dense_multivector dense+multivector
  sparse_only       sparse만 사용

Examples:
  bash scripts/run_bgem3_experiments.sh --mode core
  bash scripts/run_bgem3_experiments.sh --experiments full_hybrid,dense_sparse --limit 50
EOF
}

log() {
  printf '[run_bgem3_experiments] %s\n' "$*"
}

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

cleanup() {
  cleanup_server
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-file)
      EVAL_FILE="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --base-url-host)
      BASE_URL_HOST="$2"
      shift 2
      ;;
    --port)
      BASE_PORT="$2"
      shift 2
      ;;
    --uvicorn-host)
      UVICORN_HOST="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --offset)
      OFFSET="$2"
      shift 2
      ;;
    --mode)
      RUN_MODE="$2"
      shift 2
      ;;
    --experiments)
      EXPERIMENT_LIST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${EVAL_FILE}" ]]; then
  printf 'Eval file not found: %s\n' "${EVAL_FILE}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

resolve_experiments() {
  if [[ -n "${EXPERIMENT_LIST}" ]]; then
    printf '%s\n' "${EXPERIMENT_LIST}" | tr ',' '\n'
    return 0
  fi

  case "${RUN_MODE}" in
    core)
      printf '%s\n' "full_hybrid" "dense_only" "dense_sparse"
      ;;
    all)
      printf '%s\n' "full_hybrid" "dense_only" "dense_sparse" "dense_multivector" "sparse_only"
      ;;
    *)
      printf 'Unsupported mode: %s\n' "${RUN_MODE}" >&2
      exit 1
      ;;
  esac
}

wait_for_server() {
  local base_url="$1"
  local attempts=60
  local sleep_seconds=2

  for ((i=1; i<=attempts; i++)); do
    if python3 - "${base_url}" <<'PY'
import json
import sys
from urllib.request import urlopen

base_url = sys.argv[1].rstrip("/")
with urlopen(base_url + "/health", timeout=5) as response:
    payload = json.loads(response.read().decode("utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep "${sleep_seconds}"
  done

  return 1
}

run_experiment() {
  local name="$1"
  local enable_dense="1"
  local enable_sparse="1"
  local enable_multivector="1"

  case "${name}" in
    full_hybrid)
      enable_dense="1"
      enable_sparse="1"
      enable_multivector="1"
      ;;
    dense_only)
      enable_dense="1"
      enable_sparse="0"
      enable_multivector="0"
      ;;
    dense_sparse)
      enable_dense="1"
      enable_sparse="1"
      enable_multivector="0"
      ;;
    dense_multivector)
      enable_dense="1"
      enable_sparse="0"
      enable_multivector="1"
      ;;
    sparse_only)
      enable_dense="0"
      enable_sparse="1"
      enable_multivector="0"
      ;;
    *)
      printf 'Unsupported experiment: %s\n' "${name}" >&2
      exit 1
      ;;
  esac

  local experiment_tag="bgem3_${name}"
  local experiment_results_dir="${RESULTS_DIR}/${experiment_tag}"
  local log_dir="${experiment_results_dir}/logs"
  local server_log="${log_dir}/server.log"
  local build_log="${log_dir}/build_indexes.log"
  local base_url="http://${BASE_URL_HOST}:${BASE_PORT}"

  mkdir -p "${log_dir}"
  cleanup_server

  log "starting experiment=${experiment_tag}"
  log "results_dir=${experiment_results_dir}"

  (
    export FAIRDATA_EXPERIMENT_TAG="${experiment_tag}"
    export FAIRDATA_DENSE_BACKEND="bgem3"
    export FAIRDATA_SPARSE_BACKEND="bgem3"
    export FAIRDATA_MULTIVECTOR_BACKEND="bgem3"
    export FAIRDATA_RERANK_BACKEND="${FAIRDATA_RERANK_BACKEND:-bge_reranker}"
    export FAIRDATA_ENABLE_DENSE="${enable_dense}"
    export FAIRDATA_ENABLE_SPARSE="${enable_sparse}"
    export FAIRDATA_ENABLE_MULTIVECTOR="${enable_multivector}"
    export FAIRDATA_INDEX_NAMESPACE="${experiment_tag}"

    python3 scripts/build_indexes.py
  ) >"${build_log}" 2>&1

  (
    export FAIRDATA_EXPERIMENT_TAG="${experiment_tag}"
    export FAIRDATA_DENSE_BACKEND="bgem3"
    export FAIRDATA_SPARSE_BACKEND="bgem3"
    export FAIRDATA_MULTIVECTOR_BACKEND="bgem3"
    export FAIRDATA_RERANK_BACKEND="${FAIRDATA_RERANK_BACKEND:-bge_reranker}"
    export FAIRDATA_ENABLE_DENSE="${enable_dense}"
    export FAIRDATA_ENABLE_SPARSE="${enable_sparse}"
    export FAIRDATA_ENABLE_MULTIVECTOR="${enable_multivector}"
    export FAIRDATA_INDEX_NAMESPACE="${experiment_tag}"

    cd "${PROJECT_DIR}"
    python3 -m uvicorn server:app --host "${UVICORN_HOST}" --port "${BASE_PORT}"
  ) >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  if ! wait_for_server "${base_url}"; then
    log "server did not become healthy: ${server_log}"
    return 1
  fi

  (
    export FAIRDATA_EXPERIMENT_TAG="${experiment_tag}"
    export FAIRDATA_DENSE_BACKEND="bgem3"
    export FAIRDATA_SPARSE_BACKEND="bgem3"
    export FAIRDATA_MULTIVECTOR_BACKEND="bgem3"
    export FAIRDATA_RERANK_BACKEND="${FAIRDATA_RERANK_BACKEND:-bge_reranker}"
    export FAIRDATA_ENABLE_DENSE="${enable_dense}"
    export FAIRDATA_ENABLE_SPARSE="${enable_sparse}"
    export FAIRDATA_ENABLE_MULTIVECTOR="${enable_multivector}"
    export FAIRDATA_INDEX_NAMESPACE="${experiment_tag}"

    cd "${PROJECT_DIR}"
    python3 scripts/evaluate_local.py \
      --base-url "${base_url}" \
      --eval-file "${EVAL_FILE}" \
      --results-dir "${experiment_results_dir}" \
      --timeout "${TIMEOUT}" \
      --limit "${LIMIT}" \
      --offset "${OFFSET}" \
      --experiment-tag "${experiment_tag}"
  )

  cleanup_server
  log "finished experiment=${experiment_tag}"
}

mapfile -t experiments < <(resolve_experiments)

if [[ "${#experiments[@]}" -eq 0 ]]; then
  printf 'No experiments selected.\n' >&2
  exit 1
fi

log "project_dir=${PROJECT_DIR}"
log "eval_file=${EVAL_FILE}"
log "results_dir=${RESULTS_DIR}"
log "experiments=${experiments[*]}"

for experiment_name in "${experiments[@]}"; do
  run_experiment "${experiment_name}"
done

log "all experiments finished"
