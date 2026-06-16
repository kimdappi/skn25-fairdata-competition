#!/usr/bin/env bash
set -uo pipefail
cd /workspace/skn25-fairdata-competition
PY="${PYTHON_BIN:-/root/venvs/fairdata-a100/bin/python}"
if [ ! -x "$PY" ]; then PY=python3; fi

EXP_TAG="V2-E3-GTE-DENSE-TOP75"
export EXP_TAG
PORT="8017"
RESULTS_DIR="results/${EXP_TAG}"
LOG="/tmp/${EXP_TAG,,}_server.log"
EVAL_LOG="/tmp/${EXP_TAG,,}_eval.log"
PID_FILE="/tmp/${EXP_TAG,,}_server.pid"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export FAIRDATA_BERTSCORE_DEVICE=cpu
export FAIRDATA_DENSE_DEVICE="${FAIRDATA_DENSE_DEVICE:-cuda:0}"
export FAIRDATA_LLM_DEVICE="${FAIRDATA_LLM_DEVICE:-cuda:0}"
export FAIRDATA_SKIP_LLM_WARMUP="${FAIRDATA_SKIP_LLM_WARMUP:-1}"
export FAIRDATA_EXPERIMENT_TAG="$EXP_TAG"
export FAIRDATA_ENABLE_DENSE=1
export FAIRDATA_ENABLE_SPARSE=0
export FAIRDATA_ENABLE_MULTIVECTOR=0
export FAIRDATA_DENSE_BACKEND=gte_multilingual
export FAIRDATA_SPARSE_BACKEND=bm25
export FAIRDATA_RERANK_BACKEND="${FAIRDATA_RERANK_BACKEND:-bge_reranker}"
export FAIRDATA_RERANK_TOP_N=75
export FAIRDATA_RERANK_WEIGHT="${FAIRDATA_RERANK_WEIGHT:-1.0}"
export FAIRDATA_INDEX_NAMESPACE="dense_gte_multilingual__sparse_off__multivector_off__rtop75"


mkdir -p "$RESULTS_DIR"
rm -f "$LOG" "$EVAL_LOG" "$PID_FILE"
fuser -k ${PORT}/tcp 2>/dev/null || true

echo "[RUNNER] START ${EXP_TAG} $(date -Is)"
echo "[RUNNER] ENV dense=${FAIRDATA_DENSE_BACKEND} sparse=${FAIRDATA_SPARSE_BACKEND} dense_top=${FAIRDATA_DENSE_TOP_K:-default} sparse_top=${FAIRDATA_SPARSE_TOP_K:-default} rerank_top=${FAIRDATA_RERANK_TOP_N} namespace=${FAIRDATA_INDEX_NAMESPACE}"

"$PY" -u scripts/build_indexes.py > >(tee "$RESULTS_DIR/build.log") 2>&1
BUILD_RC=${PIPESTATUS[0]}
if [ "$BUILD_RC" -ne 0 ]; then
  echo "[HERMES_EXPERIMENT_FAILED] ${EXP_TAG} stage=build rc=${BUILD_RC} results=/workspace/skn25-fairdata-competition/${RESULTS_DIR}"
  exit "$BUILD_RC"
fi

"$PY" -u -m uvicorn server:app --host 0.0.0.0 --port ${PORT} > "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

READY=0
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" > "$RESULTS_DIR/health.json" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "[HERMES_EXPERIMENT_FAILED] ${EXP_TAG} stage=server_health rc=1 results=/workspace/skn25-fairdata-competition/${RESULTS_DIR} server_log=${LOG}"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

"$PY" -u scripts/evaluate_local.py \
  --eval-file ./data/test/eval_dataset_260505.json \
  --base-url "http://127.0.0.1:${PORT}" \
  --timeout 420 \
  --results-dir "$RESULTS_DIR" \
  --experiment-tag "$EXP_TAG" > >(tee "$EVAL_LOG") 2>&1
EVAL_RC=${PIPESTATUS[0]}

curl -fsS "http://127.0.0.1:${PORT}/health" > "$RESULTS_DIR/health_final.json" 2>/dev/null || true
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

if [ "$EVAL_RC" -ne 0 ]; then
  echo "[HERMES_EXPERIMENT_FAILED] ${EXP_TAG} stage=eval rc=${EVAL_RC} results=/workspace/skn25-fairdata-competition/${RESULTS_DIR} eval_log=${EVAL_LOG}"
  exit "$EVAL_RC"
fi

"$PY" - <<'METRICS'
import json, pathlib, glob, os
exp=os.environ['EXP_TAG']
paths=glob.glob(f"results/{exp}/**/summary.json", recursive=True)
if not paths:
    print(f"[HERMES_EXPERIMENT_FAILED] {exp} stage=summary_missing results=/workspace/skn25-fairdata-competition/results/{exp}")
    raise SystemExit(2)
p=pathlib.Path(max(paths, key=lambda x: pathlib.Path(x).stat().st_mtime))
o=json.loads(p.read_text())
keys=['count','recall_at_5','mrr','token_f1','bertscore_f1','final_score']
metric=' '.join(f"{k}={o.get(k)}" for k in keys)
print(f"[HERMES_EXPERIMENT_DONE] {exp} {metric} summary=/workspace/skn25-fairdata-competition/{p}")
METRICS
