#!/usr/bin/env bash
set -euo pipefail
source /workspace/skn25-fairdata-competition/scripts/v2_e0_common_env.sh

rm -f "$V2_E0_EVAL_PID_FILE"
: > "$V2_E0_EVAL_TRAP_LOG"
exec >> "$V2_E0_EVAL_LOG" 2>&1

echo $$ > "$V2_E0_EVAL_PID_FILE"
echo "[$(date --iso-8601=seconds)] eval:start pid=$$ ppid=$PPID python=$V2_E0_PY base_url=http://127.0.0.1:${V2_E0_PORT} timeout=$V2_E0_EVAL_TIMEOUT cwd=$(pwd)"

on_exit() {
  rc=$?
  echo "[$(date --iso-8601=seconds)] EXIT rc=$rc pid=$$" >> "$V2_E0_EVAL_TRAP_LOG"
  rm -f "$V2_E0_EVAL_PID_FILE"
  exit "$rc"
}
on_term() {
  sig="$1"
  echo "[$(date --iso-8601=seconds)] SIGNAL $sig pid=$$" >> "$V2_E0_EVAL_TRAP_LOG"
}
trap on_exit EXIT
trap 'on_term TERM' TERM
trap 'on_term HUP' HUP
trap 'on_term INT' INT

curl -fsS "http://127.0.0.1:${V2_E0_PORT}/health" > "$V2_E0_HEALTH_JSON"

"$V2_E0_PY" -u scripts/evaluate_local.py \
  --eval-file "$V2_E0_EVAL_FILE" \
  --base-url "http://127.0.0.1:${V2_E0_PORT}" \
  --results-dir "$V2_E0_RESULTS_DIR" \
  --experiment-tag "$FAIRDATA_EXPERIMENT_TAG" \
  --timeout "$V2_E0_EVAL_TIMEOUT"

echo "[$(date --iso-8601=seconds)] eval:complete pid=$$"
