#!/usr/bin/env bash
set -euo pipefail
source /workspace/skn25-fairdata-competition/scripts/v2_e0_common_env.sh

rm -f "$V2_E0_BUILD_PID_FILE"
: > "$V2_E0_BUILD_TRAP_LOG"
exec >> "$V2_E0_BUILD_LOG" 2>&1

echo $$ > "$V2_E0_BUILD_PID_FILE"
echo "[$(date --iso-8601=seconds)] build:start pid=$$ ppid=$PPID python=$V2_E0_PY cwd=$(pwd)" 

on_exit() {
  rc=$?
  echo "[$(date --iso-8601=seconds)] EXIT rc=$rc pid=$$" >> "$V2_E0_BUILD_TRAP_LOG"
  rm -f "$V2_E0_BUILD_PID_FILE"
  exit "$rc"
}
on_term() {
  sig="$1"
  echo "[$(date --iso-8601=seconds)] SIGNAL $sig pid=$$" >> "$V2_E0_BUILD_TRAP_LOG"
}
trap on_exit EXIT
trap 'on_term TERM' TERM
trap 'on_term HUP' HUP
trap 'on_term INT' INT

"$V2_E0_PY" -u scripts/build_indexes.py

echo "[$(date --iso-8601=seconds)] build:complete pid=$$"
