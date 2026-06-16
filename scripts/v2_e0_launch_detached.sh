#!/usr/bin/env bash
set -euo pipefail
cd /workspace/skn25-fairdata-competition

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 {build|server|eval}" >&2
  exit 2
fi

case "$PHASE" in
  build)
    SCRIPT="scripts/v2_e0_build.sh"
    LAUNCH_LOG="/tmp/v2_e0_build_launcher.log"
    PID_FILE="/tmp/v2_e0_build_launcher.pid"
    ;;
  server)
    SCRIPT="scripts/v2_e0_server.sh"
    LAUNCH_LOG="/tmp/v2_e0_server_launcher.log"
    PID_FILE="/tmp/v2_e0_server_launcher.pid"
    ;;
  eval)
    SCRIPT="scripts/v2_e0_eval.sh"
    LAUNCH_LOG="/tmp/v2_e0_eval_launcher.log"
    PID_FILE="/tmp/v2_e0_eval_launcher.pid"
    ;;
  *)
    echo "invalid phase: $PHASE" >&2
    exit 2
    ;;
esac

rm -f "$PID_FILE"
nohup setsid bash "$SCRIPT" > "$LAUNCH_LOG" 2>&1 < /dev/null &
LAUNCH_PID=$!
echo "$LAUNCH_PID" > "$PID_FILE"
echo "phase=$PHASE launcher_pid=$LAUNCH_PID launcher_log=$LAUNCH_LOG script=$SCRIPT"
