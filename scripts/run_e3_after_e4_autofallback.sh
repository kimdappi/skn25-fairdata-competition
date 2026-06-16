#!/usr/bin/env bash
set -euo pipefail
cd /workspace/skn25-fairdata-competition

START_TS="${1:?missing start timestamp}"
E4_SUMMARY="/workspace/skn25-fairdata-competition/results/V2-E4-E5-BM25/summary.json"
E4_PRED="/workspace/skn25-fairdata-competition/results/V2-E4-E5-BM25/predictions.jsonl"
LOG="/tmp/run_e3_after_e4_autofallback.log"

{
  echo "[autoe3] wait start ts=${START_TS}"
  while true; do
    if [ -f "$E4_SUMMARY" ]; then
      MTIME=$(stat -c %Y "$E4_SUMMARY" 2>/dev/null || echo 0)
      if [ "$MTIME" -ge "$START_TS" ]; then
        echo "[autoe3] detected fresh E4 summary at ${MTIME}"
        break
      fi
    fi
    if [ -f "$E4_PRED" ]; then
      MTIME=$(stat -c %Y "$E4_PRED" 2>/dev/null || echo 0)
      if [ "$MTIME" -ge "$START_TS" ]; then
        echo "[autoe3] detected fresh E4 predictions at ${MTIME} (summary pending)"
        break
      fi
    fi
    sleep 30
  done

  echo "[autoe3] trying jina_v4 first"
  if bash /workspace/skn25-fairdata-competition/scripts/exp_v2_e3_dense_only_reranker.sh jina_v4; then
    echo "[autoe3] jina_v4 completed"
    exit 0
  fi

  echo "[autoe3] jina_v4 failed; fallback to gte_multilingual"
  bash /workspace/skn25-fairdata-competition/scripts/exp_v2_e3_dense_only_reranker.sh gte_multilingual
  echo "[autoe3] gte_multilingual completed"
} >>"$LOG" 2>&1
