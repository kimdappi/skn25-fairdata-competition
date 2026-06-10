#!/usr/bin/env bash
set -uo pipefail
cd /workspace/skn25-fairdata-competition
LOG="/workspace/skn25-fairdata-competition/results/requested_v2_experiments_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

echo "[RUNNER] MASTER_START $(date -Is) log=$LOG"
for s in \
  scripts/exp_v2_e4_gte_bm25.sh \
  scripts/exp_v2_e4_gte_bm25_top75.sh \
  scripts/exp_v2_e2_bgem3_ds_d30_s50.sh \
  scripts/exp_v2_e3_gte_dense_top75.sh
 do
  echo "[RUNNER] launching $s $(date -Is)"
  bash "$s"
  rc=$?
  echo "[RUNNER] finished $s rc=$rc $(date -Is)"
 done

echo "[HERMES_ALL_EXPERIMENTS_DONE] requested_v2_experiments log=$LOG"
