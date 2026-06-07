#!/usr/bin/env bash
set -euo pipefail
cd /workspace/skn25-fairdata-competition

nohup bash scripts/exp_v2_e2_bgem3_dense_sparse.sh > /tmp/launch_v2_e2_bgem3_ds.log 2>&1 &
echo $! > /tmp/launch_v2_e2_bgem3_ds.pid
nohup bash scripts/exp_v2_e4_e5_bm25.sh > /tmp/launch_v2_e4_e5_bm25.log 2>&1 &
echo $! > /tmp/launch_v2_e4_e5_bm25.pid

echo "started V2-E2-BGEM3-DS pid=$(cat /tmp/launch_v2_e2_bgem3_ds.pid)"
echo "started V2-E4-E5-BM25 pid=$(cat /tmp/launch_v2_e4_e5_bm25.pid)"
