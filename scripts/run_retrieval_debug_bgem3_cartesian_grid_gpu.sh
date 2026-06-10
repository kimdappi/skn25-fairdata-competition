#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/skn25-fairdata-competition"
DATASET_PATH="$PROJECT_ROOT/data/test/eval_dataset_260505.json"
RUN_ROOT="${1:-/workspace/runs/skn25-retrieval-grid-bgem3-gpu-20260608-013213}"
COMMANDS_DIR="$RUN_ROOT/commands"
LOGS_DIR="$RUN_ROOT/logs"
RESULTS_DIR="$RUN_ROOT/results"
META_DIR="$RUN_ROOT/meta"
MODEL_DIR="$PROJECT_ROOT/models/embedding/bge-m3"
RERANK_DIR="$PROJECT_ROOT/models/reranker/bge-reranker-v2-m3"

mkdir -p "$COMMANDS_DIR" "$LOGS_DIR" "$RESULTS_DIR" "$META_DIR"

for required in \
  "$MODEL_DIR/config.json" \
  "$MODEL_DIR/sparse_linear.pt" \
  "$MODEL_DIR/colbert_linear.pt" \
  "$RERANK_DIR/config.json" \
  "$PROJECT_ROOT/scripts/evaluate_retrieval_debug.py"
  do
  if [[ ! -f "$required" ]]; then
    echo "[error] missing required file: $required" >&2
    exit 1
  fi
done

cat > "$COMMANDS_DIR/command-template.txt" <<'EOF'
CUDA_VISIBLE_DEVICES=0 \
FAIRDATA_TORCH_DTYPE=float16 \
FAIRDATA_EMBED_BATCH_SIZE=4 \
FAIRDATA_EMBED_MAX_LENGTH=1024 \
FAIRDATA_DENSE_BACKEND=bgem3 \
FAIRDATA_DENSE_MODEL_DIR="$PROJECT_ROOT/models/embedding/bge-m3" \
FAIRDATA_SPARSE_BACKEND=bgem3 \
FAIRDATA_SPARSE_MODEL_DIR="$PROJECT_ROOT/models/embedding/bge-m3" \
FAIRDATA_ENABLE_MULTIVECTOR=false \
FAIRDATA_RERANK_BACKEND=bge_reranker \
FAIRDATA_RERANK_MODEL_DIR="$PROJECT_ROOT/models/reranker/bge-reranker-v2-m3" \
PYTHONPATH="$PROJECT_ROOT" \
PYTHONUNBUFFERED=1 \
python3 -u scripts/evaluate_retrieval_debug.py \
  --dataset-path "$DATASET_PATH" \
  --output-dir <combo-output-dir> \
  --recall-ks 5,10,20,30,40,50 \
  --mrr-ks 5,10,20,30,40,50 \
  --dense-topks <dense> \
  --sparse-topks <sparse> \
  --final-topks <final> \
  --compare-reranker \
  --save-rows \
  --tag <combo-tag>
EOF

cd "$PROJECT_ROOT"
python3 -V | tee "$META_DIR/python-version.txt"
pwd | tee "$META_DIR/project-root.txt"
printf '%s\n' "$DATASET_PATH" > "$META_DIR/dataset-path.txt"
printf '%s\n' "$RUN_ROOT" > "$META_DIR/run-root.txt"
printf '%s\n' 'CUDA_VISIBLE_DEVICES=0' > "$META_DIR/runtime.txt"
printf '%s\n' 'FAIRDATA_TORCH_DTYPE=float16' >> "$META_DIR/runtime.txt"
printf '%s\n' 'FAIRDATA_EMBED_BATCH_SIZE=4' >> "$META_DIR/runtime.txt"
printf '%s\n' 'FAIRDATA_DENSE_BACKEND=bgem3' > "$META_DIR/backend-profile.txt"
printf '%s\n' 'FAIRDATA_SPARSE_BACKEND=bgem3' >> "$META_DIR/backend-profile.txt"
printf '%s\n' 'FAIRDATA_ENABLE_MULTIVECTOR=false' >> "$META_DIR/backend-profile.txt"
printf '%s\n' 'FAIRDATA_RERANK_BACKEND=bge_reranker' >> "$META_DIR/backend-profile.txt"
printf '%s\n' "$MODEL_DIR" > "$META_DIR/bgem3-model-dir.txt"
printf '%s\n' "$RERANK_DIR" > "$META_DIR/reranker-model-dir.txt"

BASELINE_DENSE=30
BASELINE_SPARSE=30
BASELINE_FINAL=50

COMBOS=(
  "30 30 50"
  "50 30 50"
  "70 30 50"
  "100 30 50"
  "30 50 50"
  "30 70 50"
  "30 100 50"
  "30 30 75"
  "30 30 100"
)

planned_count=${#COMBOS[@]}
combo_count=0
skip_count=0
printf '%s\n' "baseline=dense${BASELINE_DENSE}_sparse${BASELINE_SPARSE}_final${BASELINE_FINAL}" > "$META_DIR/plan.txt"
printf '%s\n' 'strategy=baseline + dense sweep(3) + sparse sweep(3) + final sweep(2)' >> "$META_DIR/plan.txt"
for combo in "${COMBOS[@]}"; do
  read -r dense sparse final <<< "$combo"
  combo_count=$((combo_count + 1))
  combo_name="dense${dense}_sparse${sparse}_final${final}"
  combo_tag="gpu_bgem3_${combo_name}"
  combo_output_dir="$RESULTS_DIR/$combo_name"
  combo_log="$LOGS_DIR/${combo_name}.log"
  combo_summary="$combo_output_dir/summary.json"
  mkdir -p "$combo_output_dir"
  printf '%s\n' "$combo_output_dir" > "$META_DIR/last-output-dir.txt"
  if [[ -f "$combo_summary" ]]; then
    skip_count=$((skip_count + 1))
    printf '[skip] %s (%d/%d) summary exists\n' "$combo_name" "$combo_count" "$planned_count" | tee -a "$LOGS_DIR/grid-progress.log"
    continue
  fi
  printf '[run] %s (%d/%d)\n' "$combo_name" "$combo_count" "$planned_count" | tee -a "$LOGS_DIR/grid-progress.log"
  CUDA_VISIBLE_DEVICES=0 \
  FAIRDATA_TORCH_DTYPE=float16 \
  FAIRDATA_EMBED_BATCH_SIZE=4 \
  FAIRDATA_EMBED_MAX_LENGTH=1024 \
  FAIRDATA_DENSE_BACKEND=bgem3 \
  FAIRDATA_DENSE_MODEL_DIR="$MODEL_DIR" \
  FAIRDATA_SPARSE_BACKEND=bgem3 \
  FAIRDATA_SPARSE_MODEL_DIR="$MODEL_DIR" \
  FAIRDATA_ENABLE_MULTIVECTOR=false \
  FAIRDATA_RERANK_BACKEND=bge_reranker \
  FAIRDATA_RERANK_MODEL_DIR="$RERANK_DIR" \
  PYTHONPATH="$PROJECT_ROOT" \
  PYTHONUNBUFFERED=1 \
  python3 -u scripts/evaluate_retrieval_debug.py \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$combo_output_dir" \
    --recall-ks 5,10,20,30,40,50 \
    --mrr-ks 5,10,20,30,40,50 \
    --dense-topks "$dense" \
    --sparse-topks "$sparse" \
    --final-topks "$final" \
    --compare-reranker \
    --save-rows \
    --tag "$combo_tag" \
    | tee "$combo_log"
done

echo "[done] total_planned=$planned_count total_seen=$combo_count skipped=$skip_count" | tee -a "$LOGS_DIR/grid-progress.log"
