#!/usr/bin/env bash
set -euo pipefail

# Hugging Face에서 검색/생성 모델을 내려받고 제출 코드가 기대하는 위치를 만듭니다.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BGEM3_MODEL_DIR="${FAIRDATA_BGEM3_MODEL_DIR:-${PROJECT_DIR}/models/bge-m3}"
BGE_RERANKER_MODEL_DIR="${FAIRDATA_BGE_RERANKER_MODEL_DIR:-${PROJECT_DIR}/models/bge-reranker-v2-m3}"
QWEN_MODEL_DIR="${FAIRDATA_QWEN_MODEL_DIR:-${PROJECT_DIR}/models/qwen2-7b-instruct}"

BGEM3_REPO_ID="${HF_REPO_ID:-BAAI/bge-m3}"
BGE_RERANKER_REPO_ID="${BGE_RERANKER_REPO_ID:-BAAI/bge-reranker-v2-m3}"
QWEN_REPO_ID="${QWEN_REPO_ID:-Qwen/Qwen2-7B-Instruct}"

DOWNLOAD_BGEM3="${DOWNLOAD_BGEM3:-1}"
DOWNLOAD_RERANKER="${DOWNLOAD_RERANKER:-1}"
DOWNLOAD_QWEN="${DOWNLOAD_QWEN:-1}"

download_repo() {
  local repo_id="$1"
  local model_dir="$2"
  local label="$3"

  mkdir -p "${model_dir}"
  echo "[download] ${label}: ${repo_id} -> ${model_dir}"
  python -m huggingface_hub download \
    "${repo_id}" \
    --local-dir "${model_dir}" \
    --local-dir-use-symlinks False
  echo "[download] ${label} 완료"
}

python -m pip install --upgrade "huggingface_hub>=0.24,<1.0"

if [[ "${DOWNLOAD_BGEM3}" == "1" ]]; then
  download_repo "${BGEM3_REPO_ID}" "${BGEM3_MODEL_DIR}" "BGE-M3"
  echo "다음 파일이 추가로 필요합니다."
  echo "  - ${BGEM3_MODEL_DIR}/sparse_linear.pt"
  echo "  - ${BGEM3_MODEL_DIR}/colbert_linear.pt"
  echo "위 두 파일은 현재 검색 코드가 사용하는 sparse / multi-vector 헤드 가중치입니다."
fi

if [[ "${DOWNLOAD_RERANKER}" == "1" ]]; then
  download_repo "${BGE_RERANKER_REPO_ID}" "${BGE_RERANKER_MODEL_DIR}" "BGE reranker v2 m3"
fi

if [[ "${DOWNLOAD_QWEN}" == "1" ]]; then
  download_repo "${QWEN_REPO_ID}" "${QWEN_MODEL_DIR}" "Qwen 7B"
fi

echo "모델 다운로드 완료"
echo "  - BGE-M3: ${BGEM3_MODEL_DIR}"
echo "  - Reranker: ${BGE_RERANKER_MODEL_DIR}"
echo "  - Qwen 7B: ${QWEN_MODEL_DIR}"
