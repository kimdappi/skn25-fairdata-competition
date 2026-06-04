#!/usr/bin/env bash
set -euo pipefail

# 패치노트 기준 실험 모델을 기능군(embedding / reranker / llm)별로 내려받습니다.
# 기존 download_bgem3_model.sh 스타일을 유지하면서 기능군/모델별 토글을 추가했습니다.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${FAIRDATA_MODELS_DIR:-${PROJECT_DIR}/models}"

INSTALL_HF_HUB="${INSTALL_HF_HUB:-1}"
DOWNLOAD_EMBEDDINGS="${DOWNLOAD_EMBEDDINGS:-1}"
DOWNLOAD_RERANKERS="${DOWNLOAD_RERANKERS:-1}"
DOWNLOAD_LLMS="${DOWNLOAD_LLMS:-1}"

download_repo() {
  local repo_id="$1"
  local model_dir="$2"
  local label="$3"

  mkdir -p "${model_dir}"
  echo "[download] ${label}: ${repo_id} -> ${model_dir}"
  huggingface-cli download "${repo_id}" --local-dir "${model_dir}"
  echo "[download] ${label} 완료"
}

download_if_enabled() {
  local enabled="$1"
  local repo_id="$2"
  local model_dir="$3"
  local label="$4"

  if [[ "${enabled}" == "1" ]]; then
    download_repo "${repo_id}" "${model_dir}" "${label}"
  else
    echo "[skip] ${label}"
  fi
}

print_usage() {
  cat <<'EOF'
사용 예시
  bash scripts/download_model_matrix.sh
  DOWNLOAD_EMBEDDINGS=1 DOWNLOAD_RERANKERS=0 DOWNLOAD_LLMS=0 bash scripts/download_model_matrix.sh
  DOWNLOAD_JINA_EMBEDDINGS_V4=1 DOWNLOAD_QWEN3_8B=1 bash scripts/download_model_matrix.sh

기능군 토글
  DOWNLOAD_EMBEDDINGS=1|0
  DOWNLOAD_RERANKERS=1|0
  DOWNLOAD_LLMS=1|0

모델별 토글
  Embedding
    DOWNLOAD_BGE_M3
    DOWNLOAD_MULTILINGUAL_E5_LARGE
    DOWNLOAD_KO_SBERT_NLI
    DOWNLOAD_JINA_EMBEDDINGS_V4
    DOWNLOAD_GTE_MULTILINGUAL_BASE
    DOWNLOAD_KURE_V1
    DOWNLOAD_SNOWFLAKE_ARCTIC_EMBED_L_V20_KO

  Reranker
    DOWNLOAD_BGE_RERANKER_V2_M3
    DOWNLOAD_BGE_RERANKER_V2_GEMMA
    DOWNLOAD_BGE_RERANKER_V25_GEMMA2_LIGHTWEIGHT
    DOWNLOAD_JINA_RERANKER_V3
    DOWNLOAD_KO_RERANKER

  LLM
    DOWNLOAD_QWEN25_7B_INSTRUCT
    DOWNLOAD_QWEN3_8B
    DOWNLOAD_QWEN3_4B
    DOWNLOAD_EXAONE_35_78B_INSTRUCT
    DOWNLOAD_LLAMA3_OPEN_KO_8B
    DOWNLOAD_LLAMA_VARCO_8B_INSTRUCT

주의
  - 현재 코드의 기본 config 경로/백엔드명과 패치노트 모델 목록은 일부 불일치할 수 있습니다.
  - 이 스크립트는 실험용 모델 아카이브를 기능군별로 받는 목적입니다.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

if [[ "${INSTALL_HF_HUB}" == "1" ]]; then
  python -m pip install --upgrade "huggingface_hub>=0.24,<1.0"
fi

echo "[group] embeddings=${DOWNLOAD_EMBEDDINGS} rerankers=${DOWNLOAD_RERANKERS} llms=${DOWNLOAD_LLMS}"

# Embedding models
EMBEDDING_DIR_BGE_M3="${EMBEDDING_DIR_BGE_M3:-${MODELS_DIR}/bge-m3}"
EMBEDDING_DIR_MULTILINGUAL_E5_LARGE="${EMBEDDING_DIR_MULTILINGUAL_E5_LARGE:-${MODELS_DIR}/multilingual-e5-large}"
EMBEDDING_DIR_KO_SBERT_NLI="${EMBEDDING_DIR_KO_SBERT_NLI:-${MODELS_DIR}/ko-sbert-nli}"
EMBEDDING_DIR_JINA_EMBEDDINGS_V4="${EMBEDDING_DIR_JINA_EMBEDDINGS_V4:-${MODELS_DIR}/jina-embeddings-v4}"
EMBEDDING_DIR_GTE_MULTILINGUAL_BASE="${EMBEDDING_DIR_GTE_MULTILINGUAL_BASE:-${MODELS_DIR}/gte-multilingual-base}"
EMBEDDING_DIR_KURE_V1="${EMBEDDING_DIR_KURE_V1:-${MODELS_DIR}/KURE-v1}"
EMBEDDING_DIR_SNOWFLAKE_ARCTIC_EMBED_L_V20_KO="${EMBEDDING_DIR_SNOWFLAKE_ARCTIC_EMBED_L_V20_KO:-${MODELS_DIR}/snowflake-arctic-embed-l-v2.0-ko}"

if [[ "${DOWNLOAD_EMBEDDINGS}" == "1" ]]; then
  download_if_enabled "${DOWNLOAD_BGE_M3:-1}" \
    "BAAI/bge-m3" "${EMBEDDING_DIR_BGE_M3}" "Embedding / BGE-M3"
  download_if_enabled "${DOWNLOAD_MULTILINGUAL_E5_LARGE:-1}" \
    "intfloat/multilingual-e5-large" "${EMBEDDING_DIR_MULTILINGUAL_E5_LARGE}" "Embedding / multilingual-e5-large"
  download_if_enabled "${DOWNLOAD_KO_SBERT_NLI:-1}" \
    "jhgan/ko-sbert-nli" "${EMBEDDING_DIR_KO_SBERT_NLI}" "Embedding / ko-sbert-nli"
  download_if_enabled "${DOWNLOAD_JINA_EMBEDDINGS_V4:-1}" \
    "jinaai/jina-embeddings-v4" "${EMBEDDING_DIR_JINA_EMBEDDINGS_V4}" "Embedding / jina-embeddings-v4"
  download_if_enabled "${DOWNLOAD_GTE_MULTILINGUAL_BASE:-1}" \
    "Alibaba-NLP/gte-multilingual-base" "${EMBEDDING_DIR_GTE_MULTILINGUAL_BASE}" "Embedding / gte-multilingual-base"
  download_if_enabled "${DOWNLOAD_KURE_V1:-1}" \
    "nlpai-lab/KURE-v1" "${EMBEDDING_DIR_KURE_V1}" "Embedding / KURE-v1"
  download_if_enabled "${DOWNLOAD_SNOWFLAKE_ARCTIC_EMBED_L_V20_KO:-1}" \
    "dragonkue/snowflake-arctic-embed-l-v2.0-ko" "${EMBEDDING_DIR_SNOWFLAKE_ARCTIC_EMBED_L_V20_KO}" "Embedding / snowflake-arctic-embed-l-v2.0-ko"
  echo "[note] BGE-M3는 sparse / multi-vector 헤드용 sparse_linear.pt, colbert_linear.pt 포함 여부를 확인하세요."
fi

# Reranker models
RERANKER_DIR_BGE_RERANKER_V2_M3="${RERANKER_DIR_BGE_RERANKER_V2_M3:-${MODELS_DIR}/bge-reranker-v2-m3}"
RERANKER_DIR_BGE_RERANKER_V2_GEMMA="${RERANKER_DIR_BGE_RERANKER_V2_GEMMA:-${MODELS_DIR}/bge-reranker-v2-gemma}"
RERANKER_DIR_BGE_RERANKER_V25_GEMMA2_LIGHTWEIGHT="${RERANKER_DIR_BGE_RERANKER_V25_GEMMA2_LIGHTWEIGHT:-${MODELS_DIR}/bge-reranker-v2.5-gemma2-lightweight}"
RERANKER_DIR_JINA_RERANKER_V3="${RERANKER_DIR_JINA_RERANKER_V3:-${MODELS_DIR}/jina-reranker-v3}"
RERANKER_DIR_KO_RERANKER="${RERANKER_DIR_KO_RERANKER:-${MODELS_DIR}/ko-reranker}"

if [[ "${DOWNLOAD_RERANKERS}" == "1" ]]; then
  download_if_enabled "${DOWNLOAD_BGE_RERANKER_V2_M3:-1}" \
    "BAAI/bge-reranker-v2-m3" "${RERANKER_DIR_BGE_RERANKER_V2_M3}" "Reranker / bge-reranker-v2-m3"
  download_if_enabled "${DOWNLOAD_BGE_RERANKER_V2_GEMMA:-1}" \
    "BAAI/bge-reranker-v2-gemma" "${RERANKER_DIR_BGE_RERANKER_V2_GEMMA}" "Reranker / bge-reranker-v2-gemma"
  download_if_enabled "${DOWNLOAD_BGE_RERANKER_V25_GEMMA2_LIGHTWEIGHT:-1}" \
    "BAAI/bge-reranker-v2.5-gemma2-lightweight" "${RERANKER_DIR_BGE_RERANKER_V25_GEMMA2_LIGHTWEIGHT}" "Reranker / bge-reranker-v2.5-gemma2-lightweight"
  download_if_enabled "${DOWNLOAD_JINA_RERANKER_V3:-1}" \
    "jinaai/jina-reranker-v3" "${RERANKER_DIR_JINA_RERANKER_V3}" "Reranker / jina-reranker-v3"
  download_if_enabled "${DOWNLOAD_KO_RERANKER:-1}" \
    "Dongjin-kr/ko-reranker" "${RERANKER_DIR_KO_RERANKER}" "Reranker / ko-reranker"
fi

# LLM models
LLM_DIR_QWEN25_7B_INSTRUCT="${LLM_DIR_QWEN25_7B_INSTRUCT:-${MODELS_DIR}/qwen2.5-7b-instruct}"
LLM_DIR_QWEN3_8B="${LLM_DIR_QWEN3_8B:-${MODELS_DIR}/qwen3-8b}"
LLM_DIR_QWEN3_4B="${LLM_DIR_QWEN3_4B:-${MODELS_DIR}/qwen3-4b}"
LLM_DIR_EXAONE_35_78B_INSTRUCT="${LLM_DIR_EXAONE_35_78B_INSTRUCT:-${MODELS_DIR}/EXAONE-3.5-7.8B-Instruct}"
LLM_DIR_LLAMA3_OPEN_KO_8B="${LLM_DIR_LLAMA3_OPEN_KO_8B:-${MODELS_DIR}/Llama-3-Open-Ko-8B}"
LLM_DIR_LLAMA_VARCO_8B_INSTRUCT="${LLM_DIR_LLAMA_VARCO_8B_INSTRUCT:-${MODELS_DIR}/Llama-VARCO-8B-Instruct}"

if [[ "${DOWNLOAD_LLMS}" == "1" ]]; then
  download_if_enabled "${DOWNLOAD_QWEN25_7B_INSTRUCT:-1}" \
    "Qwen/Qwen2.5-7B-Instruct" "${LLM_DIR_QWEN25_7B_INSTRUCT}" "LLM / Qwen2.5-7B-Instruct"
  download_if_enabled "${DOWNLOAD_QWEN3_8B:-1}" \
    "Qwen/Qwen3-8B" "${LLM_DIR_QWEN3_8B}" "LLM / Qwen3-8B"
  download_if_enabled "${DOWNLOAD_QWEN3_4B:-1}" \
    "Qwen/Qwen3-4B" "${LLM_DIR_QWEN3_4B}" "LLM / Qwen3-4B"
  download_if_enabled "${DOWNLOAD_EXAONE_35_78B_INSTRUCT:-1}" \
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" "${LLM_DIR_EXAONE_35_78B_INSTRUCT}" "LLM / EXAONE-3.5-7.8B-Instruct"
  download_if_enabled "${DOWNLOAD_LLAMA3_OPEN_KO_8B:-1}" \
    "beomi/Llama-3-Open-Ko-8B" "${LLM_DIR_LLAMA3_OPEN_KO_8B}" "LLM / Llama-3-Open-Ko-8B"
  download_if_enabled "${DOWNLOAD_LLAMA_VARCO_8B_INSTRUCT:-1}" \
    "NCSOFT/Llama-VARCO-8B-Instruct" "${LLM_DIR_LLAMA_VARCO_8B_INSTRUCT}" "LLM / Llama-VARCO-8B-Instruct"
fi

echo "[done] 기능군별 모델 다운로드 스크립트 실행 완료"
echo "[root] ${MODELS_DIR}"
