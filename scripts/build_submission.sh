#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-rag-submission:latest}"
OUTPUT_TAR="${2:-submission.tar}"

docker build -t "${IMAGE_NAME}" .
docker save "${IMAGE_NAME}" -o "${OUTPUT_TAR}"

echo "Saved ${IMAGE_NAME} to ${OUTPUT_TAR}"
