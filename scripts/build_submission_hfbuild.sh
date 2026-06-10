#!/usr/bin/env bash
set -euo pipefail

cd /workspace/skn25-fairdata-competition

TAG="${TAG:-fairdata-rag-submission:hfbuild}"
OUT="${OUT:-/workspace/skn25-fairdata-competition/fairdata-rag-submission-hfbuild.tar}"

BACKUP=".dockerignore.backup.$(date +%s).$$"
if [[ -f .dockerignore ]]; then
  cp .dockerignore "$BACKUP"
else
  BACKUP=""
fi
restore_ignore() {
  if [[ -n "$BACKUP" && -f "$BACKUP" ]]; then
    mv "$BACKUP" .dockerignore
  fi
}
trap restore_ignore EXIT

cp .dockerignore.hfbuild .dockerignore

if docker buildx version >/dev/null 2>&1; then
  export DOCKER_BUILDKIT=1
  BUILD_PROGRESS=(--progress=plain)
else
  export DOCKER_BUILDKIT=0
  BUILD_PROGRESS=()
fi
docker build "${BUILD_PROGRESS[@]}" -f Dockerfile.hfbuild -t "$TAG" .
docker save "$TAG" -o "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
du -h "$OUT"
echo "saved: $OUT"
echo "sha256: $OUT.sha256"

if command -v zstd >/dev/null 2>&1; then
  zstd -T0 -19 -f "$OUT" -o "$OUT.zst"
  sha256sum "$OUT.zst" > "$OUT.zst.sha256"
  du -h "$OUT.zst"
  echo "compressed: $OUT.zst"
fi
