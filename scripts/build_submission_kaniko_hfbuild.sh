#!/usr/bin/env bash
set -euo pipefail

cd /workspace/skn25-fairdata-competition

KANIKO="${KANIKO:-/workspace/tools/executor}"
TAG="${TAG:-fairdata-rag-submission:hfbuild}"
OUT="${OUT:-/workspace/skn25-fairdata-competition/fairdata-rag-submission-hfbuild.tar}"

if [[ ! -x "$KANIKO" ]]; then
  echo "Kaniko executor not found: $KANIKO" >&2
  exit 1
fi

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
rm -f "$OUT" "$OUT.sha256" "$OUT.zst" "$OUT.zst.sha256"

"$KANIKO" \
  --context dir:///workspace/skn25-fairdata-competition \
  --dockerfile /workspace/skn25-fairdata-competition/Dockerfile.hfbuild \
  --destination "$TAG" \
  --no-push \
  --tarPath "$OUT" \
  --single-snapshot \
  --cleanup \
  --verbosity=info

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
