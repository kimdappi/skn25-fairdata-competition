#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER=(uv run --python 3.11 python)
  UVICORN_RUNNER=(uv run --python 3.11 python -m uvicorn)
else
  PYTHON_RUNNER=(python3)
  UVICORN_RUNNER=(python3 -m uvicorn)
fi

cd "${PROJECT_ROOT}"

run_python() {
  "${PYTHON_RUNNER[@]}" "$@"
}

run_python_u() {
  "${PYTHON_RUNNER[@]}" -u "$@"
}

run_uvicorn() {
  "${UVICORN_RUNNER[@]}" "$@"
}
