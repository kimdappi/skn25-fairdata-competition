#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/_run_route_experiment.sh" lcel V2-E4-E5-BM25-ROUTER-LCEL 8014 0,1
