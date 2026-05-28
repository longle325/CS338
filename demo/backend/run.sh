#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338

export OMNITRY_CPU_OFFLOAD="${OMNITRY_CPU_OFFLOAD:-1}"
export OMNITRY_BACKEND_HOST="${OMNITRY_BACKEND_HOST:-0.0.0.0}"
export OMNITRY_BACKEND_PORT="${OMNITRY_BACKEND_PORT:-8010}"

exec python -m uvicorn demo.backend.app:app \
  --host "$OMNITRY_BACKEND_HOST" \
  --port "$OMNITRY_BACKEND_PORT"
