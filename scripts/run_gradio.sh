#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${OMNITRY_ENV_NAME:-omnitry}"

cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or use a Vast.ai image that includes conda." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

python scripts/check_runtime.py
python gradio_demo.py
