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

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating conda env: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  echo "Creating conda env: $ENV_NAME"
  conda env create -n "$ENV_NAME" -f environment.yml
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ "${OMNITRY_SKIP_DOWNLOAD:-0}" != "1" ]]; then
  python scripts/download_checkpoints.py
else
  echo "Skipping checkpoint download because OMNITRY_SKIP_DOWNLOAD=1"
fi

python scripts/check_runtime.py || true

echo
echo "Setup finished. Start the demo with:"
echo "  bash scripts/run_gradio.sh"
