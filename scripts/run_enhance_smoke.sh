#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${OMNITRY_ENV_NAME:-omnitry}"

cd "$ROOT_DIR"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

python scripts/build_hard_cases.py \
  --demo-fallback \
  --require-local-images \
  --top-k 8 \
  --output data/hard_cases/smoke_hard_cases.json

python scripts/train_affordance_planner.py \
  --manifest data/hard_cases/smoke_hard_cases.json \
  --output checkpoints/enhance/smoke_affordance_planner.pt \
  --metrics-output outputs/enhance/smoke_planner_train_metrics.json \
  --epochs 1 \
  --batch-size 2 \
  --max-items 4 \
  --device cpu

python scripts/eval_affordance_planner.py \
  --manifest data/hard_cases/smoke_hard_cases.json \
  --checkpoint checkpoints/enhance/smoke_affordance_planner.pt \
  --output outputs/enhance/smoke_planner_eval.json \
  --max-items 4 \
  --device cpu

echo "Enhance smoke pipeline completed."
