#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338

export TOKENIZERS_PARALLELISM=false

RESULT_DIR="${RESULT_DIR:-outputs/tryon_benchmark/paper360_pretrained_geo_geo2_20260527_054323}"
BENCHMARK_FILE="${BENCHMARK_FILE:-data/OmniTry_Bench/omni_vtryon_bench_small_v1.json}"
BENCH_ROOT="${BENCH_ROOT:-data/OmniTry_Bench}"
DEVICE="${DEVICE:-cuda}"
METRIC_DEVICE="${METRIC_DEVICE:-cuda:0}"
LOG_DIR="${LOG_DIR:-outputs/tryon_benchmark}"
RUN_NAME="${RUN_NAME:-paper360_pretrained_geo_official}"

mkdir -p "$LOG_DIR"

{
  echo "[$(date -Is)] official OmniTry metric pipeline start"
  echo "[$(date -Is)] result_dir=$RESULT_DIR"
  python scripts/run_official_omnitry_masks.py \
    --benchmark-file "$BENCHMARK_FILE" \
    --bench-root "$BENCH_ROOT" \
    --result-dir "$RESULT_DIR" \
    --summary-output "$RESULT_DIR/official_mask_summary.json" \
    --device "$DEVICE" \
    --skip-existing

  python scripts/run_official_omnitry_metrics.py \
    --benchmark-file "$BENCHMARK_FILE" \
    --bench-root "$BENCH_ROOT" \
    --result-dir "$RESULT_DIR" \
    --result-json "$RESULT_DIR/official_result.json" \
    --detail-json "$RESULT_DIR/official_result_detail.json" \
    --table-output "$RESULT_DIR/official_metric_table.md" \
    --device "$METRIC_DEVICE"
  echo "[$(date -Is)] official OmniTry metric pipeline complete"
} 2>&1 | tee "$LOG_DIR/${RUN_NAME}.log"
