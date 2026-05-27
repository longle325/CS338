#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-2,3,7}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
BENCH_MAX_ITEMS="${BENCH_MAX_ITEMS:-32}"

LOG_DIR="outputs/enhance"
BENCH_DIR="outputs/tryon_benchmark"
mkdir -p "$LOG_DIR" "$BENCH_DIR"

TRAIN_LOG="$LOG_DIR/geo_lora_train_screen.log"
BENCH_LOG="$BENCH_DIR/enhanced_ft_benchmark.log"
CKPT="checkpoints/enhance/omnitry_geo_lora.safetensors"
METRICS="outputs/enhance/geo_lora_train_metrics.json"
BENCH_OUT_DIR="$BENCH_DIR/enhanced_ft"
BENCH_SUMMARY="$BENCH_DIR/enhanced_ft_summary.json"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

summarize_json() {
  local path="$1"
  local key="${2:-}"
  if [[ -s "$path" ]]; then
    python - "$path" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
payload = json.loads(path.read_text())
print(json.dumps(payload.get(key, payload) if key else payload, indent=2))
PY
  fi
}

{
  log "Starting Geometry/FLUX LoRA training"
  log "Train GPUs: $TRAIN_CUDA_VISIBLE_DEVICES"
  log "Train log: $TRAIN_LOG"
  log "Checkpoint: $CKPT"
} | tee -a "$TRAIN_LOG"

CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" accelerate launch --num_processes 3 scripts/train_geo_lora.py \
  --manifest data/hard_cases/omnitry_pseudo_paired_train.json \
  --output "$CKPT" \
  --metrics-output "$METRICS" \
  --resolution 512 \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  2>&1 | tee -a "$TRAIN_LOG"

log "Training command exited successfully" | tee -a "$TRAIN_LOG"
log "Training metrics:" | tee -a "$TRAIN_LOG"
summarize_json "$METRICS" | tee -a "$TRAIN_LOG"

if [[ ! -s "$CKPT" ]]; then
  log "Missing checkpoint after training: $CKPT" | tee -a "$TRAIN_LOG"
  exit 1
fi

{
  log "Starting benchmark"
  log "Benchmark GPU: $BENCH_CUDA_VISIBLE_DEVICES"
  log "Benchmark log: $BENCH_LOG"
} | tee -a "$BENCH_LOG"

CUDA_VISIBLE_DEVICES="$BENCH_CUDA_VISIBLE_DEVICES" python scripts/run_tryon_benchmark.py \
  --manifest data/hard_cases/omnitry_full_local_hard_cases.json \
  --output-dir "$BENCH_OUT_DIR" \
  --summary-output "$BENCH_SUMMARY" \
  --mode Enhanced \
  --max-items "$BENCH_MAX_ITEMS" \
  --lora-path "$CKPT" \
  2>&1 | tee -a "$BENCH_LOG"

log "Benchmark summary:" | tee -a "$BENCH_LOG"
summarize_json "$BENCH_SUMMARY" summary | tee -a "$BENCH_LOG"
log "Done" | tee -a "$BENCH_LOG"
