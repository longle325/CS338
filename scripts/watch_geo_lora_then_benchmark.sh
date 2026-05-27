#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:-}"
if [[ -z "$TRAIN_PID" ]]; then
  echo "Usage: $0 <train-pid>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338

WATCH_INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-300}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-2}"
BENCH_MAX_ITEMS="${BENCH_MAX_ITEMS:-32}"

LOG_DIR="outputs/enhance"
BENCH_DIR="outputs/tryon_benchmark"
mkdir -p "$LOG_DIR" "$BENCH_DIR"

WATCH_LOG="$LOG_DIR/geo_lora_followup.log"
BENCH_LOG="$BENCH_DIR/enhanced_ft_benchmark.log"
CKPT="checkpoints/enhance/omnitry_geo_lora.safetensors"
METRICS="outputs/enhance/geo_lora_train_metrics.json"
BENCH_OUT_DIR="$BENCH_DIR/enhanced_ft"
BENCH_SUMMARY="$BENCH_DIR/enhanced_ft_summary.json"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$WATCH_LOG"
}

summarize_training() {
  if [[ ! -s "$METRICS" ]]; then
    log "Training metrics file is missing: $METRICS"
    return 0
  fi

  python - <<'PY' | tee -a "$WATCH_LOG"
import json
from pathlib import Path

path = Path("outputs/enhance/geo_lora_train_metrics.json")
payload = json.loads(path.read_text())
history = payload.get("history", [])
last = history[-1] if history else {}
print("Training metrics:")
print(f"- checkpoint: {payload.get('checkpoint')}")
print(f"- manifest: {payload.get('manifest')}")
print(f"- items: {payload.get('items')}")
print(f"- resolution: {payload.get('resolution')}")
print(f"- steps: {payload.get('steps')}")
print(f"- cuda_visible_devices: {payload.get('cuda_visible_devices')}")
print(f"- last_history: {last}")
PY
}

summarize_benchmark() {
  if [[ ! -s "$BENCH_SUMMARY" ]]; then
    log "Benchmark summary is missing: $BENCH_SUMMARY"
    return 0
  fi

  python - <<'PY' | tee -a "$WATCH_LOG"
import json
from pathlib import Path

path = Path("outputs/tryon_benchmark/enhanced_ft_summary.json")
payload = json.loads(path.read_text())
print("Benchmark summary:")
print(json.dumps(payload.get("summary", {}), indent=2))
PY
}

log "Watching training PID $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if command -v nvidia-smi >/dev/null 2>&1; then
    {
      printf '[%s] GPU status for cards 2,3,7\n' "$(date -Is)"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits \
        | awk -F', ' '$1 == 2 || $1 == 3 || $1 == 7 {print "- gpu " $1 ": util=" $2 "% mem=" $3 "/" $4 " MiB"}'
    } >> "$WATCH_LOG" || true
  fi
  sleep "$WATCH_INTERVAL_SECONDS"
done

sleep 10
log "Training PID $TRAIN_PID exited; checking outputs."
summarize_training

if [[ ! -s "$CKPT" ]]; then
  log "Missing fine-tuned LoRA checkpoint: $CKPT"
  log "Benchmark skipped because training did not produce the checkpoint."
  pgrep -af "train_geo_lora|accelerate launch" | tee -a "$WATCH_LOG" || true
  exit 1
fi

log "Running enhanced benchmark with LoRA: $CKPT"
if ! CUDA_VISIBLE_DEVICES="$BENCH_CUDA_VISIBLE_DEVICES" python scripts/run_tryon_benchmark.py \
  --manifest data/hard_cases/omnitry_full_local_hard_cases.json \
  --output-dir "$BENCH_OUT_DIR" \
  --summary-output "$BENCH_SUMMARY" \
  --mode Enhanced \
  --max-items "$BENCH_MAX_ITEMS" \
  --lora-path "$CKPT" \
  2>&1 | tee -a "$BENCH_LOG"; then
  log "Benchmark failed. See $BENCH_LOG"
  exit 1
fi

summarize_benchmark
log "Follow-up complete."
