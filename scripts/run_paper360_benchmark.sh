#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338

export TOKENIZERS_PARALLELISM=false

MANIFEST="data/hard_cases/omnitry_paper_360.json"
BENCH_INDEX="data/OmniTry_Bench/omni_vtryon_bench_small_v1.json"
MAX_ITEMS=360
STEPS="${STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-30}"
LORA_PATH="${LORA_PATH:-checkpoints/omnitry_v1_unified.safetensors}"
LOG_DIR="outputs/tryon_benchmark"
RUN_SUFFIX="${RUN_SUFFIX:-}"
IFS=',' read -r -a GPUS <<< "${PAPER360_GPUS:-2,3,7}"
MIN_FREE_MB="${MIN_FREE_MB:-28000}"
LAUNCH_DELAY_SECONDS="${LAUNCH_DELAY_SECONDS:-0}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "PAPER360_GPUS must contain at least one GPU id" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" outputs/enhance data/hard_cases

python scripts/build_hard_cases.py \
  --bench-root data/OmniTry_Bench \
  --index "$BENCH_INDEX" \
  --output "$MANIFEST" \
  --top-k "$MAX_ITEMS" \
  --require-local-images

is_shard_complete() {
  local summary_path="$1"
  local expected_count="$2"
  python - "$summary_path" "$expected_count" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
payload = json.loads(path.read_text())
items = payload.get("summary", {}).get("items", len(payload.get("items", [])))
raise SystemExit(0 if payload.get("complete") and items >= expected else 1)
PY
}

wait_for_gpu_memory() {
  local gpu="$1"
  local needed_mb="$2"
  while true; do
    local free_mb
    free_mb="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ -n "$free_mb" && "$free_mb" -ge "$needed_mb" ]]; then
      echo "[$(date -Is)] gpu=$gpu free=${free_mb}MiB >= ${needed_mb}MiB"
      break
    fi
    echo "[$(date -Is)] gpu=$gpu waiting: free=${free_mb:-unknown}MiB < ${needed_mb}MiB"
    sleep 60
  done
}

run_method() {
  local name="$1"
  local mode="$2"
  local candidates="$3"
  local run_name="${name}${RUN_SUFFIX}"
  local out_dir="$LOG_DIR/$run_name"
  local summaries=()
  local pids=()

  mkdir -p "$out_dir"
  echo "[$(date -Is)] Starting $run_name on paper 360 split: mode=$mode candidates=$candidates gpus=${GPUS[*]}"

  local shard_count="${#GPUS[@]}"
  for ((i=0; i<shard_count; i++)); do
    local gpu="${GPUS[$i]}"
    local start=$((i * MAX_ITEMS / shard_count))
    local end=$(((i + 1) * MAX_ITEMS / shard_count))
    local shard_expected=$((end - start))
    local summary="$LOG_DIR/${run_name}_shard_${i}_summary.json"
    local log="$LOG_DIR/${run_name}_shard_${i}.log"
    summaries+=("$summary")

    if is_shard_complete "$summary" "$shard_expected"; then
      echo "[$(date -Is)] $run_name shard $i already complete: $summary" | tee -a "$log"
      continue
    fi

    wait_for_gpu_memory "$gpu" "$MIN_FREE_MB" 2>&1 | tee -a "$log"
    (
      set -euo pipefail
      echo "[$(date -Is)] $run_name shard $i start: gpu=$gpu range=$start:$end" | tee -a "$log"
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_tryon_benchmark.py \
        --manifest "$MANIFEST" \
        --output-dir "$out_dir" \
        --summary-output "$summary" \
        --mode "$mode" \
        --max-items "$MAX_ITEMS" \
        --start-index "$start" \
        --end-index "$end" \
        --steps "$STEPS" \
        --guidance-scale "$GUIDANCE_SCALE" \
        --candidate-count "$candidates" \
        --lora-path "$LORA_PATH" \
        --skip-existing \
        2>&1 | tee -a "$log"
      echo "[$(date -Is)] $run_name shard $i complete" | tee -a "$log"
    ) &
    pids+=("$!")
    if [[ "$LAUNCH_DELAY_SECONDS" -gt 0 ]]; then
      sleep "$LAUNCH_DELAY_SECONDS"
    fi
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date -Is)] $run_name failed in at least one shard" >&2
    return "$status"
  fi

  local merge_args=()
  for summary in "${summaries[@]}"; do
    merge_args+=(--summary "$summary")
  done
  python scripts/merge_benchmark_summaries.py \
    --manifest "$MANIFEST" \
    --output-dir "$out_dir" \
    --expected-count "$MAX_ITEMS" \
    --summary-output "$LOG_DIR/${run_name}_summary.json" \
    "${merge_args[@]}"
}

{
  echo "[$(date -Is)] Paper-360 benchmark start"
  echo "[$(date -Is)] Protocol: OmniTry-Bench small v1, 360 samples from arXiv:2508.13632"
  run_method "paper360_pretrained" "Baseline" 1
  run_method "paper360_pretrained_geo" "Enhanced" 2
  echo "[$(date -Is)] Paper-360 benchmark complete"
} 2>&1 | tee "$LOG_DIR/paper360_benchmark.log"
