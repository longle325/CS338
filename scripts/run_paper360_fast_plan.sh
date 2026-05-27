#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-4}"

MANIFEST="data/hard_cases/omnitry_paper_360.json"
BENCH_INDEX="data/OmniTry_Bench/omni_vtryon_bench_small_v1.json"
MAX_ITEMS=360
STEPS="${STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-30}"
LORA_PATH="${LORA_PATH:-checkpoints/omnitry_v1_unified.safetensors}"
LOG_DIR="outputs/tryon_benchmark"
RUN_SUFFIX="${RUN_SUFFIX:-_fast}"
PAPER360_METHODS="${PAPER360_METHODS:-both}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/paper360_fast_plan${RUN_SUFFIX}.log}"
LAUNCH_DELAY_SECONDS="${LAUNCH_DELAY_SECONDS:-0}"

# Format: shard_name:gpu:start:end:cpu_offload:min_free_mb
# Full-resident shards use offload=0 on mostly free A100s. Partially occupied GPUs
# keep offload=1 so we still use them without OOM.
PLAN="${PAPER360_PLAN:-g1:1:0:60:0:65000,g2:2:60:120:0:65000,g3:3:120:180:0:65000,g4:4:180:240:0:65000,g7:7:240:300:0:65000,g0:0:300:330:1:25000,g5:5:330:360:1:25000}"

mkdir -p "$LOG_DIR" outputs/enhance data/hard_cases

python scripts/build_hard_cases.py \
  --bench-root data/OmniTry_Bench \
  --index "$BENCH_INDEX" \
  --output "$MANIFEST" \
  --top-k "$MAX_ITEMS" \
  --require-local-images

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

run_method() {
  local name="$1"
  local mode="$2"
  local candidates="$3"
  local run_name="${name}${RUN_SUFFIX}"
  local out_dir="$LOG_DIR/$run_name"
  local pids=()
  local summaries=()

  mkdir -p "$out_dir"
  echo "[$(date -Is)] Starting $run_name with explicit plan"
  echo "[$(date -Is)] plan=$PLAN"

  IFS=',' read -r -a entries <<< "$PLAN"
  for entry in "${entries[@]}"; do
    IFS=':' read -r shard gpu start end offload min_free <<< "$entry"
    local expected=$((end - start))
    local summary="$LOG_DIR/${run_name}_${shard}_summary.json"
    local log="$LOG_DIR/${run_name}_${shard}.log"
    summaries+=("$summary")

    if is_shard_complete "$summary" "$expected"; then
      echo "[$(date -Is)] $run_name $shard already complete: $summary" | tee -a "$log"
      continue
    fi

    (
      set -euo pipefail
      echo "[$(date -Is)] $run_name $shard start gpu=$gpu range=$start:$end offload=$offload" | tee -a "$log"
      wait_for_gpu_memory "$gpu" "$min_free" 2>&1 | tee -a "$log"
      CUDA_VISIBLE_DEVICES="$gpu" OMNITRY_CPU_OFFLOAD="$offload" python scripts/run_tryon_benchmark.py \
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
      echo "[$(date -Is)] $run_name $shard complete expected=$expected" | tee -a "$log"
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
  echo "[$(date -Is)] Paper-360 fast-plan benchmark start"
  echo "[$(date -Is)] Protocol: OmniTry-Bench small v1, 360 samples from arXiv:2508.13632"
  echo "[$(date -Is)] methods=$PAPER360_METHODS"
  case "$PAPER360_METHODS" in
    both)
      run_method "paper360_pretrained" "Baseline" 1
      run_method "paper360_pretrained_geo" "Enhanced" 2
      ;;
    baseline)
      run_method "paper360_pretrained" "Baseline" 1
      ;;
    geo)
      run_method "paper360_pretrained_geo" "Enhanced" 2
      ;;
    *)
      echo "PAPER360_METHODS must be one of: both, baseline, geo" >&2
      exit 1
      ;;
  esac
  echo "[$(date -Is)] Paper-360 fast-plan benchmark complete"
} 2>&1 | tee -a "$LOG_PATH"
