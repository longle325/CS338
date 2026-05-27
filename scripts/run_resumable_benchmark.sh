#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-2}"
MANIFEST="${MANIFEST:-data/hard_cases/omnitry_full_local_hard_cases.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tryon_benchmark/original_enhanced_c2}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-outputs/tryon_benchmark/original_enhanced_c2_summary.json}"
LOG_PATH="${LOG_PATH:-outputs/tryon_benchmark/original_enhanced_c2_benchmark.log}"
MODE="${MODE:-Enhanced}"
MAX_ITEMS="${MAX_ITEMS:-32}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-2}"
LORA_PATH="${LORA_PATH:-checkpoints/omnitry_v1_unified.safetensors}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"

is_complete() {
  python - "$SUMMARY_OUTPUT" "$MAX_ITEMS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit(1)
payload = json.loads(path.read_text())
items = payload.get("summary", {}).get("items", len(payload.get("items", [])))
raise SystemExit(0 if payload.get("complete") and items >= expected else 1)
PY
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if is_complete; then
    echo "[$(date -Is)] Benchmark already complete: $SUMMARY_OUTPUT" | tee -a "$LOG_PATH"
    exit 0
  fi

  echo "[$(date -Is)] Benchmark attempt $attempt/$MAX_ATTEMPTS" | tee -a "$LOG_PATH"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python scripts/run_tryon_benchmark.py \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --summary-output "$SUMMARY_OUTPUT" \
    --mode "$MODE" \
    --max-items "$MAX_ITEMS" \
    --candidate-count "$CANDIDATE_COUNT" \
    --lora-path "$LORA_PATH" \
    --skip-existing \
    2>&1 | tee -a "$LOG_PATH"

  status=${PIPESTATUS[0]}
  echo "[$(date -Is)] Benchmark attempt $attempt exited with status $status" | tee -a "$LOG_PATH"
  sleep 10
done

echo "[$(date -Is)] Benchmark did not complete after $MAX_ATTEMPTS attempts" | tee -a "$LOG_PATH"
exit 1
