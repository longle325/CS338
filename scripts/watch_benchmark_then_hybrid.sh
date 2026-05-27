#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338

WATCH_SUMMARY="${WATCH_SUMMARY:-outputs/tryon_benchmark/original_enhanced_c2_summary.json}"
EXPECTED_ITEMS="${EXPECTED_ITEMS:-32}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LOG_PATH="${LOG_PATH:-outputs/tryon_benchmark/best_total_followup.log}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tryon_benchmark/best_total_3way}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-outputs/tryon_benchmark/best_total_3way_summary.json}"

is_complete() {
  python - "$WATCH_SUMMARY" "$EXPECTED_ITEMS" <<'PY'
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

while ! is_complete; do
  python - "$WATCH_SUMMARY" <<'PY' | tee -a "$LOG_PATH"
import json
import time
from pathlib import Path

path = Path(__import__("sys").argv[1])
if path.exists() and path.stat().st_size:
    payload = json.loads(path.read_text())
    summary = payload.get("summary", {})
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] waiting: items={summary.get('items')} total={summary.get('total_mean')}")
else:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] waiting: summary missing")
PY
  sleep "$POLL_SECONDS"
done

echo "[$(date -Is)] C2 benchmark complete; building 3-way best-total summary" | tee -a "$LOG_PATH"
python scripts/build_hybrid_benchmark_summary.py \
  --summary outputs/tryon_benchmark/original_enhanced_summary.json --label original \
  --summary outputs/tryon_benchmark/enhanced_ft_summary.json --label finetuned \
  --summary "$WATCH_SUMMARY" --label original_c2 \
  --output-dir "$OUTPUT_DIR" \
  --summary-output "$SUMMARY_OUTPUT" \
  2>&1 | tee -a "$LOG_PATH"

python - <<'PY' | tee -a "$LOG_PATH"
import json
from pathlib import Path

paths = {
    "original": "outputs/tryon_benchmark/original_enhanced_summary.json",
    "finetuned": "outputs/tryon_benchmark/enhanced_ft_summary.json",
    "hybrid_2way": "outputs/tryon_benchmark/hybrid_best_total_summary.json",
    "original_c2": "outputs/tryon_benchmark/original_enhanced_c2_summary.json",
    "best_3way": "outputs/tryon_benchmark/best_total_3way_summary.json",
}
for name, path in paths.items():
    p = Path(path)
    if not p.exists():
        continue
    summary = json.loads(p.read_text()).get("summary", {})
    print(
        f"{name}: total={summary.get('total_mean')} object={summary.get('object_mean')} "
        f"person={summary.get('person_mean')} artifact={summary.get('artifact_mean')}"
    )
PY
