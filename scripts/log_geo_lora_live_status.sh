#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:-}"
if [[ -z "$TRAIN_PID" ]]; then
  echo "Usage: $0 <train-pid>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
LOG_PATH="${LOG_PATH:-outputs/enhance/geo_lora_live_status.log}"
CKPT="checkpoints/enhance/omnitry_geo_lora.safetensors"
METRICS="outputs/enhance/geo_lora_train_metrics.json"

mkdir -p "$(dirname "$LOG_PATH")"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  {
    echo "[$(date -Is)] train pid $TRAIN_PID is running"
    ps -o pid,ppid,etime,pcpu,pmem,stat,cmd -p "$TRAIN_PID" \
      -p "$(pgrep -P "$TRAIN_PID" | tr '\n' ',' | sed 's/,$//')" 2>/dev/null || true
    echo "-- gpu 2,3,7 --"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits \
      | awk -F', ' '$1 == 2 || $1 == 3 || $1 == 7 {print "gpu " $1 ": util=" $2 "% mem=" $3 "/" $4 " MiB"}' || true
    echo "-- outputs --"
    ls -lh "$CKPT" "$METRICS" 2>/dev/null || true
    echo
  } >> "$LOG_PATH"
  sleep "$INTERVAL_SECONDS"
done

{
  echo "[$(date -Is)] train pid $TRAIN_PID exited"
  ls -lh "$CKPT" "$METRICS" 2>/dev/null || true
  echo
} >> "$LOG_PATH"
