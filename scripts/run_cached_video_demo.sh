#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/demo/frontend"

FRONTEND_HOST="${OMNITRY_FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${OMNITRY_FRONTEND_PORT:-8080}"
CACHE_DELAY_MS="${VITE_TRYON_CACHE_DELAY_MS:-10000}"

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx cs338; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cs338
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[cached-demo] npm is required on this machine."
  exit 1
fi

if [[ ! -d node_modules ]]; then
  echo "[cached-demo] Installing frontend dependencies with npm ci..."
  npm ci
fi

LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "[cached-demo] GPU-free cached demo mode"
echo "[cached-demo] Delay: ${CACHE_DELAY_MS} ms"
echo "[cached-demo] Upload inputs from: ${ROOT_DIR}/demo/frontend/public/demo-cache-inputs"
echo "[cached-demo] Frontend: http://127.0.0.1:${FRONTEND_PORT}/"
if [[ -n "${LOCAL_IP}" ]]; then
  echo "[cached-demo] Network:  http://${LOCAL_IP}:${FRONTEND_PORT}/"
fi
echo

VITE_TRYON_USE_CACHE=true \
VITE_TRYON_CACHE_DELAY_MS="$CACHE_DELAY_MS" \
npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" --strictPort
