#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKEND_HOST="${OMNITRY_BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${OMNITRY_BACKEND_PORT:-8010}"
FRONTEND_HOST="${OMNITRY_FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${OMNITRY_FRONTEND_PORT:-8080}"
BACKEND_LOG="${OMNITRY_BACKEND_LOG:-outputs/live_demo_backend_server.log}"
DEMO_GPU="${OMNITRY_DEMO_GPU:-7}"
CPU_OFFLOAD="${OMNITRY_CPU_OFFLOAD:-0}"
MAX_AREA="${OMNITRY_MAX_AREA:-262144}"
MAX_CANDIDATES="${OMNITRY_MAX_CANDIDATES:-1}"

BACKEND_PID=""
BACKEND_TAIL_PID=""

port_in_use() {
  local port="$1"
  ss -ltn | awk -v pattern=":${port}$" '$4 ~ pattern { found = 1 } END { exit found ? 0 : 1 }'
}

cleanup() {
  set +e
  echo
  echo "[live-demo] Stopping frontend/backend..."
  if [[ -n "${BACKEND_TAIL_PID}" ]]; then
    kill "${BACKEND_TAIL_PID}" 2>/dev/null || true
    wait "${BACKEND_TAIL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID}" ]]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if port_in_use "$BACKEND_PORT"; then
  echo "[live-demo] Backend port ${BACKEND_PORT} is already in use."
  exit 1
fi

if port_in_use "$FRONTEND_PORT"; then
  echo "[live-demo] Frontend port ${FRONTEND_PORT} is already in use."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338

if ! command -v npm >/dev/null 2>&1; then
  echo "[live-demo] npm is not available in conda env cs338."
  echo "[live-demo] Install it with: conda install -y -c conda-forge 'nodejs>=20,<23'"
  exit 1
fi

mkdir -p "$(dirname "$BACKEND_LOG")"
: > "$BACKEND_LOG"

echo "[live-demo] Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "[live-demo] Backend GPU: ${DEMO_GPU} (override with OMNITRY_DEMO_GPU=2, for example)"
echo "[live-demo] CPU offload: ${CPU_OFFLOAD} (set OMNITRY_CPU_OFFLOAD=1 if VRAM is tight)"
echo "[live-demo] Max area: ${MAX_AREA} pixels (512x512 default for fast demo)"
echo "[live-demo] Max geometry candidates: ${MAX_CANDIDATES}"
OMNITRY_BACKEND_HOST="$BACKEND_HOST" \
OMNITRY_BACKEND_PORT="$BACKEND_PORT" \
OMNITRY_CPU_OFFLOAD="$CPU_OFFLOAD" \
OMNITRY_MAX_AREA="$MAX_AREA" \
OMNITRY_MAX_CANDIDATES="$MAX_CANDIDATES" \
CUDA_VISIBLE_DEVICES="$DEMO_GPU" \
bash demo/backend/run.sh > "$BACKEND_LOG" 2>&1 &
BACKEND_PID="$!"

tail -n +1 -f "$BACKEND_LOG" &
BACKEND_TAIL_PID="$!"

echo "[live-demo] Waiting for backend health..."
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${BACKEND_PORT}/api/v1/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "[live-demo] Backend exited early. See ${BACKEND_LOG}"
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${BACKEND_PORT}/api/v1/health" >/dev/null 2>&1; then
  echo "[live-demo] Backend did not become healthy. See ${BACKEND_LOG}"
  exit 1
fi

if [[ ! -d demo/frontend/node_modules ]]; then
  echo "[live-demo] Installing frontend dependencies with npm ci..."
  (cd demo/frontend && npm ci)
fi

LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "[live-demo] Backend OK:  http://127.0.0.1:${BACKEND_PORT}/api/v1/health"
echo "[live-demo] Frontend:    http://127.0.0.1:${FRONTEND_PORT}/"
if [[ -n "${LOCAL_IP}" ]]; then
  echo "[live-demo] Network:     http://${LOCAL_IP}:${FRONTEND_PORT}/"
fi
echo "[live-demo] Press Ctrl+C to stop both."
echo

cd demo/frontend
VITE_BACKEND_PROXY_TARGET="http://127.0.0.1:${BACKEND_PORT}" \
npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" --strictPort
