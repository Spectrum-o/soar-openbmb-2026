#!/usr/bin/env bash
# Run the full-W4A16 platform_acc local eval only if the GPU and target port
# are idle right now. This is the manual counterpart to the polling helper:
# it never waits, never kills processes, and exits 124 if fp8kv or another
# service is still occupying the machine.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VARIANT="${VARIANT:-submission_gptqmodel_full_w4a16}"
QUANT_OUT="${QUANT_OUT:-/root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16_platform_acc-quantized}"
NUM_SAMPLES="${NUM_SAMPLES:-150}"
CONCURRENCY="${CONCURRENCY:-32}"
PORT="${PORT:-31111}"
IDLE_MEM_MIB="${IDLE_MEM_MIB:-2000}"

export FULL_QUANT_PROFILE="${FULL_QUANT_PROFILE:-platform_acc}"
export GROUP_SIZE="${GROUP_SIZE:-64}"
export NUM_CALIB="${NUM_CALIB:-150}"
export CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"
export MAX_CALIB_LEN="${MAX_CALIB_LEN:-8192}"
export MAX_CALIB_WINDOWS="${MAX_CALIB_WINDOWS:-4}"
export QUANT_TIMEOUT_MIN="${QUANT_TIMEOUT_MIN:-120}"

gpu_mem_used_mib() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "0"
        return
    fi
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk '{sum += int($1)} END {print sum + 0}'
}

port_is_free() {
    python3 - "${PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        raise SystemExit(1)
PY
}

used="$(gpu_mem_used_mib)"
echo "[full-now] GPU memory used: ${used} MiB (threshold ${IDLE_MEM_MIB})"
if [ "${used}" -gt "${IDLE_MEM_MIB}" ]; then
    echo "[full-now] GPU is not idle; refusing to start full-W4A16" >&2
    exit 124
fi

if ! port_is_free; then
    echo "[full-now] port ${PORT} is busy; refusing to start full-W4A16" >&2
    exit 124
fi

echo "[full-now] starting full-W4A16 platform_acc local eval"
exec bash scripts/local_eval.sh \
    --variant "${VARIANT}" \
    --quant-out "${QUANT_OUT}" \
    --num-samples "${NUM_SAMPLES}" \
    --concurrency "${CONCURRENCY}" \
    --port "${PORT}" \
    --force-requant
