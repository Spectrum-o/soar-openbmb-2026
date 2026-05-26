#!/usr/bin/env bash
# Wait for CUDA VRAM to be idle, then run the full-W4A16 platform_acc local eval.
#
# This script is intentionally non-destructive: it never kills GPU processes.
# Use it while another experiment is running; it will poll until memory use
# drops below the threshold, then start the full quantization/eval job.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VARIANT="${VARIANT:-submission_gptqmodel_full_w4a16}"
QUANT_OUT="${QUANT_OUT:-/root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16_platform_acc-quantized}"
NUM_SAMPLES="${NUM_SAMPLES:-150}"
SHARD_SIZE="${SHARD_SIZE:-30}"
SHARDED_EVAL="${SHARDED_EVAL:-1}"
CONCURRENCY="${CONCURRENCY:-32}"
PORT="${PORT:-31111}"
IDLE_MEM_MIB="${IDLE_MEM_MIB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-0}"  # 0 = wait forever

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

echo "============================================================"
echo " run_full_w4a16_platform_acc_when_idle"
echo "============================================================"
echo "variant:       ${VARIANT}"
echo "quant_out:     ${QUANT_OUT}"
echo "samples:       ${NUM_SAMPLES}"
echo "sharded_eval:  ${SHARDED_EVAL}"
echo "shard_size:    ${SHARD_SIZE}"
echo "concurrency:   ${CONCURRENCY}"
echo "idle_mem_mib:  ${IDLE_MEM_MIB}"
echo "poll_seconds:  ${POLL_SECONDS}"
echo "max_wait_min:  ${MAX_WAIT_MIN}"
echo "profile:       ${FULL_QUANT_PROFILE}"
echo "group_size:    ${GROUP_SIZE}"
echo "num_calib:     ${NUM_CALIB}"
echo "window_mode:   ${CALIB_WINDOW_MODE}"
echo "max_calib_len: ${MAX_CALIB_LEN}"
echo "max_windows:   ${MAX_CALIB_WINDOWS}"
echo "============================================================"

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

started=$(date +%s)
while true; do
    used="$(gpu_mem_used_mib)"
    now="$(date '+%F %T')"
    echo "[${now}] GPU memory used: ${used} MiB"
    if [ "${used}" -le "${IDLE_MEM_MIB}" ]; then
        if port_is_free; then
            echo "[wait] GPU appears idle and port ${PORT} is free; starting full-W4A16 platform_acc run"
            break
        fi
        echo "[wait] GPU appears idle, but port ${PORT} is still busy; continuing to poll"
    fi

    if [ "${MAX_WAIT_MIN}" -gt 0 ]; then
        elapsed_min=$(( ($(date +%s) - started) / 60 ))
        if [ "${elapsed_min}" -ge "${MAX_WAIT_MIN}" ]; then
            echo "[wait] exceeded MAX_WAIT_MIN=${MAX_WAIT_MIN}; exiting without starting" >&2
            exit 124
        fi
    fi
    sleep "${POLL_SECONDS}"
done

if [ "${SHARDED_EVAL}" = "1" ]; then
    exec bash scripts/local_eval_sharded.sh \
        --variant "${VARIANT}" \
        --quant-out "${QUANT_OUT}" \
        --max-samples "${NUM_SAMPLES}" \
        --shard-size "${SHARD_SIZE}" \
        --concurrency "${CONCURRENCY}" \
        --port "${PORT}" \
        --force-requant
fi

exec bash scripts/local_eval.sh \
    --variant "${VARIANT}" \
    --quant-out "${QUANT_OUT}" \
    --num-samples "${NUM_SAMPLES}" \
    --concurrency "${CONCURRENCY}" \
    --port "${PORT}" \
    --force-requant
