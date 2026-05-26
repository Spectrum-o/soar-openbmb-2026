#!/usr/bin/env bash
# Run the full-W4A16 sharded local eval with the platform-shaped Python 3.10
# environment built under this repo.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_py310_full/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    echo "hint: build the local py310 env first, or pass PYTHON_BIN=/path/to/python" >&2
    exit 2
fi
export PYTHON_BIN

# The bundled cp310 flash-attn wheel needs a newer libstdc++ than the base OS
# path provides on this AutoDL node. Platform images already have a compatible
# runtime; this is only for local reproduction.
export LD_LIBRARY_PATH="/root/miniconda3/lib:${LD_LIBRARY_PATH:-}"

export FULL_QUANT_PROFILE="${FULL_QUANT_PROFILE:-platform_acc}"
export GROUP_SIZE="${GROUP_SIZE:-64}"
export NUM_CALIB="${NUM_CALIB:-150}"
export CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"
export MAX_CALIB_LEN="${MAX_CALIB_LEN:-8192}"
export MAX_CALIB_WINDOWS="${MAX_CALIB_WINDOWS:-4}"
export QUANT_TIMEOUT_MIN="${QUANT_TIMEOUT_MIN:-120}"
export SHARDED_EVAL="${SHARDED_EVAL:-1}"
export NUM_SAMPLES="${NUM_SAMPLES:-150}"
export SHARD_SIZE="${SHARD_SIZE:-30}"
export CONCURRENCY="${CONCURRENCY:-32}"
export PORT="${PORT:-31111}"
export IDLE_MEM_MIB="${IDLE_MEM_MIB:-2000}"

echo "[full-py310] repo:       ${REPO_ROOT}"
echo "[full-py310] python:     ${PYTHON_BIN}"
echo "[full-py310] group size: ${GROUP_SIZE}"
echo "[full-py310] calib:      ${NUM_CALIB} x ${MAX_CALIB_LEN}, ${CALIB_WINDOW_MODE}/${MAX_CALIB_WINDOWS}"
echo "[full-py310] shards:     ${NUM_SAMPLES} samples, shard size ${SHARD_SIZE}"

exec bash scripts/run_full_w4a16_platform_acc_now.sh
