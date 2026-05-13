#!/bin/bash
# Launch MiniCPM-SALA inference server with tuned chunked prefill parameters.
#
# Tuning history (RTX PRO 6000 Blackwell, 64 prompts × 4096-in × 512-out random-ids):
#   chunked_prefill_size=8192  -> 267.67 tok/s out, 34049ms TTFT  (baseline)
#   chunked_prefill_size=32768 -> 425.58 tok/s out, 13545ms TTFT  (+59% throughput)
#   chunked_prefill_size=65536 -> 488.84 tok/s out, 12529ms TTFT  (+83% total)
#
# Gotchas:
#   - sglang default --max-prefill-tokens=16384 caps actual prefill regardless of
#     chunked-prefill-size. Must set --max-prefill-tokens to match.
#   - With large chunks, sglang's auto-calc of mem_fraction_static can go negative.
#     Must explicitly set --mem-fraction-static (0.80 verified working on Blackwell).
#
# Usage:
#   bash run_sala.sh
#   MODEL_PATH=/abs/path bash run_sala.sh
#   PORT=8000 bash run_sala.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/sglang_minicpm_sala_env"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/MiniCPM-SALA}"
PORT="${PORT:-31111}"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Error: venv not found at ${VENV_DIR}"
    echo "Run: bash install_minicpm_sala.sh"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: model not found at ${MODEL_PATH}"
    echo "Run: huggingface-cli download openbmb/MiniCPM-SALA --local-dir ${MODEL_PATH}"
    exit 1
fi

source "${VENV_DIR}/bin/activate"

python3 -m sglang.launch_server \
    --model "${MODEL_PATH}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 65536 \
    --max-prefill-tokens 65536 \
    --max-running-requests 32 \
    --mem-fraction-static 0.80 \
    --skip-server-warmup \
    --port "${PORT}" \
    --dense-as-sparse
