#!/bin/bash
# Launch MiniCPM-SALA inference server.
# Usage:
#   bash run_sala.sh                    # defaults to ./models/MiniCPM-SALA
#   MODEL_PATH=/abs/path bash run_sala.sh

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
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port "${PORT}" \
    --dense-as-sparse
