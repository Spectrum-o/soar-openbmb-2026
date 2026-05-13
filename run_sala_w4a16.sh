#!/bin/bash
# Launch MiniCPM-SALA inference server with W4A16 (GPTQ + Marlin) quantization.
#
# Expects the W4A16 model produced by quantize_to_w4a16.py at MODEL_PATH.
# Inherits the tuned chunked-prefill settings from the chunked-prefill-tuned config:
#   --chunked-prefill-size 65536 --max-prefill-tokens 65536 --mem-fraction-static 0.80
#
# Usage:
#   bash run_sala_w4a16.sh
#   MODEL_PATH=/abs/path/to/MiniCPM-SALA-W4A16 bash run_sala_w4a16.sh
#   PORT=8000 bash run_sala_w4a16.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/sglang_minicpm_sala_env"
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16}"
PORT="${PORT:-31111}"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Error: venv not found at ${VENV_DIR}"
    echo "Run: bash install_minicpm_sala.sh"
    exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Error: W4A16 model not found at ${MODEL_PATH}"
    echo "Run quantize_to_w4a16.py first:"
    echo "  python quantize_to_w4a16.py \\"
    echo "      --model /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\"
    echo "      --output ${MODEL_PATH}"
    exit 1
fi

source "${VENV_DIR}/bin/activate"

# Note: --quantization compressed-tensors is usually optional (auto-detected from
# model config.json), but setting it explicitly avoids ambiguity.
python3 -m sglang.launch_server \
    --model "${MODEL_PATH}" \
    --trust-remote-code \
    --disable-radix-cache \
    --quantization compressed-tensors \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 65536 \
    --max-prefill-tokens 65536 \
    --max-running-requests 32 \
    --mem-fraction-static 0.80 \
    --skip-server-warmup \
    --port "${PORT}" \
    --dense-as-sparse
