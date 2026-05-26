#!/bin/bash
# Launch MiniCPM-SALA inference server with FP8 KV cache quantization and
# CUDA graph enabled. These defaults mirror the current zyn full-eval run.
#
# Why FP8 KV cache:
#   - KV cache size halved (bf16 -> fp8) -> more concurrency / longer context
#   - decode is memory-bound -> less HBM traffic -> faster
#   - quality cost is typically < 1% on benchmarks (verify with eval_model.py)
#
# Usage:
#   bash run_sala.sh                    # defaults to ./models/MiniCPM-SALA
#   MODEL_PATH=/abs/path bash run_sala.sh
#   KV_DTYPE=fp8_e5m2 bash run_sala.sh  # try e5m2 variant

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/sglang_minicpm_sala_env"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/MiniCPM-SALA}"
PORT="${PORT:-31111}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-gptq_marlin}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 4 8 12 16 24 32}"

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
export PATH="${VENV_DIR}/bin:${PATH}"
export SGLANG_ENABLE_DEGENERATE_STOP="${SGLANG_ENABLE_DEGENERATE_STOP:-1}"
export SGLANG_DEGENERATE_STOP_MIN_TOKENS="${SGLANG_DEGENERATE_STOP_MIN_TOKENS:-3072}"
export SGLANG_DEGENERATE_STOP_CHECK_INTERVAL="${SGLANG_DEGENERATE_STOP_CHECK_INTERVAL:-64}"
export SGLANG_DEGENERATE_STOP_REPEAT_COUNT="${SGLANG_DEGENERATE_STOP_REPEAT_COUNT:-6}"
export SGLANG_DEGENERATE_STOP_TAIL_TOKENS="${SGLANG_DEGENERATE_STOP_TAIL_TOKENS:-2048}"

python3 -m sglang.launch_server \
    --model "${MODEL_PATH}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --quantization "${QUANTIZATION}" \
    --kv-cache-dtype "${KV_DTYPE}" \
    --dtype "${DTYPE}" \
    --skip-server-warmup \
    --port "${PORT}" \
    --dense-as-sparse \
    --cuda-graph-bs ${CUDA_GRAPH_BS}
