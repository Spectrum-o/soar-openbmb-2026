#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[prepare_env] start $(date '+%F %T')"
echo "[prepare_env] submission dir: ${SUBMISSION_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[prepare_env] fatal: uv is required by the SOAR base environment" >&2
    return 1 2>/dev/null || exit 1
fi

if [ -d "${SUBMISSION_DIR}/sglang/python" ]; then
    echo "[prepare_env] installing submitted sglang/python in editable mode"
    uv pip install --no-deps -e "${SUBMISSION_DIR}/sglang/python"
fi

# GPTQModel for onsite Hessian-based W4A16 quantization. Pin to a recent
# stable release (7.x line) — newer ones have MiniCPM family support and
# stable QuantizeConfig API.
echo "[prepare_env] installing GPTQModel + supporting packages"
uv pip install -v "gptqmodel>=7.0,<8.0" "transformers>=4.45" accelerate ninja

# GPTQ/Marlin runtime expects fp16-compatible MiniCPM sparse attention helpers.
# This patch was verified necessary in earlier submissions — the sparse
# backend hardcodes bf16, but Marlin GEMM emits fp16.
BACKEND_DIR="${SUBMISSION_DIR}/sglang/python/sglang/srt/layers/attention"
for pyfile in "${BACKEND_DIR}/minicpm_backend.py" "${BACKEND_DIR}/minicpm_sparse_utils.py"; do
    if [ -f "${pyfile}" ]; then
        sed -i 's/torch\.bfloat16/torch.float16/g' "${pyfile}"
        sed -i 's/"bfloat16"/"float16"/g' "${pyfile}"
        echo "[prepare_env] patched $(basename "${pyfile}") for fp16 quantized run"
    fi
done

# Marlin GEMM env. Use FP32 accumulation for higher numerical stability
# during decode (small perf cost; safer for correctness gate).
export GPTQMODEL_MARLIN_USE_FP32="${GPTQMODEL_MARLIN_USE_FP32:-1}"

# SGLang server args. NOTE: NO --kv-cache-dtype fp8_* (verified incompatible
# with MiniCPM sparse backend in earlier submissions).
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --dtype float16"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
