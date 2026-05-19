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

# GPTQ/Marlin runtime expects fp16-compatible MiniCPM sparse attention helpers.
# (BF16 ops in the custom backend conflict with the fp16 quantized linear path.)
BACKEND_DIR="${SUBMISSION_DIR}/sglang/python/sglang/srt/layers/attention"
for pyfile in "${BACKEND_DIR}/minicpm_backend.py" "${BACKEND_DIR}/minicpm_sparse_utils.py"; do
    if [ -f "${pyfile}" ]; then
        sed -i 's/torch\.bfloat16/torch.float16/g' "${pyfile}"
        sed -i 's/"bfloat16"/"float16"/g' "${pyfile}"
        echo "[prepare_env] patched $(basename "${pyfile}") for fp16 quantized run"
    fi
done

# Marlin path: symmetric W4A16 GPTQ weights produced by prepare_model.sh.
# Keep the MiniCPM-SALA sparse backend defaults; add gptq_marlin + fp16.
export GPTQMODEL_MARLIN_USE_FP32="${GPTQMODEL_MARLIN_USE_FP32:-1}"
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --dtype float16"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
