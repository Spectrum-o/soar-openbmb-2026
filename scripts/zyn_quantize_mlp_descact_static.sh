#!/usr/bin/env bash
set -euo pipefail

# Local MLP-only GPTQ candidate:
#   - BF16 source model, not the compressed W4A16 platform artifact
#   - group_size=128, sym=True
#   - desc_act=True + static_groups=True
#   - same tail calibration recipe as the current best runnable artifact

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
CALIB_JSONL="${CALIB_JSONL:-/root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl}"
QUANT_SCRIPT="${QUANT_SCRIPT:-/root/autodl-tmp/zyn/sglang_full_w4a16/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe/quantize_gptqmodel_w4a16.py}"
QUANT_VENV="${QUANT_VENV:-/root/autodl-tmp/zyn/sglang_full_w4a16/.venv_py310_full}"
OUTPUT_MODEL="${OUTPUT_MODEL:-/autodl-fs/data/zyn/models/submission_gptqmodel_mlp_descact_static_20260527-quantized}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/zyn_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/quant_mlp_descact_static_$(date +%Y%m%d_%H%M%S).log}"

NUM_CALIB="${NUM_CALIB:-256}"
MAX_CALIB_LEN="${MAX_CALIB_LEN:-8192}"
GROUP_SIZE="${GROUP_SIZE:-128}"
QUANT_TIMEOUT_MIN="${QUANT_TIMEOUT_MIN:-120}"
FORCE="${FORCE:-0}"

if [ ! -d "${BASE_MODEL}" ]; then
    echo "error: BASE_MODEL does not exist: ${BASE_MODEL}" >&2
    exit 1
fi
if [ ! -f "${CALIB_JSONL}" ]; then
    echo "error: CALIB_JSONL does not exist: ${CALIB_JSONL}" >&2
    exit 1
fi
if [ ! -f "${QUANT_SCRIPT}" ]; then
    echo "error: QUANT_SCRIPT does not exist: ${QUANT_SCRIPT}" >&2
    exit 1
fi
if [ ! -x "${QUANT_VENV}/bin/python" ]; then
    echo "error: QUANT_VENV missing python: ${QUANT_VENV}" >&2
    exit 1
fi
if [ -e "${OUTPUT_MODEL}" ] && [ "${FORCE}" != "1" ]; then
    echo "error: OUTPUT_MODEL already exists: ${OUTPUT_MODEL}" >&2
    echo "set FORCE=1 to overwrite after checking it is safe" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_MODEL}")"

if [ -e "${OUTPUT_MODEL}" ]; then
    rm -rf "${OUTPUT_MODEL}"
fi

echo "[quant] repo:        ${REPO_ROOT}"
echo "[quant] base:        ${BASE_MODEL}"
echo "[quant] calib:       ${CALIB_JSONL}"
echo "[quant] output:      ${OUTPUT_MODEL}"
echo "[quant] quant script:${QUANT_SCRIPT}"
echo "[quant] log:         ${LOG_FILE}"
echo "[quant] knobs:       group=${GROUP_SIZE} desc_act=True static_groups=True num_calib=${NUM_CALIB} max_len=${MAX_CALIB_LEN}"

source "${QUANT_VENV}/bin/activate"

export GPTQ_SYM=True
export GPTQ_DESC_ACT=True
export GPTQ_STATIC_GROUPS=True

set +e
timeout "${QUANT_TIMEOUT_MIN}m" python "${QUANT_SCRIPT}" \
    --input "${BASE_MODEL}" \
    --output "${OUTPUT_MODEL}" \
    --calib-jsonl "${CALIB_JSONL}" \
    --num-calib "${NUM_CALIB}" \
    --max-calib-len "${MAX_CALIB_LEN}" \
    --calib-window-mode tail \
    --group-size "${GROUP_SIZE}" \
    --no-offload-disk \
    2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

if [ "${status}" -eq 124 ]; then
    echo "[quant] FATAL: quantization exceeded ${QUANT_TIMEOUT_MIN} min" >&2
    exit 124
fi
if [ "${status}" -ne 0 ]; then
    echo "[quant] FATAL: quantization failed with status ${status}" >&2
    exit "${status}"
fi

echo "[quant] done: ${OUTPUT_MODEL}"
if [ -f "${OUTPUT_MODEL}/quantize_config.json" ]; then
    echo "[quant] quantize_config:"
    sed -n '1,220p' "${OUTPUT_MODEL}/quantize_config.json"
fi
