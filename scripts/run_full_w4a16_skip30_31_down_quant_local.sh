#!/usr/bin/env bash
# Quantize the current narrow mixed-sensitive full-W4A16 candidate.
#
# This is a local AutoDL reproduction helper for the artifact currently being
# tested. It does not change the platform serving path; prepare_env.sh remains
# the authority for platform SGLang args and keeps CUDA graph enabled.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_py310_full/bin/python}"
INPUT_DIR="${INPUT_DIR:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16_g64_skip30_31_down-quantized}"
LOG_DIR="${LOG_DIR:-/root/autodl-fs/zyn/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/full_w4a16_g64_skip30_31_down_quant.log}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [ ! -d "${INPUT_DIR}" ]; then
    echo "error: INPUT_DIR does not exist: ${INPUT_DIR}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

export PYTHON_BIN
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

# Local AutoDL has an older system libstdc++; the CP310 flash-attn wheel needs
# CXXABI_1.3.15. Platform images already provide a compatible runtime, so keep
# this local-only.
if [ -d /root/miniconda3/lib ]; then
    export LD_LIBRARY_PATH="/root/miniconda3/lib:${LD_LIBRARY_PATH:-}"
fi

export FULL_QUANT_PROFILE="${FULL_QUANT_PROFILE:-platform_acc_skip30_31_down}"
export GROUP_SIZE="${GROUP_SIZE:-64}"
export NUM_CALIB="${NUM_CALIB:-150}"
export MAX_CALIB_LEN="${MAX_CALIB_LEN:-8192}"
export CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"
export MAX_CALIB_WINDOWS="${MAX_CALIB_WINDOWS:-4}"
export QUANT_TIMEOUT_MIN="${QUANT_TIMEOUT_MIN:-150}"
export MIXED_SKIP_LAYERS="${MIXED_SKIP_LAYERS:-30,31}"
export MIXED_SKIP_MODULES="${MIXED_SKIP_MODULES:-down}"

echo "============================================================"
echo " run_full_w4a16_skip30_31_down_quant_local"
echo "============================================================"
echo "python:            ${PYTHON_BIN}"
echo "input:             ${INPUT_DIR}"
echo "output:            ${OUTPUT_DIR}"
echo "log:               ${LOG_PATH}"
echo "profile:           ${FULL_QUANT_PROFILE}"
echo "group_size:        ${GROUP_SIZE}"
echo "mixed_skip_layers: ${MIXED_SKIP_LAYERS}"
echo "mixed_skip_modules:${MIXED_SKIP_MODULES}"
echo "calib:             ${NUM_CALIB} x ${MAX_CALIB_LEN}, ${CALIB_WINDOW_MODE}/${MAX_CALIB_WINDOWS}"
echo "============================================================"

exec bash submission_gptqmodel_full_w4a16/prepare_model.sh \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_PATH}"
