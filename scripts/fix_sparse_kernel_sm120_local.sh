#!/usr/bin/env bash
# Install a Blackwell-capable sparse_kernel_extension for local AutoDL runs.
#
# The MiniCPM sparse attention backend imports sparse_kernel_extension from the
# OpenBMB sparse_kernel checkout. Some local top-level .so builds only contain
# sm_80/sm_90 cubins, which makes SGLang fail on Blackwell (sm_120) during CUDA
# graph capture with "no kernel image is available for execution on the device".
# This script is local-only; platform submissions must keep their own runtime.

set -euo pipefail

SPARSE_DIR="${SPARSE_DIR:-/autodl-fs/data/ziqian/sglang_minicpm_sala/3rdparty/sparse_kernel}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/zyn/sglang_full_w4a16/.venv_py310_full/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUOBJDUMP="${CUOBJDUMP:-${CUDA_HOME}/bin/cuobjdump}"
TARGET_SO="${SPARSE_DIR}/sparse_kernel_extension.cpython-310-x86_64-linux-gnu.so"
BUILT_SO="${SPARSE_DIR}/build/lib.linux-x86_64-cpython-310/sparse_kernel_extension.cpython-310-x86_64-linux-gnu.so"

has_sm120() {
    local so_path="$1"
    [ -f "${so_path}" ] || return 1
    if [ -x "${CUOBJDUMP}" ]; then
        "${CUOBJDUMP}" --list-elf "${so_path}" 2>/dev/null | grep -q 'sm_120'
    else
        strings "${so_path}" | grep -qE 'sm_120|compute_120'
    fi
}

echo "============================================================"
echo " fix_sparse_kernel_sm120_local"
echo "============================================================"
echo "sparse_dir: ${SPARSE_DIR}"
echo "python:     ${PYTHON_BIN}"
echo "cuda_home:  ${CUDA_HOME}"
echo "target_so:  ${TARGET_SO}"
echo "built_so:   ${BUILT_SO}"
echo "============================================================"

if [ ! -d "${SPARSE_DIR}" ]; then
    echo "error: SPARSE_DIR does not exist: ${SPARSE_DIR}" >&2
    exit 2
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi

if has_sm120 "${TARGET_SO}"; then
    echo "[sparse-sm120] target already contains sm_120"
    exit 0
fi

if has_sm120 "${BUILT_SO}"; then
    backup="${TARGET_SO}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "[sparse-sm120] installing existing sm_120 build"
    if [ -f "${TARGET_SO}" ]; then
        cp -a "${TARGET_SO}" "${backup}"
        echo "[sparse-sm120] backup: ${backup}"
    fi
    cp -a "${BUILT_SO}" "${TARGET_SO}"
else
    echo "[sparse-sm120] no existing sm_120 build found; rebuilding"
    if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
        echo "error: nvcc not found at ${CUDA_HOME}/bin/nvcc" >&2
        exit 2
    fi
    PATH="${CUDA_HOME}/bin:${PATH}" \
    CUDA_HOME="${CUDA_HOME}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0;12.0}" \
        "${PYTHON_BIN}" setup.py build_ext --inplace
fi

if ! has_sm120 "${TARGET_SO}"; then
    echo "error: target still lacks sm_120 after install/rebuild: ${TARGET_SO}" >&2
    exit 1
fi

echo "[sparse-sm120] verified target contains sm_120"
"${PYTHON_BIN}" - <<'PY'
import torch
import sparse_kernel_extension

print("[sparse-sm120] import ok:", sparse_kernel_extension.__file__)
print("[sparse-sm120] torch:", torch.__version__)
PY
