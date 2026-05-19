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

# BF16 path — NO fp16 patch, NO quantization. Sparse backend keeps its bf16
# implementation as-is, matching baseline.

# Scheduling-only optimizations on top of baseline:
#   --chunked-prefill-size 32768  (was 8192)  — local 5/14 bench: +59% throughput, -60% TTFT
#   --max-prefill-tokens 32768                  — must match, sglang's hidden default is 16384
#   --enable-mixed-chunk                        — SARATHI piggyback for decode under prefill
# Correctness MUST be identical to baseline (no model change).
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --enable-mixed-chunk --skip-server-warmup --dense-as-sparse"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
