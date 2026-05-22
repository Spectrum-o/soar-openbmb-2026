#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[prepare_env] start $(date '+%F %T')"
echo "[prepare_env] submission dir: ${SUBMISSION_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[prepare_env] fatal: uv is required by the SOAR base environment" >&2
    return 1 2>/dev/null || exit 1
fi

# v4 baseline-matching safetynet: use the EXACT SGLANG_SERVER_ARGS from
# SOAR-Toolkit README's default. No --max-prefill-tokens, no
# --enable-mixed-chunk, just chunked-prefill-size 8192 (baseline default).
# If this passes, the identity package and default args are healthy; the
# v2/v3 failure is likely caused by one of the added flags.
# If this fails, the problem is elsewhere (model load / platform contract).
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
