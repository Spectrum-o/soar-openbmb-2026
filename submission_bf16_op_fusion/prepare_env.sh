#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[prepare_env] start $(date '+%F %T')"
echo "[prepare_env] submission dir: ${SUBMISSION_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[prepare_env] fatal: uv is required by the SOAR base environment" >&2
    return 1 2>/dev/null || exit 1
fi

# BF16 + op-fusion variant.
#
# WHAT THIS DOES (vs BF16 baseline 19.13):
#   1. chunked-prefill 65536 + max-prefill-tokens 65536 + mixed-chunk
#      → per local benchmark (2026-05-14), 8K→32K chunked-prefill gives
#        +59% throughput. Pushing to 65K is the run_sala.sh setting (not
#        independently benchmarked, but no known correctness concern).
#   2. mem-fraction-static 0.80
#      → matches run_sala.sh. Lets KV cache grow into reserved memory.
#   3. RMSNorm+residual fused via sglang's fused_add_rmsnorm
#      + drop FP32 upcast around RoPE
#      → from SOAR champion notes 笔记 04. Saves ~2 kernel launches per
#        decoder layer × 32 layers × decode step.
#
# HOW THE OP-FUSION CODE GETS APPLIED:
#   We do NOT bundle the full sglang/python source (v2 of safetynet bundled
#   the whole thing and crashed in 13s — SUBMISSIONS.md row 104). Instead,
#   we ship ONLY the patched minicpm.py model file as an "overlay" and
#   replace the base env's version at prepare_env time. This minimizes the
#   surface area of code substitution.
#
# RISK NOTE:
#   - The overlay file is from origin/perf/op-fusion at commit 79b0c20f4.
#   - Pure-torch CPU equivalence verified by commit 87c22ec88 (test/srt/...).
#   - GPU runtime correctness is NOT verified for this exact file in this
#     exact platform context. RECOMMEND: GPU-validate locally before
#     submitting (see README_SUBMISSION.md).

# Find the platform's installed sglang minicpm.py and back it up + overlay
SGLANG_MINICPM=$(python3 -c 'import sglang.srt.models.minicpm; print(sglang.srt.models.minicpm.__file__)' 2>/dev/null || true)
if [ -z "${SGLANG_MINICPM}" ] || [ ! -f "${SGLANG_MINICPM}" ]; then
    echo "[prepare_env] FATAL: could not locate base env's sglang/srt/models/minicpm.py" >&2
    echo "[prepare_env] FATAL: this variant requires platform's base env to have sglang preinstalled" >&2
    exit 1
fi

OVERLAY="${SUBMISSION_DIR}/sglang_overlay/minicpm.py"
if [ ! -f "${OVERLAY}" ]; then
    echo "[prepare_env] FATAL: overlay file missing at ${OVERLAY}" >&2
    exit 1
fi

echo "[prepare_env] target minicpm.py: ${SGLANG_MINICPM}"
echo "[prepare_env] overlay   minicpm.py: ${OVERLAY}"
echo "[prepare_env] backing up original to ${SGLANG_MINICPM}.bak.op_fusion"
cp "${SGLANG_MINICPM}" "${SGLANG_MINICPM}.bak.op_fusion"
cp "${OVERLAY}" "${SGLANG_MINICPM}"

# Sanity check: the overlay must be importable
if ! python3 -c "import sglang.srt.models.minicpm; print('[overlay] import OK')"; then
    echo "[prepare_env] FATAL: overlay minicpm.py failed to import. Restoring backup." >&2
    cp "${SGLANG_MINICPM}.bak.op_fusion" "${SGLANG_MINICPM}"
    exit 1
fi

# BF16 path — NO fp16 patch, NO quantization. Sparse backend keeps its bf16
# implementation as-is, matching baseline.
#
# Scheduling optimizations:
#   --chunked-prefill-size 65536  — push to run_sala.sh's 65K
#   --max-prefill-tokens 65536    — must match
#   --mem-fraction-static 0.80    — matches run_sala.sh
#   --enable-mixed-chunk          — SARATHI piggyback
# Correctness MUST be identical to baseline at the math level (op-fusion is
# numerically equivalent per 87c22ec88 unit test).
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 65536 --max-prefill-tokens 65536 --mem-fraction-static 0.80 --enable-mixed-chunk --skip-server-warmup --dense-as-sparse"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
