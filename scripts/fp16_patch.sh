#!/usr/bin/env bash
# Helper to apply / restore the bf16 -> fp16 patch that submission's
# prepare_env.sh does to the MiniCPM sparse attention backend.
#
# The patch is required when running with --quantization gptq_marlin
# + --dtype float16: Marlin's GEMM emits fp16, but the unmodified
# sparse backend hardcodes bf16 and crashes at the type boundary.
#
# Usage:
#   bash scripts/fp16_patch.sh apply      # forward-patch (idempotent)
#   bash scripts/fp16_patch.sh restore    # git-checkout to revert to upstream
#   bash scripts/fp16_patch.sh status     # show current state
#
# IMPORTANT: this script modifies your working tree. `restore` requires
# the files to be in git's HEAD (i.e. you haven't committed the patched
# versions).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FILES=(
    "python/sglang/srt/layers/attention/minicpm_backend.py"
    "python/sglang/srt/layers/attention/minicpm_sparse_utils.py"
)

cmd="${1:-status}"

show_status() {
    for f in "${FILES[@]}"; do
        if [ ! -f "${f}" ]; then
            echo "  ${f}: MISSING"
            continue
        fi
        bf=$(grep -cE 'bfloat16' "${f}" 2>/dev/null || true)
        f16=$(grep -cE '"float16"|torch\.float16' "${f}" 2>/dev/null || true)
        bf=${bf:-0}
        f16=${f16:-0}
        if [ "${bf}" -eq 0 ]; then
            echo "  ${f}: PATCHED (no bf16 left, ${f16} fp16 refs)"
        elif [ "${f16}" -eq 0 ]; then
            echo "  ${f}: UPSTREAM (${bf} bf16 refs, no fp16)"
        else
            echo "  ${f}: MIXED (${bf} bf16, ${f16} fp16) — manual edit?"
        fi
    done
}

case "${cmd}" in
    status)
        echo "fp16 patch status:"
        show_status
        ;;
    apply)
        echo "applying bf16 -> fp16 patch to:"
        for f in "${FILES[@]}"; do
            [ -f "${f}" ] || { echo "  skip ${f} (missing)"; continue; }
            sed -i 's/torch\.bfloat16/torch.float16/g' "${f}"
            sed -i 's/"bfloat16"/"float16"/g' "${f}"
            echo "  patched ${f}"
        done
        echo
        echo "status after apply:"
        show_status
        ;;
    restore)
        echo "git-checkout to restore upstream versions:"
        for f in "${FILES[@]}"; do
            [ -f "${f}" ] || { echo "  skip ${f} (missing)"; continue; }
            if git diff --quiet HEAD -- "${f}" 2>/dev/null; then
                echo "  ${f}: already clean"
            else
                git checkout HEAD -- "${f}"
                echo "  restored ${f}"
            fi
        done
        echo
        echo "status after restore:"
        show_status
        ;;
    *)
        echo "usage: $0 {apply|restore|status}" >&2
        exit 2
        ;;
esac
