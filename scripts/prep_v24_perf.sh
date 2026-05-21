#!/usr/bin/env bash
# prep_v24_perf.sh — restore chunked-prefill 65K into submission tarballs.
#
# v23 was deliberately submitted with chunked-prefill 8192 (the historical
# pre-cherry-pick value) so its platform result isolates the H1+H4 fixes
# from chunked-prefill as a confounding variable. Now that v23 has
# confirmed acc != 0 on the platform, v24 layers on the +83% throughput /
# -63% TTFT chunked-prefill optimization (config/chunked-prefill-tuned).
#
# This script restores submission_*/prepare_env.sh's SGLANG_SERVER_ARGS to
# the chunked-prefill 65K config (from commit 5a1480678, which was reverted
# in a4cbcfdc6 for v23 variable isolation). It uses git checkout of those
# specific files at the chunked-prefill commit, which preserves all OTHER
# fixes (gptqmodel pin, DIAGNOSTIC blocks, etc.) since they landed in
# commits AFTER 5a1480678.
#
# Idempotent + safe: --dry-run shows the planned diff, --revert undoes it.
#
# Usage:
#     # Default: apply to both v21+v22 source dirs, ready to pack v24
#     bash scripts/prep_v24_perf.sh
#
#     # Dry-run (show what would change)
#     bash scripts/prep_v24_perf.sh --dry-run
#
#     # Revert (back to v23's chunked-prefill 8192)
#     bash scripts/prep_v24_perf.sh --revert
#
#     # After applying, pack and submit:
#     #   python3 tools/pack_submission.py \
#     #       --variant submission_gptqmodel_calib_w4a16 \
#     #       --suffix _v24_perf --output-dir .

set -euo pipefail

CHUNKED_PREFILL_COMMIT="5a1480678"
PRE_CHUNKED_PREFILL_COMMIT="HEAD"  # current state on quant/w4a16 (chunked-prefill 8192)

VARIANTS=(
    "submission_gptqmodel_calib_w4a16"
    "submission_gptq_v17_minconfig"
)

DRY_RUN=0
REVERT=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --revert)  REVERT=1;  shift ;;
        -h|--help)
            grep -E '^# (Usage|    bash )' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *) echo "unknown: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Verify the target commit exists in history (paranoid check)
if ! git rev-parse --verify "$CHUNKED_PREFILL_COMMIT^{commit}" >/dev/null 2>&1; then
    echo "error: commit $CHUNKED_PREFILL_COMMIT not in history — run from quant/w4a16" >&2
    exit 2
fi

echo "===================================================================="
if [ "$REVERT" -eq 1 ]; then
    echo " prep_v24_perf.sh --revert"
    SOURCE_REF="HEAD"
    DESC="restore chunked-prefill 8192 (v23 state)"
else
    echo " prep_v24_perf.sh (apply chunked-prefill 65K)"
    SOURCE_REF="$CHUNKED_PREFILL_COMMIT"
    DESC="restore chunked-prefill 65K (v24-perf state)"
fi
echo " mode:    $DESC"
echo " dry-run: $DRY_RUN"
echo "===================================================================="
echo

for v in "${VARIANTS[@]}"; do
    f="$v/prepare_env.sh"
    if [ ! -f "$f" ]; then
        echo "  warn: $f missing; skipping"
        continue
    fi

    echo "  $f:"

    # What would change?
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    [dry-run] would: git checkout $SOURCE_REF -- $f"
        # Show diff preview
        diff_preview=$(git show "$SOURCE_REF:$f" | grep -E "chunked-prefill|max-prefill|mem-fraction" || true)
        if [ -n "$diff_preview" ]; then
            echo "    [dry-run] target file's chunked-prefill state:"
            echo "$diff_preview" | sed 's/^/      /'
        fi
    else
        if [ "$REVERT" -eq 1 ]; then
            # Restore to HEAD state (which is chunked-prefill 8192 since
            # a4cbcfdc6 reverted it)
            git checkout HEAD -- "$f"
        else
            # Get the version from chunked-prefill commit (5a1480678)
            git checkout "$CHUNKED_PREFILL_COMMIT" -- "$f"
        fi
        # Verify it changed
        if git diff --quiet HEAD -- "$f"; then
            echo "    (no change; file already matches target state)"
        else
            new_line=$(grep -E "chunked-prefill-size [0-9]+" "$f" | head -1 || echo "(no chunked-prefill line)")
            echo "    updated: $new_line"
        fi
    fi
done
echo

if [ "$DRY_RUN" -eq 0 ]; then
    echo "===================================================================="
    if [ "$REVERT" -eq 1 ]; then
        echo " REVERT DONE. submission_*/prepare_env.sh back to chunked-prefill 8192."
    else
        echo " APPLIED. submission_*/prepare_env.sh now has chunked-prefill 65K."
        echo
        echo " Next steps:"
        echo "   1. (optional) Re-quant + eval locally to confirm no regression:"
        echo "        bash scripts/local_eval.sh \\"
        echo "            --variant submission_gptqmodel_calib_w4a16 \\"
        echo "            --skip-quant \\"
        echo "            --eval-data submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \\"
        echo "            --num-samples 150"
        echo "   2. Preflight + pack v24:"
        echo "        python3 tools/pack_submission.py \\"
        echo "            --variant submission_gptqmodel_calib_w4a16 --check-only"
        echo "        python3 tools/pack_submission.py \\"
        echo "            --variant submission_gptqmodel_calib_w4a16 \\"
        echo "            --suffix _v24_perf --output-dir ."
        echo "   3. Submit. Expected: corrected_acc same as v23,"
        echo "      benchmark_duration drops 50%+."
        echo
        echo " To rollback: bash scripts/prep_v24_perf.sh --revert"
    fi
    echo "===================================================================="
fi
