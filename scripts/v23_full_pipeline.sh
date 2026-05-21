#!/usr/bin/env bash
# v23 full pipeline — one-shot wrapper from source-check to packed tarball.
#
# This script automates the v23 packaging flow:
#   1. Preflight: run pack_submission --check-only on the variant source
#      to ensure all 7 fix canaries (qzeros hardened / H4 / gptqmodel pin /
#      DIAGNOSTIC / chunked-prefill 8K / etc.) are present.
#   2. Inspect existing local quant artifact (if present) via
#      inspect_quant_artifact.py to confirm qzeros are 0x88888888.
#   3. Run check_tokenizer_compat.py against base BF16 model to confirm
#      no tokenizer drift between the artifact and the source-of-truth
#      tokenizer files.
#   4. Optionally overwrite the existing artifact's tokenizer files with
#      base BF16 originals (the H4 fix as applied retroactively).
#   5. (Skipped by default) re-quantize via local_eval.sh; pass --requant
#      to do so.
#   6. Pack v23 tarball via pack_submission.py.
#   7. Run final preflight on the tarball.
#
# All file/dir paths default to the user's AutoDL setup. Override via
# flags or environment.
#
# Usage:
#     # Full pipeline (default: skip re-quant, use existing artifact)
#     bash scripts/v23_full_pipeline.sh
#
#     # With re-quantization (~25 min on GPU)
#     bash scripts/v23_full_pipeline.sh --requant
#
#     # Dry-run: print what would happen
#     bash scripts/v23_full_pipeline.sh --dry-run
#
#     # Use a different variant / base path
#     bash scripts/v23_full_pipeline.sh \
#         --variant submission_gptq_v17_minconfig \
#         --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA
#
# Flags:
#     --variant <dir>     Source variant directory (default: submission_gptqmodel_calib_w4a16)
#     --base <dir>        Base BF16 model dir (default: /root/autodl-fs/models/OpenBMB/MiniCPM-SALA)
#     --artifact <dir>    Existing quant artifact (default: /root/autodl-fs/zyn/models/<variant>-quantized)
#     --suffix <s>        Tarball suffix tag (default: _v23)
#     --requant           Run local_eval.sh --force-requant before packing
#     --no-tokenizer-overwrite  Skip step 4 (use the artifact's tokenizer as-is)
#     --dry-run           Print actions; do not execute
#     -h, --help          Show this help

set -euo pipefail

VARIANT="submission_gptqmodel_calib_w4a16"
BASE="/root/autodl-fs/models/OpenBMB/MiniCPM-SALA"
ARTIFACT=""
SUFFIX="_v23"
REQUANT=0
SKIP_OVERWRITE=0
DRY_RUN=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    grep -E '^# (Usage|Flags|    |bash )' "$0" | sed 's/^# //; s/^#//'
    exit "${1:-0}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --variant)               VARIANT="$2";  shift 2 ;;
        --base)                  BASE="$2";     shift 2 ;;
        --artifact)              ARTIFACT="$2"; shift 2 ;;
        --suffix)                SUFFIX="$2";   shift 2 ;;
        --requant)               REQUANT=1;     shift ;;
        --no-tokenizer-overwrite) SKIP_OVERWRITE=1; shift ;;
        --dry-run)               DRY_RUN=1;     shift ;;
        -h|--help)               usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

if [ -z "$ARTIFACT" ]; then
    ARTIFACT="/root/autodl-fs/zyn/models/${VARIANT}-quantized"
fi

VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
if [ ! -d "$VARIANT_DIR" ]; then
    echo "error: variant dir not found: $VARIANT_DIR" >&2
    exit 2
fi

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] $*"
    else
        echo ">>> $*"
        "$@"
    fi
}

echo "===================================================================="
echo " v23 full pipeline"
echo "===================================================================="
echo "  variant:   $VARIANT_DIR"
echo "  base:      $BASE"
echo "  artifact:  $ARTIFACT"
echo "  suffix:    $SUFFIX"
echo "  requant:   $REQUANT"
echo "  dry-run:   $DRY_RUN"
echo "===================================================================="
echo

# ---------- Step 1: Source preflight (no GPU) ----------
echo "[1/7] preflight: pack_submission --check-only"
run python3 "${REPO_ROOT}/tools/pack_submission.py" --variant "$VARIANT" --check-only
echo

# ---------- Step 2: Inspect existing artifact (if present) ----------
echo "[2/7] inspect_quant_artifact (if artifact exists)"
if [ -d "$ARTIFACT" ]; then
    run python3 "${REPO_ROOT}/tools/inspect_quant_artifact.py" \
        --artifact "$ARTIFACT" \
        --sample-tensors 2 \
        --output-md "/tmp/v23_pipeline_inspect_artifact.md"
    echo "    full report at /tmp/v23_pipeline_inspect_artifact.md"
else
    echo "    artifact $ARTIFACT does not exist; will be created by step 5 if --requant"
    if [ "$REQUANT" -eq 0 ]; then
        echo "    error: cannot pack without an artifact AND --requant flag is off"
        if [ "$DRY_RUN" -eq 0 ]; then
            exit 3
        fi
    fi
fi
echo

# ---------- Step 3: tokenizer compat check ----------
echo "[3/7] check_tokenizer_compat (base vs artifact)"
if [ -d "$ARTIFACT" ] && [ -d "$BASE" ]; then
    set +e
    run python3 "${REPO_ROOT}/tools/check_tokenizer_compat.py" \
        --base "$BASE" \
        --artifact "$ARTIFACT" \
        --output-md "/tmp/v23_pipeline_tokenizer_compat.md"
    compat_exit=$?
    set -e
    if [ "$compat_exit" -ne 0 ] && [ "$DRY_RUN" -eq 0 ]; then
        echo "    drift detected (exit $compat_exit) — step 4 will fix if --no-tokenizer-overwrite not set"
    fi
elif [ ! -d "$BASE" ]; then
    echo "    base $BASE not present; skipping compat check"
fi
echo

# ---------- Step 4: Tokenizer overwrite ----------
echo "[4/7] tokenizer overwrite (apply H4 fix to existing artifact)"
if [ "$SKIP_OVERWRITE" -eq 1 ]; then
    echo "    skipped via --no-tokenizer-overwrite"
elif [ ! -d "$ARTIFACT" ]; then
    echo "    artifact missing; step 5 (--requant) will produce a fresh one with H4 fix baked in"
elif [ ! -d "$BASE" ]; then
    echo "    base $BASE not present; cannot overwrite"
else
    run bash "${REPO_ROOT}/scripts/overwrite_tokenizer_with_base.sh" \
        --artifact "$ARTIFACT" \
        --base "$BASE"
fi
echo

# ---------- Step 5: Re-quantize (optional) ----------
echo "[5/7] re-quantize (optional)"
if [ "$REQUANT" -eq 1 ]; then
    echo "    invoking local_eval.sh with --force-requant + --eval-only-after"
    EVAL_DATA="${VARIANT_DIR}/perf_public_set.jsonl"
    if [ -f "$EVAL_DATA" ]; then
        run bash "${REPO_ROOT}/scripts/local_eval.sh" \
            --variant "$VARIANT" \
            --force-requant \
            --eval-data "$EVAL_DATA" \
            --num-samples 150
    else
        echo "    warning: $EVAL_DATA not found; running quant without eval"
        run bash "${VARIANT_DIR}/prepare_model.sh" \
            --input "$BASE" \
            --output "$ARTIFACT"
    fi
else
    echo "    skipped (--requant not given); pack will use the existing artifact"
fi
echo

# ---------- Step 6: Pack v23 tarball ----------
echo "[6/7] pack v23 tarball"
run python3 "${REPO_ROOT}/tools/pack_submission.py" \
    --variant "$VARIANT" \
    --suffix "$SUFFIX" \
    --output-dir "${REPO_ROOT}"
echo

# ---------- Step 7: Final preflight on the produced tarball ----------
echo "[7/7] final preflight on the tarball"
if [ "$DRY_RUN" -eq 0 ]; then
    TARBALL=$(ls -t "${REPO_ROOT}"/*${SUFFIX}.tar.gz 2>/dev/null | head -1)
    if [ -z "$TARBALL" ]; then
        echo "    error: could not find packed tarball with suffix $SUFFIX" >&2
        exit 4
    fi
    echo "    tarball: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
    if [ -x "${REPO_ROOT}/tools/quant_config_validator.py" ]; then
        run python3 "${REPO_ROOT}/tools/quant_config_validator.py" --tarball "$TARBALL"
    fi
    echo
    echo "===================================================================="
    echo " v23 PIPELINE DONE"
    echo "===================================================================="
    echo " Tarball: $TARBALL"
    echo " Submit to platform when ready. After it runs:"
    echo "   python3 tools/parse_quant_diagnostic.py \\"
    echo "     --input <platform.log> \\"
    echo "     --compare scripts/logs/local_eval_quant_submission_gptqmodel_calib_w4a16_1779289794.log \\"
    echo "     --output-md /root/autodl-fs/zyn/logs/diff_v23.md"
    echo " Then walk experiments/V23_PLATFORM_LOG_CHECKLIST.md."
else
    echo "[dry-run] would search for *${SUFFIX}.tar.gz and validate it"
fi
