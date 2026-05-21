#!/usr/bin/env bash
# v24_auto.sh — one-shot wrapper for "v23 platform result → recommended v24 tarball".
#
# When you wake up and the v23 platform result is in, save the log to
# /root/autodl-fs/zyn/logs/platform_v23.log (or pass --log <path>),
# then run this script. It will:
#
#   1. Run scripts/decide_next_variant.py to apply V24_PLAN.md decision tree
#   2. Print the recommendation + rationale
#   3. Ask for confirmation (y/N)
#   4. If confirmed: invoke pack_submission.py --check-only then pack
#   5. Print the tarball path + reminder to upload
#
# Useful when half-asleep at 7am.
#
# Usage:
#     bash scripts/v24_auto.sh                          # use default log path
#     bash scripts/v24_auto.sh --log /path/to/v23.log   # explicit log
#     bash scripts/v24_auto.sh --yes                    # skip y/N confirm
#     bash scripts/v24_auto.sh --dry-run                # don't actually pack
#     bash scripts/v24_auto.sh --acc <number>           # override acc detection

set -euo pipefail

DEFAULT_LOG=/root/autodl-fs/zyn/logs/platform_v23.log
LOG_PATH=""
ACC_OVERRIDE=""
ASSUME_YES=0
DRY_RUN=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --log)      LOG_PATH="$2"; shift 2 ;;
        --acc)      ACC_OVERRIDE="$2"; shift 2 ;;
        --yes|-y)   ASSUME_YES=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)
            sed -n 's/^# \?//p' "$0" | head -25
            exit 0
            ;;
        *) echo "unknown: $1" >&2; exit 1 ;;
    esac
done

# Find the log
if [ -z "${LOG_PATH}" ]; then
    LOG_PATH="${DEFAULT_LOG}"
fi

if [ ! -f "${LOG_PATH}" ]; then
    echo "error: v23 platform log not found at ${LOG_PATH}" >&2
    echo
    echo "To save the platform log, copy the eval log from the SOAR web UI"
    echo "and save it as ${LOG_PATH}. Then re-run this script."
    echo
    echo "Or pass an explicit path: --log /your/path/v23.log"
    exit 2
fi

echo "===================================================================="
echo " v24_auto — v23 platform log → recommended v24 tarball"
echo "===================================================================="
echo "  log:         ${LOG_PATH}"
echo "  acc override: ${ACC_OVERRIDE:-(extract from log)}"
echo "  assume-yes:   ${ASSUME_YES}"
echo "  dry-run:      ${DRY_RUN}"
echo "===================================================================="
echo

# Step 1: decide
DECIDE_ARGS=(--log "${LOG_PATH}")
if [ -n "${ACC_OVERRIDE}" ]; then
    DECIDE_ARGS+=(--acc "${ACC_OVERRIDE}")
fi

DECIDE_OUTPUT=$(python3 "${REPO_ROOT}/scripts/decide_next_variant.py" "${DECIDE_ARGS[@]}")
echo "${DECIDE_OUTPUT}"
echo

# Step 2: parse the recommended variant_dir from the decide output.
# Match the `--variant <dir>` line specifically (decide_next_variant
# prints it in the "Ready-to-paste commands" block). Avoids false-positive
# matches on the log filename, which itself contains the
# "submission_gptqmodel_calib_w4a16_..." prefix.
VARIANT_DIR=$(echo "${DECIDE_OUTPUT}" \
    | grep -oP "(?<=--variant )submission_gptqmodel_calib_w4a16_\S+" \
    | head -1 || true)
if [ -z "${VARIANT_DIR}" ]; then
    echo "===================================================================="
    echo " No pre-built variant fits this case (MANUAL_FIX recommended)."
    echo " Review experiments/V23_PLATFORM_LOG_CHECKLIST.md and"
    echo " experiments/V24_PLAN.md for manual debug steps."
    echo "===================================================================="
    exit 0
fi

# Make sure it's a directory
if [ ! -d "${REPO_ROOT}/${VARIANT_DIR}" ]; then
    echo "===================================================================="
    echo " WARNING: recommended variant dir ${VARIANT_DIR} does not exist"
    echo " on disk. Build it manually following experiments/V24_PLAN.md."
    echo "===================================================================="
    exit 3
fi

echo "===================================================================="
echo " Recommended variant: ${VARIANT_DIR}"
echo "===================================================================="
echo

# Step 3: optional confirmation
if [ "${ASSUME_YES}" -eq 0 ]; then
    read -p "Proceed to preflight + pack? [y/N] " -n 1 -r
    echo
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "Aborted by user."
        exit 0
    fi
fi

# Step 4: preflight
echo
echo "[1/2] preflight check on ${VARIANT_DIR}"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  [dry-run] would run: python3 tools/pack_submission.py --variant ${VARIANT_DIR} --check-only"
else
    if ! python3 "${REPO_ROOT}/tools/pack_submission.py" --variant "${VARIANT_DIR}" --check-only; then
        echo
        echo "[1/2] PREFLIGHT FAILED. Fix issues above before packing." >&2
        exit 4
    fi
fi
echo

# Step 5: pack
# Variant suffix: drop the common prefix to get just the differentiator.
# e.g. submission_gptqmodel_calib_w4a16_v24_no_dtype_key → _v24_no_dtype_key
SUFFIX="_$(echo "${VARIANT_DIR}" | sed 's/^submission_gptqmodel_calib_w4a16_//')"
STAMP=$(date +%Y%m%d_%H%M%S)
TAR_PREFIX="soar_gptqmodel_calib_w4a16_mlp_only_submission_${STAMP}"

echo "[2/2] pack ${VARIANT_DIR} → ${TAR_PREFIX}${SUFFIX}.tar.gz"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  [dry-run] would run: python3 tools/pack_submission.py --variant ${VARIANT_DIR} --suffix ${SUFFIX} --output-dir ${REPO_ROOT}"
else
    cd "${REPO_ROOT}"
    python3 tools/pack_submission.py \
        --variant "${VARIANT_DIR}" \
        --suffix "${SUFFIX}" \
        --output-dir "${REPO_ROOT}"

    # Find the produced tarball
    TARBALL=$(ls -t "${REPO_ROOT}"/soar_*"${SUFFIX}".tar.gz 2>/dev/null | head -1)
    if [ -z "${TARBALL}" ]; then
        echo "warning: could not locate produced tarball" >&2
        exit 5
    fi

    echo
    echo "===================================================================="
    echo " DONE."
    echo "===================================================================="
    echo " Tarball: ${TARBALL}"
    echo " Size:    $(du -h "${TARBALL}" | cut -f1)"
    echo
    echo " Next: upload this tarball to the SOAR platform's submission UI."
    echo "       Platform 5h clock will start once the upload completes."
    echo "===================================================================="
fi
