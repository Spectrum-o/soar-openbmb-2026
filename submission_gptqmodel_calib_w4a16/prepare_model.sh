#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "Usage: bash prepare_model.sh --input <original> --output <processed> [quantize args...]" >&2
    exit 2
fi

# Calibration data resolution order (CRITICAL — getting this wrong was the
# root cause of the 2026-05-19 acc=42 submissions):
#   1. $CALIB_JSONL env var (if set and file exists)
#   2. perf_public_set.jsonl bundled inside this tarball  ← default on platform
#   3. /root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl  ← AutoDL convention
#   4. synthetic fallback (last resort, LOW quality, will likely fail eval)
BUNDLED_CALIB="${SCRIPT_DIR}/perf_public_set.jsonl"
AUTODL_CALIB="/root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl"

if [ -n "${CALIB_JSONL:-}" ] && [ -f "${CALIB_JSONL}" ]; then
    : # user override
elif [ -f "${BUNDLED_CALIB}" ]; then
    CALIB_JSONL="${BUNDLED_CALIB}"
elif [ -f "${AUTODL_CALIB}" ]; then
    CALIB_JSONL="${AUTODL_CALIB}"
else
    CALIB_JSONL=""
fi

# 150 rows in perf_public_set.jsonl (30 each across mcq / niah / qa / fwe / cwe).
# Use all of them by default; no random sampling needed at this size.
NUM_CALIB="${NUM_CALIB:-150}"
# max_calib_len 4096: balances coverage with quantization time. Most rows are
# much longer than 4K tokens (median ~30K, p90 ~117K) — we sample the prefix
# for activation distribution.
MAX_CALIB_LEN="${MAX_CALIB_LEN:-4096}"

CALIB_ARGS=()
if [ -n "${CALIB_JSONL}" ]; then
    CALIB_ARGS+=(--calib-jsonl "${CALIB_JSONL}")
    echo "[prepare_model] using calibration from: ${CALIB_JSONL}"
else
    echo "[prepare_model] WARNING: no calibration file found — synthetic fallback will likely fail correctness gate" >&2
fi
CALIB_ARGS+=(--num-calib "${NUM_CALIB}" --max-calib-len "${MAX_CALIB_LEN}")

python3 "${SCRIPT_DIR}/quantize_gptqmodel_w4a16.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    "${CALIB_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
