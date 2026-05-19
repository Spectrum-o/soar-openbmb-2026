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

# Safety: hard cap the quantize wall time so we fail cleanly INSIDE the
# 5h platform budget if anything hangs.
#   - Expected normal time on RTX PRO 6000 (96GB): 35-60 min, comprising:
#       JIT compile of GPTQModel CUDA kernels (first run):    10-15 min
#       BF16 model load:                                       2 min
#       Calibration forward (150 samples × 4096 tokens):       5-10 min
#       Per-layer GPTQ optimization × 32 layers:               16-32 min
#       Save quantized weights (~5GB):                         2-3 min
#   - Total budget breakdown for a successful run:
#       prepare_env:   ~ 5 min
#       quantize:      ~50 min (typical, see above)
#       bench S1+8+max:~95 min (matches the 2026-05-19 RTN runs)
#       eval_model.py: ~15 min
#       TOTAL:        ~165 min ≈ 2.75 h  (well under 5h)
#   - We abort at 75 min. Past that, prepare_model.sh exits 124 and the
#     platform marks the submission as "failed midway" (does NOT consume
#     a slot per user's reported rule), well before the 5h hard timeout
#     (which WOULD consume the slot).
QUANT_TIMEOUT_MIN="${QUANT_TIMEOUT_MIN:-75}"

# Disk offload during quantization is slow on this workload — RTX PRO
# 6000 has 96GB VRAM, which fits the 18GB BF16 model + Hessian
# workspace comfortably. Disabling disk offload prevents thrashing.
# --no-offload-disk passes through to the quantize script.
EXTRA_ARGS+=(--no-offload-disk)

echo "[prepare_model] quantize timeout: ${QUANT_TIMEOUT_MIN} min"

timeout "${QUANT_TIMEOUT_MIN}m" python3 "${SCRIPT_DIR}/quantize_gptqmodel_w4a16.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    "${CALIB_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
quant_exit=$?

if [ "${quant_exit}" -eq 124 ]; then
    echo "[prepare_model] FATAL: quantization exceeded ${QUANT_TIMEOUT_MIN} min wall time; aborting cleanly" >&2
    exit 124
elif [ "${quant_exit}" -ne 0 ]; then
    echo "[prepare_model] quantize exited ${quant_exit}" >&2
    exit "${quant_exit}"
fi
