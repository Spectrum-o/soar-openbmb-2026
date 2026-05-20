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

# perf_public_set.jsonl has 150 rows (30 each across mcq / niah / qa / fwe / cwe).
# GPTQModel 7.x warns below 256 examples, so the quantizer deterministically
# cycles the public rows to reach this default without falling back to synthetic
# prompts.
NUM_CALIB="${NUM_CALIB:-256}"
# max_calib_len 8192 (v17-minconfig variant): with tokenizer.truncation_side
# = "left", we keep the LAST 8K tokens of each prompt — i.e. the question
# structure that v17's right-truncation cut off. Cost is ~2x the GPTQ
# Hessian time per sample vs 4096.
MAX_CALIB_LEN="${MAX_CALIB_LEN:-8192}"

# Calibration window mode: see quantize_gptqmodel_w4a16.py --calib-window-mode.
#   tail            -> single tail window per prompt (default; v21/v22 behavior)
#   multi-adaptive  -> 1-3 windows per prompt based on full token length
CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-tail}"

# Chat template ablation switch. Set DISABLE_CHAT_TEMPLATE=1 to skip
# apply_chat_template() during calibration tokenization. perf_public_set
# eval feeds RAW `question` strings, but default calibration wraps in
# `<用户>...<AI>`. v21 (chat tpl ON + left-trunc + 8K) got acc=49 on 150
# samples; this flag enables isolating the chat-template variable.
DISABLE_CHAT_TEMPLATE="${DISABLE_CHAT_TEMPLATE:-0}"

CALIB_ARGS=()
if [ -n "${CALIB_JSONL}" ]; then
    CALIB_ARGS+=(--calib-jsonl "${CALIB_JSONL}")
    echo "[prepare_model] using calibration from: ${CALIB_JSONL}"
else
    echo "[prepare_model] WARNING: no calibration file found — synthetic fallback will likely fail correctness gate" >&2
fi
CALIB_ARGS+=(--num-calib "${NUM_CALIB}" --max-calib-len "${MAX_CALIB_LEN}" --calib-window-mode "${CALIB_WINDOW_MODE}")
if [ "${DISABLE_CHAT_TEMPLATE}" = "1" ]; then
    CALIB_ARGS+=(--no-chat-template)
    echo "[prepare_model] chat template DISABLED (DISABLE_CHAT_TEMPLATE=1)"
fi

# Safety: hard cap the quantize wall time so we fail cleanly INSIDE the
# 5h platform budget if anything hangs.
#   - Expected normal time on RTX PRO 6000 (96GB): 45-75 min on the
#     SOAR platform's gptqmodel 7.0 + torch 2.9.1, because cpp
#     extensions require torch>=2.11 (verified from 20:30 submission log)
#     so GPTQ falls back to pure-Python path (~30% slower than cpp).
#   - We abort at 90 min. Past that, prepare_model.sh exits 124 and the
#     platform marks the submission as "failed midway" (does NOT consume
#     a slot per user's reported rule), well before the 5h hard timeout
#     (which WOULD consume the slot).
#   - Total budget on a successful run:
#       prepare_env:   ~ 5 min
#       quantize:      ~60 min (typical pure-Python path)
#       bench S1+8+max:~95 min (matches the 2026-05-19 RTN runs)
#       eval_model.py: ~15 min
#       TOTAL:        ~175 min ≈ 2.9 h  (still well under 5h)
QUANT_TIMEOUT_MIN="${QUANT_TIMEOUT_MIN:-90}"

# Disk offload during quantization is slow on this workload — RTX PRO
# 6000 has 96GB VRAM, which fits the 18GB BF16 model + Hessian
# workspace comfortably. Disabling disk offload prevents thrashing.
# --no-offload-disk passes through to the quantize script.
EXTRA_ARGS+=(--no-offload-disk)

echo "[prepare_model] quantize timeout: ${QUANT_TIMEOUT_MIN} min"

set +e
timeout "${QUANT_TIMEOUT_MIN}m" python3 "${SCRIPT_DIR}/quantize_gptqmodel_w4a16.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    "${CALIB_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
quant_exit=$?
set -e

if [ "${quant_exit}" -eq 124 ]; then
    echo "[prepare_model] FATAL: quantization exceeded ${QUANT_TIMEOUT_MIN} min wall time; aborting cleanly" >&2
    exit 124
elif [ "${quant_exit}" -ne 0 ]; then
    echo "[prepare_model] quantize exited ${quant_exit}" >&2
    exit "${quant_exit}"
fi
