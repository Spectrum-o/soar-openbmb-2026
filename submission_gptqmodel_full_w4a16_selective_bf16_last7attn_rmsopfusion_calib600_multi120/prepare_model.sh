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
NUM_CALIB="${NUM_CALIB:-600}"
# max_calib_len 8192: tokenize_calibration uses truncation_side="left",
# so the kept window is the TAIL of each prompt — the question/answer
# structure that v17's right-truncation cut off (median ~30K, p90 ~117K
# tokens, with the actual task at the end of the row). ~2x the Hessian
# compute per sample vs 4096.
MAX_CALIB_LEN="${MAX_CALIB_LEN:-8192}"

# Calibration window mode: see quantize_gptqmodel_w4a16.py --calib-window-mode.
#   tail            -> single tail window per prompt (default; v21/v22 behavior)
#   multi-adaptive  -> 1-3 windows per prompt based on full token length
CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"

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
#   - Expected normal time on RTX PRO 6000 (96GB): 75-105 min for this
#     NUM_CALIB=600 + multi-adaptive probe on the SOAR platform's
#     gptqmodel 7.0 + torch 2.9.1, because cpp
#     extensions require torch>=2.11 (verified from 20:30 submission log)
#     so GPTQ falls back to pure-Python path (~30% slower than cpp).
#   - We abort at 120 min. Past that, prepare_model.sh exits 124 and the
#     platform marks the submission as "failed midway" (does NOT consume
#     a slot per user's reported rule), well before the 5h hard timeout
#     (which WOULD consume the slot).
#   - Total budget on a successful run:
#       prepare_env:   ~ 5 min
#       quantize:      ~75-105 min (heavier multi-adaptive calibration)
#       bench S1+8+max:~95 min (matches the 2026-05-19 RTN runs)
#       eval_model.py: ~15 min
#       TOTAL:        ~190-220 min ≈ 3.2-3.7 h  (still under 5h)
QUANT_TIMEOUT_MIN="${QUANT_TIMEOUT_MIN:-120}"

# Disk offload during quantization is slow on this workload — RTX PRO
# 6000 has 96GB VRAM, which fits the 18GB BF16 model + Hessian
# workspace comfortably. Disabling disk offload prevents thrashing.
# --no-offload-disk passes through to the quantize script.
EXTRA_ARGS+=(--no-offload-disk --group-size "${GROUP_SIZE:-128}")

# FULL W4A16 variant: quantizes attention/Lightning q/k/v/o plus MLP.
# Start with group-size 128 for speed and lower scale overhead. Override
# GROUP_SIZE=64 for a slower, potentially higher-quality follow-up run.

echo "[prepare_model] FULL W4A16 quantize timeout: ${QUANT_TIMEOUT_MIN} min"

# Diagnostic snapshot BEFORE quantize so platform logs show what we have.
# v21/v22 both scored 0 on the platform with no remote signal of why;
# these prints (script dir contents, calib file size + head row,
# input/output dir existence) close that gap so the next failure mode
# is visible from the eval log alone.
echo "[prepare_model] DIAGNOSTIC: script dir contents"
ls -la "${SCRIPT_DIR}" 2>&1 | head -40 || true
echo "[prepare_model] DIAGNOSTIC: input dir"
ls -la "${INPUT_DIR}" 2>&1 | head -20 || true
if [ -n "${CALIB_JSONL}" ] && [ -f "${CALIB_JSONL}" ]; then
    echo "[prepare_model] DIAGNOSTIC: calib jsonl path=${CALIB_JSONL}"
    echo "[prepare_model] DIAGNOSTIC: calib jsonl size=$(stat -c%s "${CALIB_JSONL}" 2>/dev/null || stat -f%z "${CALIB_JSONL}") bytes"
    echo "[prepare_model] DIAGNOSTIC: calib jsonl row count=$(wc -l < "${CALIB_JSONL}")"
    echo "[prepare_model] DIAGNOSTIC: calib jsonl first row (first 200 chars):"
    head -1 "${CALIB_JSONL}" | head -c 200 || true
    echo ""
fi

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

for arg in "${EXTRA_ARGS[@]}"; do
    if [ "${arg}" = "--dry-run" ]; then
        echo "[prepare_model] quantize dry-run completed; skipping selective overlay because no artifact was written"
        exit 0
    fi
done

# Diagnostic snapshot AFTER quantize so platform logs show the artifact
# layout (single-file vs sharded, sizes, presence of qzeros via grep).
# Independent of any Python introspection — if the Python step crashed
# partway, this still produces output.
echo "[prepare_model] DIAGNOSTIC: output dir contents after quantize"
ls -la "${OUTPUT_DIR}" 2>&1 | head -30 || true
echo "[prepare_model] DIAGNOSTIC: safetensors files in output"
find "${OUTPUT_DIR}" -maxdepth 1 -name '*.safetensors' -exec ls -la {} \; 2>&1 || true
if [ -f "${OUTPUT_DIR}/quantize_config.json" ]; then
    echo "[prepare_model] DIAGNOSTIC: quantize_config.json"
    cat "${OUTPUT_DIR}/quantize_config.json"
fi

# ----------------------------------------------------------------------------
# Step 2 (variant addition): selective BF16 recovery on top of full-W4A16.
#
# 2026-05-26 full-W4A16 platform:
#   acc_ori=78.27, final_score=22.85, S1=621.91, S8=1002.06, Smax=2315.78
# It proved the speed win is real but lost ~2pp raw acc vs the 80 gate.
#
# The extra risk introduced by full-W4A16 vs the stable MLP-only package is
# quantized attention projections. This follow-up restores one fewer lightning
# layer than the platform-proven last8-attn package, aiming to keep acc above
# 80 while recovering some full-W4A16 speed.
# ----------------------------------------------------------------------------
OVERLAY_TOOL="${SCRIPT_DIR}/apply_lightning_skip_overlay.py"
FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last7-lightning}"
FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"
FULL_SELECTIVE_OVERLAY_NAME="${FULL_SELECTIVE_OVERLAY_NAME:-model-full-selective-bf16-last7-attn-overlay.safetensors}"

if [ ! -f "${OVERLAY_TOOL}" ]; then
    echo "[prepare_model] FATAL: selective overlay tool missing at ${OVERLAY_TOOL}" >&2
    exit 1
fi

echo "[prepare_model] step 2: applying selective BF16 overlay on full-W4A16 artifact"
echo "[prepare_model] selective layers=${FULL_SELECTIVE_LAYERS} modules=${FULL_SELECTIVE_MODULES}"
python3 "${OVERLAY_TOOL}" \
    --quantized-dir "${OUTPUT_DIR}" \
    --bf16-dir "${INPUT_DIR}" \
    --layers "${FULL_SELECTIVE_LAYERS}" \
    --modules "${FULL_SELECTIVE_MODULES}" \
    --overlay-name "${FULL_SELECTIVE_OVERLAY_NAME}"

echo "[prepare_model] DIAGNOSTIC: output dir after selective overlay"
ls -la "${OUTPUT_DIR}" 2>&1 | head -30
echo "[prepare_model] DIAGNOSTIC: selective overlay shard"
find "${OUTPUT_DIR}" -maxdepth 1 -name "${FULL_SELECTIVE_OVERLAY_NAME}" -exec ls -la {} \; 2>&1 \
    || echo "  (no selective overlay shard found)"

if [ -f "${OUTPUT_DIR}/quantize_config.json" ]; then
    echo "[prepare_model] DIAGNOSTIC: quantize_config.json selective dynamic rules"
    python3 - "${OUTPUT_DIR}" <<'PY'
import json, sys
out_dir = sys.argv[1]
with open(f"{out_dir}/quantize_config.json") as f:
    cfg = json.load(f)
dyn = cfg.get("dynamic", {})
selected = [k for k in dyn if "model.layers." in k and ("qkv_proj" in k or "o_proj" in k or "mlp" in k)]
print(f"  total dynamic skip rules: {len(dyn)}")
print(f"  selected layer skip rules: {len(selected)}")
for rule in selected[:20]:
    print(f"    {rule}")
PY
fi
