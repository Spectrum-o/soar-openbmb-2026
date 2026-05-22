#!/usr/bin/env bash
# submission_w4a16_lightning_skip/prepare_model.sh
#
# Runs the 1849-style GPTQModel quant + post-processes with the
# lightning-skip overlay tool. The result is a mixed-precision artifact:
#   - Dense (minicpm4) layers' MLPs: W4A16 quantized
#   - Lightning attention layers' MLPs: BF16 (overlaid)
#
# Rationale: lightning attention's recurrent state h_t accumulates
# quantization errors across time steps. Long-context tasks (cwe / niah)
# suffered repetition collapse in v21-era runs. Skipping just the
# lightning-adjacent MLPs from quant (~75% of layers in SALA) should
# unblock those tasks at the cost of ~30% larger artifact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input)  INPUT_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *)        EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "Usage: bash prepare_model.sh --input <original> --output <processed> [quantize args...]" >&2
    exit 2
fi

# Calibration JSONL discovery (mirrors 1849)
BUNDLED_CALIB="${SCRIPT_DIR}/perf_public_set.jsonl"
AUTODL_CALIB="/root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl"

if [ -n "${CALIB_JSONL:-}" ] && [ -f "${CALIB_JSONL}" ]; then
    :
elif [ -f "${BUNDLED_CALIB}" ]; then
    CALIB_JSONL="${BUNDLED_CALIB}"
elif [ -f "${AUTODL_CALIB}" ]; then
    CALIB_JSONL="${AUTODL_CALIB}"
else
    CALIB_JSONL=""
fi

NUM_CALIB="${NUM_CALIB:-256}"
MAX_CALIB_LEN="${MAX_CALIB_LEN:-8192}"
CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-tail}"
DISABLE_CHAT_TEMPLATE="${DISABLE_CHAT_TEMPLATE:-0}"

CALIB_ARGS=()
if [ -n "${CALIB_JSONL}" ]; then
    CALIB_ARGS+=(--calib-jsonl "${CALIB_JSONL}"
                 --num-calib "${NUM_CALIB}"
                 --max-calib-len "${MAX_CALIB_LEN}"
                 --calib-window-mode "${CALIB_WINDOW_MODE}")
    if [ "${DISABLE_CHAT_TEMPLATE}" = "1" ]; then
        CALIB_ARGS+=(--no-chat-template)
    fi
    echo "[prepare_model] using calibration from: ${CALIB_JSONL}"
fi

echo "[prepare_model] DIAGNOSTIC: submission dir contents"
ls -la "${SCRIPT_DIR}" 2>&1 | head -25

echo "[prepare_model] DIAGNOSTIC: input dir"
ls -la "${INPUT_DIR}" 2>&1 | head -20

# ----------------------------------------------------------------------------
# Step 1: standard 1849-style GPTQ quant (MLP-only)
# ----------------------------------------------------------------------------
echo "[prepare_model] step 1: GPTQ quantization (1849-style MLP-only)..."
set +e
python3 "${SCRIPT_DIR}/quantize_gptqmodel_w4a16.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    "${CALIB_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
quant_exit=$?
set -e

if [ "${quant_exit}" -ne 0 ]; then
    echo "[prepare_model] FATAL: quantize_gptqmodel_w4a16.py exited ${quant_exit}" >&2
    exit "${quant_exit}"
fi

echo "[prepare_model] step 1 done. Artifact dir:"
ls -la "${OUTPUT_DIR}" 2>&1 | head -20

# ----------------------------------------------------------------------------
# Step 2: lightning-skip overlay (THE thing that makes this variant different)
# ----------------------------------------------------------------------------
OVERLAY_TOOL="${SCRIPT_DIR}/apply_lightning_skip_overlay.py"

if [ ! -f "${OVERLAY_TOOL}" ]; then
    echo "[prepare_model] FATAL: overlay tool missing at ${OVERLAY_TOOL}" >&2
    echo "[prepare_model]        check that pack_submission bundled apply_lightning_skip_overlay.py" >&2
    exit 1
fi

echo "[prepare_model] step 2: applying lightning-skip overlay"
python3 "${OVERLAY_TOOL}" \
    --quantized-dir "${OUTPUT_DIR}" \
    --bf16-dir "${INPUT_DIR}"

# ----------------------------------------------------------------------------
# Diagnostic: verify the overlay landed
# ----------------------------------------------------------------------------
echo "[prepare_model] DIAGNOSTIC: output dir after overlay"
ls -la "${OUTPUT_DIR}" 2>&1 | head -30

echo "[prepare_model] DIAGNOSTIC: overlay shard"
find "${OUTPUT_DIR}" -maxdepth 1 -name 'model-lightning-skip-overlay.safetensors' -exec ls -la {} \; 2>&1 || echo "  (no overlay shard found — was lightning-skip a no-op?)"

if [ -f "${OUTPUT_DIR}/quantize_config.json" ]; then
    echo "[prepare_model] DIAGNOSTIC: quantize_config.json (dynamic field)"
    python3 - <<PY
import json
with open("${OUTPUT_DIR}/quantize_config.json") as f:
    cfg = json.load(f)
dyn = cfg.get("dynamic", {})
n_lightning = sum(1 for k in dyn if "mlp" in k and "layers." in k)
print(f"  total dynamic skip rules: {len(dyn)}")
print(f"  lightning-MLP skip rules: {n_lightning}")
PY
fi

echo "[prepare_model] done"
