#!/usr/bin/env bash
# submission_awq_lightning_skip/prepare_model.sh
#
# Combo variant: AWQ via llm-compressor + lightning-skip overlay.
# Stacks the two highest-EV bets from the night's research + analysis:
#
#   1. AWQ (per OpenBMB-ships-AWQ-for-MiniCPM-V signal + activation-outlier
#      research): expected +5-10pp acc over GPTQ on this architecture
#   2. Lightning-skip (per SALA's hybrid-attention recurrence hypothesis):
#      expected +5-10pp on cwe/niah specifically
#
# Combined target: +10-20pp over 1849's acc=58.61 → push toward 70-75 raw.
# Still below 80% correctness gate, but significantly closer.
#
# WARNING: this is the LEAST-tested variant. Each piece individually has
# only CPU-side validation. Their combination has zero end-to-end test.

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

# Calibration resolution (mirrors awq_llmcompressor variant)
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
AWQ_SCHEME="${AWQ_SCHEME:-W4A16_ASYM}"

# MLP_ONLY default 1 — this combo variant relies on AWQ for MLP quantization
# and lightning-skip for which-layers-to-quantize. Quantizing attention on
# top of that would compound risk.
MLP_ONLY_ARG=""
if [ "${MLP_ONLY:-1}" = "1" ]; then
    MLP_ONLY_ARG="--mlp-only"
fi

echo "[prepare_model] DIAGNOSTIC: submission dir contents"
ls -la "${SCRIPT_DIR}"

echo "[prepare_model] DIAGNOSTIC: input dir"
ls -la "${INPUT_DIR}" | head -20

# ----------------------------------------------------------------------------
# Step 1: AWQ via llm-compressor (writes compressed-tensors format)
# ----------------------------------------------------------------------------
AWQ_SCRIPT="${SCRIPT_DIR}/quantize_llmcompressor_awq.py"

CMD=(python3 "${AWQ_SCRIPT}"
     --input "${INPUT_DIR}"
     --output "${OUTPUT_DIR}"
     --num-calib "${NUM_CALIB}"
     --max-calib-len "${MAX_CALIB_LEN}"
     --scheme "${AWQ_SCHEME}")

if [ -n "${CALIB_JSONL}" ]; then
    CMD+=(--calib-jsonl "${CALIB_JSONL}")
fi
if [ -n "${MLP_ONLY_ARG}" ]; then
    CMD+=("${MLP_ONLY_ARG}")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "[prepare_model] step 1: AWQ quantization at $(date '+%F %T')"
echo "[prepare_model] cmd: ${CMD[*]}"
"${CMD[@]}"
echo "[prepare_model] step 1 done at $(date '+%F %T')"

# ----------------------------------------------------------------------------
# Step 2: lightning-skip overlay (handles compressed-tensors format)
# ----------------------------------------------------------------------------
OVERLAY_TOOL="${SCRIPT_DIR}/apply_lightning_skip_overlay.py"

if [ ! -f "${OVERLAY_TOOL}" ]; then
    echo "[prepare_model] FATAL: overlay tool missing at ${OVERLAY_TOOL}" >&2
    exit 1
fi

echo "[prepare_model] step 2: applying lightning-skip overlay"
python3 "${OVERLAY_TOOL}" \
    --quantized-dir "${OUTPUT_DIR}" \
    --bf16-dir "${INPUT_DIR}"

# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
echo "[prepare_model] DIAGNOSTIC: output dir after overlay"
ls -la "${OUTPUT_DIR}" 2>&1 | head -30

if [ -f "${OUTPUT_DIR}/model-lightning-skip-overlay.safetensors" ]; then
    echo "[prepare_model] DIAGNOSTIC: overlay shard present"
    ls -la "${OUTPUT_DIR}/model-lightning-skip-overlay.safetensors"
fi

if [ -f "${OUTPUT_DIR}/quantize_config.json" ]; then
    echo "[prepare_model] DIAGNOSTIC: quantize_config.json (dynamic skip rules)"
    python3 - <<PY
import json
with open("${OUTPUT_DIR}/quantize_config.json") as f:
    cfg = json.load(f)
dyn = cfg.get("dynamic", {})
n_lightning = sum(1 for k in dyn if "mlp" in k and "layers." in k)
print(f"  total dynamic skip rules: {len(dyn)}")
print(f"  lightning-MLP skip rules: {n_lightning}")
print(f"  quant_method:             {cfg.get('quant_method', cfg.get('config_groups', {}).get('group_0', {}).get('quant_method', '?'))}")
PY
fi

echo "[prepare_model] done"
