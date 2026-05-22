#!/usr/bin/env bash
# submission_awq_llmcompressor/prepare_model.sh
#
# Runs the AWQ-via-llm-compressor quantization. The script itself
# (quantize_llmcompressor_awq.py) handles all Hard Constraints post-processing:
#   - copy_runtime_assets: overwrites tokenizer from base BF16 (H4 fix)
#   - write_sglang_compatible_config: strips auto_map.AutoConfig (row 44)
#     + strips derived read-only properties (row 45)
#     + adds compressed-tensors quantization_config block

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

# Calibration resolution order (mirrors 1849's prepare_model.sh):
#   1. $CALIB_JSONL env var (if set and file exists)
#   2. bundled perf_public_set.jsonl
#   3. /root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl (AutoDL fallback)
BUNDLED_CALIB="${SCRIPT_DIR}/perf_public_set.jsonl"
AUTODL_CALIB="/root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl"

if [ -n "${CALIB_JSONL:-}" ] && [ -f "${CALIB_JSONL}" ]; then
    :   # user override
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

# MLP-only flag (mirrors 1849's hybrid-attention safety). Set MLP_ONLY=0 to
# quantize everything (full path).
MLP_ONLY_ARG=""
if [ "${MLP_ONLY:-1}" = "1" ]; then
    MLP_ONLY_ARG="--mlp-only"
fi

# Diagnostic dumps (mirrors 1849 prepare_model.sh)
echo "[prepare_model] DIAGNOSTIC: submission dir contents"
ls -la "${SCRIPT_DIR}"

echo "[prepare_model] DIAGNOSTIC: input dir"
ls -la "${INPUT_DIR}" | head -30

if [ -n "${CALIB_JSONL}" ]; then
    echo "[prepare_model] DIAGNOSTIC: calib jsonl path=${CALIB_JSONL}"
    echo "[prepare_model] DIAGNOSTIC: calib jsonl size=$(stat -c%s "${CALIB_JSONL}") bytes"
    echo "[prepare_model] DIAGNOSTIC: calib jsonl row count=$(wc -l < "${CALIB_JSONL}")"
    echo "[prepare_model] DIAGNOSTIC: calib jsonl first row (first 200 chars):"
    head -1 "${CALIB_JSONL}" | head -c 200
    echo
else
    echo "[prepare_model] WARNING: no calibration jsonl found" >&2
fi

# ----------------------------------------------------------------------------
# Run AWQ via llmcompressor
# ----------------------------------------------------------------------------
QUANT_SCRIPT="${SCRIPT_DIR}/quantize_llmcompressor_awq.py"

if [ ! -f "${QUANT_SCRIPT}" ]; then
    echo "[prepare_model] FATAL: ${QUANT_SCRIPT} not found" >&2
    exit 1
fi

CMD=(python3 "${QUANT_SCRIPT}"
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

echo "[prepare_model] starting AWQ quantization at $(date '+%F %T')"
echo "[prepare_model] cmd: ${CMD[*]}"
"${CMD[@]}"
echo "[prepare_model] AWQ quantization done at $(date '+%F %T')"

# Final sanity: verify output config.json has the Hard Constraints applied
if [ -f "${OUTPUT_DIR}/config.json" ]; then
    echo "[prepare_model] DIAGNOSTIC: output config.json (Hard Constraints check):"
    python3 - <<PY
import json, sys
with open("${OUTPUT_DIR}/config.json") as f:
    cfg = json.load(f)
auto_map = cfg.get("auto_map", {})
print(f"  auto_map.AutoConfig present: {'AutoConfig' in auto_map}")
print(f"  auto_map.AutoModel*  present: {[k for k in auto_map if 'AutoModel' in k]}")
print(f"  has_sparse_attention present: {'has_sparse_attention' in cfg}")
print(f"  quantization_config present:  {bool(cfg.get('quantization_config'))}")
if cfg.get('quantization_config'):
    print(f"    quant_method:           {cfg['quantization_config'].get('quant_method')}")
PY
fi

echo "[prepare_model] done"
