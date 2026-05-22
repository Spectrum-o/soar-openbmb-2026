#!/usr/bin/env bash
# scripts/test_llmcompressor_artifact.sh
#
# Test the 2026-05-14 llm-compressor W4A16 artifact at
# /root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/.
#
# Memory `project_may14_llmcompressor_artifact.md` documents this artifact:
#   - Producer: llmcompressor (NOT GPTQModel)
#   - Format: compressed-tensors / pack-quantized
#   - Loader: --quantization compressed-tensors (NOT gptq_marlin)
#   - dtype: bfloat16 (NOT float16)
#   - actorder: static (desc_act=True) — Marlin can't use this, but compressed-tensors can
#   - acc: UNKNOWN — never benchmarked locally or on platform
#
# WHY THIS MATTERS:
#   Our entire GPTQModel path (v17 ... v23b ... 1849) hits acc=0 on platform.
#   RTN (numpy direct, bypasses transformers) hits 42. Hypothesis: GPTQModel
#   tool itself has SALA-incompatible behavior we haven't isolated. If this
#   llm-compressor artifact serves and scores > 0 locally, we have evidence
#   that switching tools is the way out of the GPTQModel pipeline trap.
#
# USAGE (on AutoDL):
#   bash scripts/test_llmcompressor_artifact.sh
#   bash scripts/test_llmcompressor_artifact.sh --num-samples 50  # quick check
#   bash scripts/test_llmcompressor_artifact.sh --serve-only      # just launch, no eval
#
# OUTPUT:
#   - Inspects artifact (recipe.yaml + config.json + qweight sample)
#   - Launches SGLang server with compressed-tensors loader
#   - Runs eval_model.py against perf_public_set.jsonl
#   - Prints acc_ori + per-task breakdown
#   - Logs to scripts/logs/llmcompressor_eval_<timestamp>.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_PATH="${ARTIFACT_PATH:-/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/sglang_minicpm_sala_env}"
TOOLKIT_DIR="${TOOLKIT_DIR:-/root/autodl-fs/zyn/soar_toolkit}"
EVAL_SCRIPT="${TOOLKIT_DIR}/eval_model.py"
EVAL_DATA="${TOOLKIT_DIR}/perf_public_set.jsonl"
PORT="${PORT:-31112}"   # different from default 31111 to avoid clashing
NUM_SAMPLES="${NUM_SAMPLES:-150}"
CONCURRENCY="${CONCURRENCY:-32}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-240}"
SERVE_ONLY=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --artifact)      ARTIFACT_PATH="$2"; shift 2 ;;
        --num-samples)   NUM_SAMPLES="$2"; shift 2 ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        --serve-only)    SERVE_ONLY=1; shift ;;
        -h|--help)
            sed -n 's/^# \?//p' "$0" | head -30
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/scripts/logs"
SERVER_LOG="${LOG_DIR}/llmcompressor_server_${TIMESTAMP}.log"
EVAL_LOG="${LOG_DIR}/llmcompressor_eval_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

# ---- 1. Inspect artifact ----
echo "===================================================================="
echo " llm-compressor artifact test"
echo "===================================================================="
echo " artifact : ${ARTIFACT_PATH}"
echo " port     : ${PORT}"
echo " samples  : ${NUM_SAMPLES}"
echo " serve-only: ${SERVE_ONLY}"
echo "===================================================================="

# ---- 1a. Ensure SALA sparse backend source is bf16 (not fp16-patched) ----
# This artifact loads with --dtype bfloat16. If a prior gptq_marlin run
# left python sources fp16-patched (sed-replaced bfloat16 → float16) and
# its cleanup trap didn't fire (e.g. SIGKILLed by timeout), sglang will
# crash here with "query and key must have the same dtype".
BACKEND_DIR="${REPO_ROOT}/python/sglang/srt/layers/attention"
for pyfile in "${BACKEND_DIR}/minicpm_backend.py" \
              "${BACKEND_DIR}/minicpm_sparse_utils.py"; do
    if [ -f "${pyfile}.local_eval.bak" ]; then
        echo "[bf16-guard] restoring ${pyfile##*/} from .local_eval.bak"
        mv "${pyfile}.local_eval.bak" "${pyfile}"
    fi
done
N_BF16=$(grep -c "torch\.bfloat16" "${BACKEND_DIR}/minicpm_backend.py" \
                                    "${BACKEND_DIR}/minicpm_sparse_utils.py" \
         2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
if [ "${N_BF16}" -lt 5 ]; then
    echo "[bf16-guard] WARN: only ${N_BF16} bfloat16 lines in SALA backend" >&2
    echo "[bf16-guard]       (expected ≥5+2=7). Server may crash on dtype mismatch." >&2
fi

if [ ! -d "${ARTIFACT_PATH}" ]; then
    echo "FATAL: artifact dir not found at ${ARTIFACT_PATH}" >&2
    echo "       (memory says it was last seen 2026-05-14, ~5.4 GB)" >&2
    echo "       Check labmate cleanup or /root/autodl-fs/ remount" >&2
    exit 1
fi

echo
echo "[inspect] artifact dir listing:"
ls -la "${ARTIFACT_PATH}" | head -20

echo
echo "[inspect] config.json quantization_config field:"
if [ -f "${ARTIFACT_PATH}/config.json" ]; then
    python3 -c "import json; c=json.load(open('${ARTIFACT_PATH}/config.json')); import pprint; pprint.pprint(c.get('quantization_config', '(no quantization_config field)'))"
else
    echo "  (config.json missing — likely a broken artifact)"
    exit 1
fi

echo
echo "[inspect] recipe.yaml (llm-compressor source recipe):"
if [ -f "${ARTIFACT_PATH}/recipe.yaml" ]; then
    cat "${ARTIFACT_PATH}/recipe.yaml"
else
    echo "  (recipe.yaml missing — artifact may not be from llm-compressor)"
fi

# ---- 2. Launch SGLang ----
echo
echo "[launch] starting SGLang server on port ${PORT}..."
echo "[launch] log: ${SERVER_LOG}"
source "${VENV_DIR}/bin/activate"

# Memory says: --quantization compressed-tensors  --dtype bfloat16
# (NOT --quantization gptq_marlin, NOT --dtype float16)
SGLANG_ARGS=(
    --model "${ARTIFACT_PATH}"
    --trust-remote-code
    --disable-radix-cache
    --quantization compressed-tensors
    --dtype bfloat16
    --attention-backend minicpm_flashinfer
    --chunked-prefill-size 8192
    --max-running-requests 32
    --skip-server-warmup
    --port "${PORT}"
    --dense-as-sparse
)

python3 -m sglang.launch_server "${SGLANG_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[launch] server PID: ${SERVER_PID}"

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[cleanup] killing server PID ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for /health
echo "[launch] waiting for /health (up to ${STARTUP_TIMEOUT}s)..."
elapsed=0
while [ "${elapsed}" -lt "${STARTUP_TIMEOUT}" ]; do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[launch] server READY at ${elapsed}s"
        break
    fi
    # Crash detection
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "FATAL: server died during startup. Last 30 lines of log:" >&2
        tail -30 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ "${elapsed}" -ge "${STARTUP_TIMEOUT}" ]; then
    echo "FATAL: server did not become ready within ${STARTUP_TIMEOUT}s" >&2
    tail -30 "${SERVER_LOG}" >&2
    exit 1
fi

if [ "${SERVE_ONLY}" = 1 ]; then
    echo
    echo "[serve-only] server up on port ${PORT}. Send a test prompt:"
    echo "  curl -X POST http://127.0.0.1:${PORT}/v1/chat/completions \\"
    echo "       -H 'Content-Type: application/json' \\"
    echo "       -d '{\"model\":\"minicpm-sala\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
    echo
    echo "Ctrl+C to terminate."
    wait "${SERVER_PID}"
    exit 0
fi

# ---- 3. Eval ----
echo
echo "[eval] running eval_model.py against ${EVAL_DATA}"
echo "[eval] log: ${EVAL_LOG}"

if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "FATAL: eval_model.py not found at ${EVAL_SCRIPT}" >&2
    exit 1
fi

python3 "${EVAL_SCRIPT}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --data "${EVAL_DATA}" \
    --max-samples "${NUM_SAMPLES}" \
    --concurrency "${CONCURRENCY}" \
    2>&1 | tee "${EVAL_LOG}"

# ---- 4. Summary ----
echo
echo "===================================================================="
echo " RESULT SUMMARY"
echo "===================================================================="
echo " server log: ${SERVER_LOG}"
echo " eval   log: ${EVAL_LOG}"
echo
echo " key lines from eval:"
grep -iE "acc_ori|overall_accuracy|Average Score|per.task|task.*=" "${EVAL_LOG}" | head -20

echo
echo "===================================================================="
echo " DECISION GUIDE"
echo "===================================================================="
echo " - If acc > 42 → llm-compressor path BEATS RTN; serious alternative"
echo "                 to the GPTQModel pipeline. Build a submission tarball."
echo " - If 0 < acc < 42 → llm-compressor path WORKS but quality below RTN."
echo "                     Maybe calibration tuning could help. Not worth"
echo "                     priority unless GPTQModel path also dies."
echo " - If acc = 0 → same failure mode as our GPTQModel path. Means the"
echo "                problem is deeper than tool choice (likely SGLang"
echo "                loader / MiniCPM-SALA structural issue)."
echo "===================================================================="
