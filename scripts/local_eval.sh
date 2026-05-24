#!/usr/bin/env bash
# Local correctness eval for SOAR submission variants — run this BEFORE
# uploading to the platform (saves 5h of platform eval per iteration).
#
# What it does:
#   1. Extract SGLANG_SERVER_ARGS from <variant>/prepare_env.sh.
#   2. Quantize the model using <variant>/quantize_gptq_rtn_sym.py
#      (or skip if a cached quantized model already exists).
#   3. Download SOAR-Toolkit's eval_model.py + perf_public_set.jsonl if missing.
#   4. Launch sglang.launch_server with the variant's args + the quantized model.
#   5. Wait for /health.
#   6. Run eval_model.py against the live server.
#   7. Parse and print acc_ori + overall_accuracy.
#   8. Kill server on exit.
#
# Total time: ~10-20 min first run (most is quantization), ~5-7 min cached.
# Compare to 5h on the platform.
#
# Usage:
#   bash scripts/local_eval.sh --variant submission_rtn_sym_w4a16
#   bash scripts/local_eval.sh --variant submission_rtn_sym_w4a16 --num-samples 100
#   bash scripts/local_eval.sh --variant submission_rtn_sym_w4a16 --force-requant
#   bash scripts/local_eval.sh --variant submission_rtn_sym_w4a16 --skip-quant
#
# Required:
#   - venv with sglang installed: source sglang_minicpm_sala_env/bin/activate
#   - BF16 source model at /root/autodl-fs/models/OpenBMB/MiniCPM-SALA
#   - Network for first run (downloads eval_model.py + dataset)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${REPO_ROOT}/scripts/eval_results.csv"
CSV_HEADER="timestamp,variant,quant_model,num_samples,concurrency,acc_ori,acc_overall,duration_s,server_log,eval_log"

# --- Defaults ---
VARIANT_DIR=""
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
QUANT_BASE="${QUANT_BASE:-/root/autodl-fs/zyn/models}"
TOOLKIT_DIR="${TOOLKIT_DIR:-/root/autodl-fs/zyn/soar_toolkit}"
EVAL_SCRIPT="${TOOLKIT_DIR}/eval_model.py"
EVAL_DATA="${TOOLKIT_DIR}/perf_public_set.jsonl"
PORT="${PORT:-31111}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
CONCURRENCY="${CONCURRENCY:-32}"
STARTUP_TIMEOUT=240
FORCE_REQUANT=0
SKIP_QUANT=0
NO_EVAL=0

# --- Helpers ---
ensure_csv_header() {
    if [ ! -f "${RESULTS_CSV}" ]; then
        echo "${CSV_HEADER}" > "${RESULTS_CSV}"
    fi
}

# --- Arg parsing ---
while [ "$#" -gt 0 ]; do
    case "$1" in
        --variant)        VARIANT_DIR="$2"; shift 2 ;;
        --model-path)     MODEL_PATH="$2"; shift 2 ;;
        --quant-out)      QUANT_OUT="$2"; shift 2 ;;
        --eval-script)    EVAL_SCRIPT="$2"; shift 2 ;;
        --eval-data)      EVAL_DATA="$2"; shift 2 ;;
        --num-samples)    NUM_SAMPLES="$2"; shift 2 ;;
        --concurrency)    CONCURRENCY="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
        --force-requant)  FORCE_REQUANT=1; shift ;;
        --skip-quant)     SKIP_QUANT=1; shift ;;
        --no-eval)        NO_EVAL=1; shift ;;
        -h|--help)
            head -32 "$0" | tail -31
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "${VARIANT_DIR}" ]; then
    echo "error: --variant is required (e.g. submission_rtn_sym_w4a16)" >&2
    exit 2
fi
if [ ! -d "${REPO_ROOT}/${VARIANT_DIR}" ]; then
    echo "error: variant dir not found: ${REPO_ROOT}/${VARIANT_DIR}" >&2
    exit 2
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "error: source model not found: ${MODEL_PATH}" >&2
    exit 2
fi

VARIANT_NAME="$(basename "${VARIANT_DIR}")"
QUANT_OUT="${QUANT_OUT:-${QUANT_BASE}/${VARIANT_NAME}-quantized}"

# --- Extract SGLANG_SERVER_ARGS from variant's prepare_env.sh ---
PREPARE_ENV="${REPO_ROOT}/${VARIANT_DIR}/prepare_env.sh"
PREPARE_MODEL="${REPO_ROOT}/${VARIANT_DIR}/prepare_model.sh"
# Prefer the variant's own prepare_model.sh (it knows about calibration
# args, timeouts, etc.). Fall back to the legacy direct-script invocation
# for older RTN-style variants.
if [ -f "${PREPARE_MODEL}" ]; then
    QUANT_SCRIPT=""  # signal to use prepare_model.sh
else
    QUANT_SCRIPT="$(find "${REPO_ROOT}/${VARIANT_DIR}" -maxdepth 1 -name 'quantize_*.py' | sort | head -1)"
fi
if [ ! -f "${PREPARE_ENV}" ]; then
    echo "error: ${PREPARE_ENV} missing" >&2
    exit 2
fi
SGLANG_SERVER_ARGS="$(grep -E '^export SGLANG_SERVER_ARGS=' "${PREPARE_ENV}" \
    | sed -E 's/^export SGLANG_SERVER_ARGS="//; s/"$//')"

# --- Quant-variant fp16 sed patch on the venv's bundled SGLang ---
# SALA's sparse attention helpers in minicpm_backend.py / minicpm_sparse_utils.py
# hardcode torch.bfloat16. gptq_marlin emits fp16. Without this patch the
# server crashes at first query with `RuntimeError: query and key must have
# the same dtype`. prepare_env.sh runs this patch on the platform, but
# local_eval.sh never sources prepare_env.sh, so we replicate the patch here
# when the variant's launch args include `gptq_marlin`. The cleanup trap at
# the bottom of this script reverts the patch on exit, so subsequent baseline
# (BF16) runs against the same venv stay unaffected.
SED_PATCH_FILES=()
if echo "${SGLANG_SERVER_ARGS}" | grep -q "gptq_marlin" \
        && ! echo "${SGLANG_SERVER_ARGS}" | grep -qE -- "--dtype[ =]bfloat16"; then
    BACKEND_DIR="${REPO_ROOT}/python/sglang/srt/layers/attention"
    for pyfile in "${BACKEND_DIR}/minicpm_backend.py" \
                  "${BACKEND_DIR}/minicpm_sparse_utils.py"; do
        if [ -f "${pyfile}" ] && grep -qE 'torch\.bfloat16|"bfloat16"' "${pyfile}"; then
            cp "${pyfile}" "${pyfile}.local_eval.bak"
            sed -i 's/torch\.bfloat16/torch.float16/g' "${pyfile}"
            sed -i 's/"bfloat16"/"float16"/g' "${pyfile}"
            SED_PATCH_FILES+=("${pyfile}")
            echo "[local_eval] sed-patched $(basename "${pyfile}") for fp16 quant run"
        fi
    done
elif echo "${SGLANG_SERVER_ARGS}" | grep -q "gptq_marlin"; then
    echo "[local_eval] gptq_marlin + --dtype bfloat16 detected; SKIPPING fp16 sed-patch (mirrors v5j_dtype_bf16+ variants' prepare_env.sh behavior)"
fi

LOG_DIR="${REPO_ROOT}/scripts/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%s)"
SERVER_LOG="${LOG_DIR}/local_eval_server_${VARIANT_NAME}_${STAMP}.log"
EVAL_LOG="${LOG_DIR}/local_eval_eval_${VARIANT_NAME}_${STAMP}.log"
QUANT_LOG="${LOG_DIR}/local_eval_quant_${VARIANT_NAME}_${STAMP}.log"

echo "=================================================="
echo " local_eval"
echo "=================================================="
echo " variant:      ${VARIANT_NAME}"
echo " source model: ${MODEL_PATH}"
echo " quant cache:  ${QUANT_OUT}"
echo " eval script:  ${EVAL_SCRIPT}"
echo " eval data:    ${EVAL_DATA}"
echo " samples:      ${NUM_SAMPLES} (concurrency=${CONCURRENCY})"
echo " sglang args:  ${SGLANG_SERVER_ARGS}"
echo "=================================================="

START_TS=$(date +%s)

# --- Step 1: Quantize (or skip) ---
if [ "${SKIP_QUANT}" -eq 1 ]; then
    echo "[1/3] quantize: SKIPPED via --skip-quant"
elif [ "${FORCE_REQUANT}" -eq 0 ] && [ -d "${QUANT_OUT}" ] && [ -f "${QUANT_OUT}/config.json" ]; then
    echo "[1/3] quantize: CACHED at ${QUANT_OUT} (use --force-requant to rebuild)"
else
    rm -rf "${QUANT_OUT}"
    mkdir -p "${QUANT_OUT}"
    if [ -z "${QUANT_SCRIPT}" ]; then
        echo "[1/3] quantize: running ${PREPARE_MODEL}..."
        bash "${PREPARE_MODEL}" \
            --input "${MODEL_PATH}" \
            --output "${QUANT_OUT}" \
            > "${QUANT_LOG}" 2>&1 || {
            echo "[1/3] quantize FAILED. tail of log:" >&2
            tail -50 "${QUANT_LOG}" >&2
            exit 1
        }
    else
        echo "[1/3] quantize: running ${QUANT_SCRIPT}..."
        python3 "${QUANT_SCRIPT}" \
            --input "${MODEL_PATH}" \
            --output "${QUANT_OUT}" \
            > "${QUANT_LOG}" 2>&1 || {
            echo "[1/3] quantize FAILED. tail of log:" >&2
            tail -30 "${QUANT_LOG}" >&2
            exit 1
        }
    fi
    echo "  ✓ quantize done. config: $(jq -r '.bits, .group_size, .sym' "${QUANT_OUT}/quantize_config.json" 2>/dev/null | tr '\n' ' ')"
fi

# --- Step 2: Make sure eval_model.py + dataset are present ---
if [ "${NO_EVAL}" -eq 0 ]; then
    mkdir -p "${TOOLKIT_DIR}"
    if [ ! -f "${EVAL_SCRIPT}" ]; then
        echo "[2/3] downloading eval_model.py..."
        curl -fsSL -o "${EVAL_SCRIPT}" \
            "https://raw.githubusercontent.com/OpenBMB/SOAR-Toolkit/main/eval_model.py" || {
            echo "  ! failed to download eval_model.py; pass --eval-script <path> manually" >&2
            exit 1
        }
    fi
    if [ ! -f "${EVAL_DATA}" ]; then
        echo "[2/3] downloading perf_public_set.jsonl..."
        curl -fsSL -o "${EVAL_DATA}" \
            "https://raw.githubusercontent.com/OpenBMB/SOAR-Toolkit/main/eval_dataset/perf_public_set.jsonl" || {
            echo "  ! failed to download dataset; pass --eval-data <path> manually" >&2
            exit 1
        }
    fi
    echo "  ✓ eval artifacts ready"
fi

# --- Step 3: Launch SGLang ---
echo "[3/3] launching sglang server..."
# shellcheck disable=SC2086
python3 -m sglang.launch_server \
    --model-path "${QUANT_OUT}" \
    --trust-remote-code \
    --port "${PORT}" \
    --tp-size 1 \
    --max-running-requests 32 \
    --enable-metrics \
    ${SGLANG_SERVER_ARGS} \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "  killing server pid=${SERVER_PID}"
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    # Revert the fp16 sed patch so subsequent baseline (BF16) runs are not poisoned.
    for pyfile in "${SED_PATCH_FILES[@]}"; do
        if [ -f "${pyfile}.local_eval.bak" ]; then
            mv "${pyfile}.local_eval.bak" "${pyfile}"
            echo "  reverted fp16 sed patch on $(basename "${pyfile}")"
        fi
    done
}
trap cleanup EXIT INT TERM

# wait for /health
echo "  waiting for /health (max ${STARTUP_TIMEOUT}s, log: ${SERVER_LOG})..."
ELAPSED=0
while [ "${ELAPSED}" -lt "${STARTUP_TIMEOUT}" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "  ! server died. tail of log:" >&2
        tail -40 "${SERVER_LOG}" >&2
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  ✓ server up after ${ELAPSED}s"
        break
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done

if [ "${ELAPSED}" -ge "${STARTUP_TIMEOUT}" ]; then
    echo "  ! startup timeout. tail of log:" >&2
    tail -40 "${SERVER_LOG}" >&2
    exit 1
fi

if [ "${NO_EVAL}" -eq 1 ]; then
    echo "  --no-eval set; server is up. Hit Ctrl+C to stop."
    wait "${SERVER_PID}"
    exit 0
fi

# --- Run eval_model.py ---
echo "  running eval_model.py..."

# Most SOAR eval_model.py variants accept these args. If --num-samples isn't
# supported the run still works (it just uses the full dataset).
EVAL_ARGS=(
    --api_base "http://127.0.0.1:${PORT}"
    --model_path "${QUANT_OUT}"
    --data_path "${EVAL_DATA}"
    --concurrency "${CONCURRENCY}"
)
if [ "${NUM_SAMPLES}" -gt 0 ]; then
    EVAL_ARGS+=(--num_samples "${NUM_SAMPLES}")
fi

python3 "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}" 2>&1 | tee "${EVAL_LOG}"
EVAL_EXIT=${PIPESTATUS[0]}

if [ "${EVAL_EXIT}" -ne 0 ]; then
    echo "  ! eval_model.py exited ${EVAL_EXIT}. tail of log:" >&2
    tail -20 "${EVAL_LOG}" >&2
    exit "${EVAL_EXIT}"
fi

# Parse acc_ori / overall_accuracy from log. Different eval_model.py versions
# print these in slightly different shapes:
#   - older: "ori_accuracy: 42.51" / "overall_accuracy: 21.25"
#   - current SOAR toolkit: "Average Score: 63.33%" (single number)
# Match whichever shows up; allow integer-only ("Average Score: 63%").
ACC_ORI=$(grep -oE '(ori_accuracy|Average Score)[^0-9-]*[0-9]+(\.[0-9]+)?' "${EVAL_LOG}" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -1 || echo "?")
ACC_OVERALL=$(grep -oE '(overall_accuracy|Average Score)[^0-9-]*[0-9]+(\.[0-9]+)?' "${EVAL_LOG}" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -1 || echo "?")
END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))

# Record in CSV
ensure_csv_header
echo "$(date '+%F %T'),${VARIANT_NAME},${QUANT_OUT},${NUM_SAMPLES},${CONCURRENCY},${ACC_ORI},${ACC_OVERALL},${DURATION},${SERVER_LOG},${EVAL_LOG}" >> "${RESULTS_CSV}"

echo "=================================================="
echo " RESULT"
echo "   acc_ori (raw on dataset):  ${ACC_ORI}"
echo "   overall_accuracy (vs 80):  ${ACC_OVERALL}"
echo "   total wall clock:          ${DURATION}s"
echo "=================================================="
echo " >> Pass platform correctness gate? "
if [ "${ACC_ORI}" != "?" ] && python3 -c "import sys; sys.exit(0 if float('${ACC_ORI}') >= 80 else 1)"; then
    echo "    ✅ YES (acc_ori >= 80) — safe to submit"
else
    echo "    ❌ NO (acc_ori < 80) — would score 0 on the platform; keep iterating"
fi
echo "=================================================="
echo " server log:  ${SERVER_LOG}"
echo " eval log:    ${EVAL_LOG}"
echo " results csv: ${RESULTS_CSV}"
