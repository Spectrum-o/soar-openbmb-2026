#!/usr/bin/env bash
# Local smoke-bench for SOAR submission variants.
#
# What it does, in order:
#   1. Pull SGLANG_SERVER_ARGS out of <variant>/prepare_env.sh (no patching,
#      no uv install — we use the already-installed sglang in the venv).
#   2. Optionally strip --quantization/--dtype if --bf16 (lets you smoke-test
#      any variant against the stock BF16 model without running prepare_model.sh).
#   3. Launch sglang.launch_server in the background.
#   4. Poll /health until ready (or timeout).
#   5. Send a smoke curl to /v1/chat/completions — fails fast if reply is junk.
#   6. Run a mini sglang.bench_serving (defaults: 16 prompts × in 4096 / out 256
#      at concurrency 4) and parse the JSONL output.
#   7. Append a single CSV row to scripts/bench_results.csv.
#   8. Kill the server on exit or interrupt.
#
# Total runtime is roughly 30-90s per variant on RTX PRO 6000 with the BF16
# model (most of it is server startup; quantized models add prepare-model time
# which this script does NOT do — quantize separately first).
#
# Usage:
#   scripts/smoke_bench.sh --variant submission_rtn_sym_w4a16 --bf16
#   scripts/smoke_bench.sh --variant submission_rtn_sym_v2_chunk32k \
#       --model-path /root/autodl-tmp/models/MiniCPM-SALA-W4A16-RTN-SYM
#
# Required env:
#   - venv with sglang installed (or run: source <repo>/sglang_minicpm_sala_env/bin/activate)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${REPO_ROOT}/scripts/bench_results.csv"
CSV_HEADER="timestamp,variant,bf16,conc,num_prompts,in_len,out_len,output_tok_s,input_tok_s,mean_ttft_ms,mean_tpot_ms,mean_itl_ms,bench_duration_s,status,server_log,bench_jsonl"

# --- Defaults ---
VARIANT_DIR=""
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
BF16_MODE=0
PORT="${PORT:-31111}"
NUM_PROMPTS=16
CONCURRENCY=4
INPUT_LEN=4096
OUTPUT_LEN=256
STARTUP_TIMEOUT=180     # seconds to wait for server ready

# --- Globals filled by setup ---
VARIANT_NAME=""
SERVER_LOG=""
BENCH_JSONL=""
SGLANG_SERVER_ARGS=""

# --- Helpers (must be defined before first use) ---
ensure_csv_header() {
    if [ ! -f "${RESULTS_CSV}" ]; then
        echo "${CSV_HEADER}" > "${RESULTS_CSV}"
    fi
}

append_csv_failure() {
    local status="$1"
    ensure_csv_header
    echo "$(date '+%F %T'),${VARIANT_NAME:-?},${BF16_MODE},${CONCURRENCY},${NUM_PROMPTS},${INPUT_LEN},${OUTPUT_LEN},-,-,-,-,-,-,${status},${SERVER_LOG:-?},-" >> "${RESULTS_CSV}"
}

append_csv_ok() {
    ensure_csv_header
    echo "$(date '+%F %T'),${VARIANT_NAME},${BF16_MODE},${CONCURRENCY},${NUM_PROMPTS},${INPUT_LEN},${OUTPUT_LEN},$1,$2,$3,$4,$5,$6,ok,${SERVER_LOG},${BENCH_JSONL}" >> "${RESULTS_CSV}"
}

extract_json_num() {
    python3 - "$1" "$2" <<'PY' 2>/dev/null || echo "-"
import json, sys
key = sys.argv[1]
try:
    row = json.loads(sys.argv[2])
    val = row.get(key, "-")
    if isinstance(val, float):
        print(f"{val:.3f}")
    else:
        print(val)
except Exception:
    print("-")
PY
}

# --- Arg parsing ---
while [ "$#" -gt 0 ]; do
    case "$1" in
        --variant)        VARIANT_DIR="$2"; shift 2 ;;
        --model-path)     MODEL_PATH="$2"; shift 2 ;;
        --bf16)           BF16_MODE=1; shift ;;
        --port)           PORT="$2"; shift 2 ;;
        --num-prompts)    NUM_PROMPTS="$2"; shift 2 ;;
        --concurrency)    CONCURRENCY="$2"; shift 2 ;;
        --input-len)      INPUT_LEN="$2"; shift 2 ;;
        --output-len)     OUTPUT_LEN="$2"; shift 2 ;;
        --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
        -h|--help)
            head -28 "$0" | tail -27
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "${VARIANT_DIR}" ]; then
    echo "error: --variant is required (e.g. submission_rtn_sym_v2_chunk32k)" >&2
    exit 2
fi
if [ ! -d "${REPO_ROOT}/${VARIANT_DIR}" ]; then
    echo "error: variant dir not found: ${REPO_ROOT}/${VARIANT_DIR}" >&2
    exit 2
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "error: model path not found: ${MODEL_PATH}" >&2
    exit 2
fi

VARIANT_NAME="$(basename "${VARIANT_DIR}")"
LOG_DIR="${REPO_ROOT}/scripts/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%s)"
SERVER_LOG="${LOG_DIR}/server_${VARIANT_NAME}_${STAMP}.log"
BENCH_JSONL="${LOG_DIR}/bench_${VARIANT_NAME}_${STAMP}.jsonl"
BENCH_STDOUT="${LOG_DIR}/bench_${VARIANT_NAME}_${STAMP}.stdout"

# --- Extract SGLANG_SERVER_ARGS from the variant's prepare_env.sh ---
PREPARE_ENV="${REPO_ROOT}/${VARIANT_DIR}/prepare_env.sh"
if [ ! -f "${PREPARE_ENV}" ]; then
    echo "error: ${PREPARE_ENV} missing" >&2
    exit 2
fi
SGLANG_SERVER_ARGS="$(grep -E '^export SGLANG_SERVER_ARGS=' "${PREPARE_ENV}" \
    | sed -E 's/^export SGLANG_SERVER_ARGS="//; s/"$//')"
if [ -z "${SGLANG_SERVER_ARGS}" ]; then
    echo "error: could not extract SGLANG_SERVER_ARGS from ${PREPARE_ENV}" >&2
    exit 2
fi

# If --bf16, drop --quantization X and --dtype X so we can smoke-test against
# the original BF16 weights without running prepare_model.sh.
if [ "${BF16_MODE}" -eq 1 ]; then
    SGLANG_SERVER_ARGS="$(echo "${SGLANG_SERVER_ARGS}" | sed -E 's/--quantization [^ ]+//; s/--dtype [^ ]+//' | tr -s ' ')"
fi

echo "=================================================="
echo " variant:     ${VARIANT_NAME}"
echo " model:       ${MODEL_PATH}"
echo " bf16 mode:   ${BF16_MODE}"
echo " port:        ${PORT}"
echo " bench:       ${NUM_PROMPTS} prompts, conc=${CONCURRENCY}, in=${INPUT_LEN}, out=${OUTPUT_LEN}"
echo " server args: ${SGLANG_SERVER_ARGS}"
echo " server log:  ${SERVER_LOG}"
echo "=================================================="

# --- Launch server ---
echo "[smoke] launching sglang server..."
# shellcheck disable=SC2086
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
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
        echo "[smoke] killing server pid=${SERVER_PID}"
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --- Wait for /health ---
echo "[smoke] waiting for /health (max ${STARTUP_TIMEOUT}s)..."
ELAPSED=0
while [ "${ELAPSED}" -lt "${STARTUP_TIMEOUT}" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[smoke] server died during startup. tail of log:" >&2
        tail -40 "${SERVER_LOG}" >&2
        append_csv_failure "server_died_startup"
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[smoke] server is up (${ELAPSED}s)"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ "${ELAPSED}" -ge "${STARTUP_TIMEOUT}" ]; then
    echo "[smoke] timeout waiting for server. tail of log:" >&2
    tail -40 "${SERVER_LOG}" >&2
    append_csv_failure "startup_timeout"
    exit 1
fi

# --- Smoke curl ---
SMOKE_RESP=$(curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"minicpm-sala","messages":[{"role":"user","content":"hello"}],"temperature":0.9,"max_tokens":16}' \
    || echo "CURL_FAILED")
if [ "${SMOKE_RESP}" = "CURL_FAILED" ] || ! echo "${SMOKE_RESP}" | grep -q '"content"'; then
    echo "[smoke] smoke curl failed. response:" >&2
    echo "${SMOKE_RESP}" >&2
    append_csv_failure "smoke_curl_failed"
    exit 1
fi
echo "[smoke] smoke curl ok"

# --- Mini bench_serving ---
echo "[smoke] running bench_serving..."
python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port "${PORT}" \
    --dataset-name random \
    --num-prompts "${NUM_PROMPTS}" \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-range-ratio 1.0 \
    --max-concurrency "${CONCURRENCY}" \
    --output-file "${BENCH_JSONL}" \
    > "${BENCH_STDOUT}" 2>&1 || {
    echo "[smoke] bench_serving failed. tail of stdout:" >&2
    tail -40 "${BENCH_STDOUT}" >&2
    append_csv_failure "bench_failed"
    exit 1
}

# --- Parse the last line of the JSONL (the summary record) ---
LAST_LINE="$(tail -1 "${BENCH_JSONL}" 2>/dev/null || true)"
if [ -z "${LAST_LINE}" ]; then
    echo "[smoke] no bench output. tail of stdout:" >&2
    tail -40 "${BENCH_STDOUT}" >&2
    append_csv_failure "no_bench_output"
    exit 1
fi

OUTPUT_THROUGHPUT=$(extract_json_num output_throughput "${LAST_LINE}")
INPUT_THROUGHPUT=$(extract_json_num input_throughput "${LAST_LINE}")
MEAN_TTFT=$(extract_json_num mean_ttft_ms "${LAST_LINE}")
MEAN_ITL=$(extract_json_num mean_itl_ms "${LAST_LINE}")
MEAN_TPOT=$(extract_json_num mean_tpot_ms "${LAST_LINE}")
DURATION=$(extract_json_num duration "${LAST_LINE}")

append_csv_ok "${OUTPUT_THROUGHPUT}" "${INPUT_THROUGHPUT}" "${MEAN_TTFT}" "${MEAN_TPOT}" "${MEAN_ITL}" "${DURATION}"

echo "=================================================="
echo " RESULT"
echo "   output throughput: ${OUTPUT_THROUGHPUT} tok/s"
echo "   input throughput:  ${INPUT_THROUGHPUT} tok/s"
echo "   mean TTFT:         ${MEAN_TTFT} ms"
echo "   mean TPOT:         ${MEAN_TPOT} ms"
echo "   mean ITL:          ${MEAN_ITL} ms"
echo "   bench duration:    ${DURATION} s"
echo " logs: ${SERVER_LOG}"
echo " csv:  ${RESULTS_CSV}"
echo "=================================================="
