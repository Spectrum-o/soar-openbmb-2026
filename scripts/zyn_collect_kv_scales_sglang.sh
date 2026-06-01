#!/usr/bin/env bash
set -euo pipefail

# Collect FP8 KV scales from the exact SGLang W4 serving path.
# This avoids a separate HF GPTQ loader and measures the selected W4 artifact.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/autodl-fs/data/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized}"
DATA_PATH="${DATA_PATH:-/autodl-fs/data/zyn/calib_sets/w4_kv_selected_20260601.jsonl}"
RUN_NAME="${RUN_NAME:-kv_calib_$(date +%Y%m%d_%H%M%S)}"
STATS_DIR="${STATS_DIR:-/autodl-fs/data/zyn/kv_calib_stats/${RUN_NAME}}"
OUTPUT="${OUTPUT:-/autodl-fs/data/zyn/kv_scales/$(basename "${MODEL_PATH}")_fp8_e4m3_kv_scales.json}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/zyn_logs}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/${RUN_NAME}_server.log}"
SEND_LOG="${SEND_LOG:-${LOG_DIR}/${RUN_NAME}_send.log}"
PORT="${PORT:-31111}"
SERVER_WAIT_SECS="${SERVER_WAIT_SECS:-1200}"
MAX_TOKENS="${MAX_TOKENS:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
COLLECT_KV_DTYPE="${COLLECT_KV_DTYPE:-bfloat16}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "error: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
if [ ! -f "${DATA_PATH}" ]; then
    echo "error: DATA_PATH does not exist: ${DATA_PATH}" >&2
    exit 1
fi
if [ ! -x "${REPO_ROOT}/sglang_minicpm_sala_env/bin/python" ]; then
    echo "error: runtime venv missing; run install_minicpm_sala.sh first" >&2
    exit 1
fi

mkdir -p "${STATS_DIR}" "${LOG_DIR}" "$(dirname "${OUTPUT}")"
rm -f "${STATS_DIR}"/minicpm_kv_stats_rank*_pid*.json

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[kv-calib] model:     ${MODEL_PATH}"
echo "[kv-calib] data:      ${DATA_PATH}"
echo "[kv-calib] stats_dir: ${STATS_DIR}"
echo "[kv-calib] output:    ${OUTPUT}"
echo "[kv-calib] server_log:${SERVER_LOG}"
echo "[kv-calib] send_log:  ${SEND_LOG}"
echo "[kv-calib] collect_kv_dtype: ${COLLECT_KV_DTYPE}"

(
    cd "${REPO_ROOT}"
    MODEL_PATH="${MODEL_PATH}" \
    QUANTIZATION_PARAM_PATH="" \
    SGLANG_MINICPM_KV_CALIB_DIR="${STATS_DIR}" \
    SGLANG_MINICPM_KV_CALIB_FLUSH_SECS=0 \
    PORT="${PORT}" \
    KV_DTYPE="${COLLECT_KV_DTYPE}" \
    DTYPE=bfloat16 \
    QUANTIZATION=gptq_marlin \
    DISABLE_CUDA_GRAPH=1 \
    MAX_RUNNING_REQUESTS=1 \
    CUDA_GRAPH_BS='1' \
    bash run_sala.sh
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SERVER_WAIT_SECS))
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "error: server exited before readiness; tail follows" >&2
        tail -160 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if [ "${SECONDS}" -gt "${deadline}" ]; then
        echo "error: server did not become ready within ${SERVER_WAIT_SECS}s" >&2
        tail -200 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 5
done

source "${REPO_ROOT}/sglang_minicpm_sala_env/bin/activate"
python "${REPO_ROOT}/tools/zyn_send_kv_calib_requests.py" \
    --data-path "${DATA_PATH}" \
    --api-base "http://127.0.0.1:${PORT}" \
    --max-tokens "${MAX_TOKENS}" \
    --limit "${NUM_SAMPLES}" \
    --output "${STATS_DIR}/requests.jsonl" \
    2>&1 | tee "${SEND_LOG}"

sleep 2
python "${REPO_ROOT}/tools/zyn_merge_minicpm_kv_stats.py" \
    --model-path "${MODEL_PATH}" \
    --stats-dir "${STATS_DIR}" \
    --output "${OUTPUT}" \
    --force

echo "[kv-calib] done: ${OUTPUT}"
echo "[kv-calib] run server with: QUANTIZATION_PARAM_PATH=${OUTPUT}"
