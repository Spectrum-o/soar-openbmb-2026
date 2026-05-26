#!/usr/bin/env bash
set -euo pipefail

# Serve one local candidate with the standard platform-like runtime settings,
# run first30, then compare against the current best MLP-only fp8_e4m3 baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/autodl-fs/data/zyn/models/submission_gptqmodel_mlp_descact_static_20260527-quantized}"
DATA_PATH="${DATA_PATH:-/autodl-fs/data/zyn/soar_toolkit/perf_public_set.jsonl}"
BASELINE_PRED="${BASELINE_PRED:-/root/autodl-tmp/zyn/eval_runs/fp8kv_cg_e4m3_20260526_180812/predictions.jsonl}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/zyn/eval_runs}"
RUN_NAME="${RUN_NAME:-$(basename "${MODEL_PATH}")_first30_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/zyn_logs}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/${RUN_NAME}_server.log}"
PORT="${PORT:-31111}"
SERVER_WAIT_SECS="${SERVER_WAIT_SECS:-1200}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "error: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
if [ ! -f "${DATA_PATH}" ]; then
    echo "error: DATA_PATH does not exist: ${DATA_PATH}" >&2
    exit 1
fi
if [ ! -f "${BASELINE_PRED}" ]; then
    echo "error: BASELINE_PRED does not exist: ${BASELINE_PRED}" >&2
    exit 1
fi
if [ ! -x "${REPO_ROOT}/sglang_minicpm_sala_env/bin/python" ]; then
    echo "error: runtime venv missing; run install_minicpm_sala.sh first" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[serve] model: ${MODEL_PATH}"
echo "[serve] log:   ${SERVER_LOG}"
(
    cd "${REPO_ROOT}"
    MODEL_PATH="${MODEL_PATH}" \
    KV_DTYPE=fp8_e4m3 \
    DTYPE=bfloat16 \
    QUANTIZATION=gptq_marlin \
    PORT="${PORT}" \
    MAX_RUNNING_REQUESTS=32 \
    CUDA_GRAPH_BS='1 2 4 8 12 16 24 32' \
    bash run_sala.sh
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SERVER_WAIT_SECS))
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "error: server exited before readiness; tail follows" >&2
        tail -120 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if [ "${SECONDS}" -gt "${deadline}" ]; then
        echo "error: server did not become ready within ${SERVER_WAIT_SECS}s" >&2
        tail -160 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 5
done

echo "[eval] run_dir: ${RUN_DIR}"
source "${REPO_ROOT}/sglang_minicpm_sala_env/bin/activate"
python "${REPO_ROOT}/scripts/zyn_partition_eval.py" \
    --model-path "${MODEL_PATH}" \
    --api-base "http://127.0.0.1:${PORT}" \
    --data-path "${DATA_PATH}" \
    --run-dir "${RUN_DIR}" \
    --start-index 0 \
    --end-index 30 \
    --part-size 30 \
    --concurrency 32 \
    --max-out-len 65536 \
    --timeout 3000

python "${REPO_ROOT}/scripts/zyn_analyze_partition_results.py" --run-dir "${RUN_DIR}"

python "${REPO_ROOT}/scripts/zyn_compare_eval_runs.py" \
    --start-index 0 \
    --end-index 30 \
    --run "safe_e4m3=${BASELINE_PRED}" \
    --run "candidate=${RUN_DIR}/predictions.jsonl" | tee "${RUN_DIR}/compare_first30.txt"

echo "[eval] done: ${RUN_DIR}"
