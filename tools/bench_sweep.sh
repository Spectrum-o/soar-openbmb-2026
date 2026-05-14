#!/bin/bash
# Bench sweep: run sglang against the SALA model with multiple configurations
# and collect bench_serving results for comparison.
#
# Each config is a single line in a config file: tag,extra_launch_args
# Lines starting with # are comments.
#
# Usage:
#   bash tools/bench_sweep.sh tools/configs/chunk_size_sweep.txt
#
# Environment:
#   MODEL_PATH        - default /root/autodl-fs/models/OpenBMB/MiniCPM-SALA
#   PORT              - default 31111
#   OUTPUT_DIR        - default /root/autodl-fs/zyn/bench_sweeps/<timestamp>
#   BENCH_NUM_PROMPTS - default 64
#   BENCH_INPUT_LEN   - default 4096
#   BENCH_OUTPUT_LEN  - default 512
#   BENCH_DATASET     - default random-ids
#   SERVER_BOOT_TIMEOUT - default 600 (seconds)
#   PAUSE_BETWEEN     - default 15 (seconds to wait after kill before next launch)
#
# Output:
#   ${OUTPUT_DIR}/
#     ${tag}.server.log     - server stdout/stderr per config
#     ${tag}.bench.json     - bench_serving output per config
#     ${tag}.bench.txt      - bench_serving stdout per config
#     summary.md            - markdown summary table written after each run
#
# Resilience:
#   - One config failing does not abort the sweep; failures are recorded.
#   - Re-running with the same OUTPUT_DIR skips already-completed configs
#     (looks for *.bench.json files).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/sglang_minicpm_sala_env"

CONFIG_FILE="${1:-}"
if [ -z "${CONFIG_FILE}" ] || [ ! -f "${CONFIG_FILE}" ]; then
    echo "Usage: $0 <config-file>"
    echo "Example: $0 tools/configs/chunk_size_sweep.txt"
    exit 2
fi

MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
PORT="${PORT:-31111}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/zyn/bench_sweeps/$(date +%Y%m%d_%H%M%S)}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-64}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-4096}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-512}"
BENCH_DATASET="${BENCH_DATASET:-random-ids}"
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-600}"
PAUSE_BETWEEN="${PAUSE_BETWEEN:-15}"

mkdir -p "${OUTPUT_DIR}"
SUMMARY="${OUTPUT_DIR}/summary.md"

log() { printf "\033[1;36m[sweep]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[sweep-warn]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[sweep-err]\033[0m %s\n" "$*"; }

# Check venv
if [ ! -d "${VENV_DIR}" ]; then
    err "venv not found at ${VENV_DIR}"
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Kill any leftover sglang
kill_server() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep "${PAUSE_BETWEEN}"
}

wait_for_health() {
    local elapsed=0
    while [ "$elapsed" -lt "$SERVER_BOOT_TIMEOUT" ]; do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 1
}

run_one() {
    local tag="$1"
    local extra_args="$2"

    local server_log="${OUTPUT_DIR}/${tag}.server.log"
    local bench_json="${OUTPUT_DIR}/${tag}.bench.json"
    local bench_txt="${OUTPUT_DIR}/${tag}.bench.txt"

    if [ -f "${bench_json}" ]; then
        log "skip ${tag} (already has ${bench_json})"
        return 0
    fi

    log "=== ${tag} ==="
    log "args: ${extra_args}"
    kill_server

    # shellcheck disable=SC2086
    nohup python3 -m sglang.launch_server \
        --model "${MODEL_PATH}" \
        --trust-remote-code \
        --disable-radix-cache \
        --skip-server-warmup \
        --port "${PORT}" \
        --dense-as-sparse \
        ${extra_args} \
        > "${server_log}" 2>&1 &
    local server_pid=$!
    log "server PID ${server_pid}, waiting for health..."

    if ! wait_for_health; then
        err "${tag}: server did not become healthy in ${SERVER_BOOT_TIMEOUT}s"
        kill_server
        echo "FAIL ${tag} (boot timeout). See ${server_log}" >> "${OUTPUT_DIR}/failures.log"
        return 1
    fi
    log "${tag}: healthy"

    if ! python3 -m sglang.bench_serving \
        --backend sglang \
        --host 127.0.0.1 --port "${PORT}" \
        --dataset-name "${BENCH_DATASET}" \
        --num-prompts "${BENCH_NUM_PROMPTS}" \
        --random-input-len "${BENCH_INPUT_LEN}" \
        --random-output-len "${BENCH_OUTPUT_LEN}" \
        --output-file "${bench_json}" \
        2>&1 | tee "${bench_txt}"; then
        err "${tag}: bench_serving failed"
        kill_server
        echo "FAIL ${tag} (bench failed)" >> "${OUTPUT_DIR}/failures.log"
        return 1
    fi

    log "${tag}: bench saved -> ${bench_json}"
    kill_server
    return 0
}

# Rebuild summary after each run so partial progress is visible
build_summary() {
    python3 "${REPO_ROOT}/tools/compare_bench.py" "${OUTPUT_DIR}"/*.bench.json \
        --output "${SUMMARY}" 2>/dev/null || true
}

log "sweep output dir: ${OUTPUT_DIR}"
log "config file:      ${CONFIG_FILE}"
log "workload:         ${BENCH_DATASET} ${BENCH_NUM_PROMPTS}p in${BENCH_INPUT_LEN} out${BENCH_OUTPUT_LEN}"
log ""

while IFS= read -r line || [ -n "$line" ]; do
    # Skip blank/comment lines
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    tag="${line%%,*}"
    extra="${line#*,}"
    # Trim whitespace
    tag="$(echo "$tag" | xargs)"
    extra="$(echo "$extra" | xargs)"

    run_one "$tag" "$extra" || warn "config ${tag} failed, continuing"
    build_summary
done < "${CONFIG_FILE}"

log ""
log "DONE. Summary: ${SUMMARY}"
log "All artifacts: ${OUTPUT_DIR}/"
build_summary
cat "${SUMMARY}" 2>/dev/null || true
