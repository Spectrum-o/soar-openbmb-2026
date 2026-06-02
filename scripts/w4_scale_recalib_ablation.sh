#!/usr/bin/env bash
# GPU-side ablation for GPTQ W4A16 scale recalibration.
#
# This script keeps the experiment narrow:
#   1. copy an existing GPTQ-Marlin artifact;
#   2. rewrite only .scales tensors via tools/recalibrate_gptq_w4_scales.py;
#   3. verify dequantization on the recalibrated copy;
#   4. launch original and recalibrated artifacts with identical SGLang args;
#   5. run a small benchmark matrix and optional SOAR public-set eval.
#
# The recalibration itself is CPU/weight-only. GPU is needed for the serving
# and eval/benchmark comparison that tells us whether the bottleneck is weight
# reconstruction error, runtime latency, or long-context memory behavior.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
SOURCE_ARTIFACT="${SOURCE_ARTIFACT:-}"
WORK_DIR="${WORK_DIR:-/root/autodl-fs/zyn/w4_scale_recalib_$(date '+%Y%m%d_%H%M%S')}"
OUT_ARTIFACT="${OUT_ARTIFACT:-}"
VARIANT_DIR="${VARIANT_DIR:-}"
PORT="${PORT:-31111}"
HOST="${HOST:-127.0.0.1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
RUN_EVAL="${RUN_EVAL:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-150}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-32}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/root/autodl-fs/zyn/soar_toolkit/eval_model.py}"
EVAL_DATA="${EVAL_DATA:-/root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl}"
MODULE_REGEX="${MODULE_REGEX:-}"
LIMIT_MODULES="${LIMIT_MODULES:-}"
SKIP_COPY="${SKIP_COPY:-0}"
SKIP_RECAL="${SKIP_RECAL:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"

# S1-like decode, S8-like concurrent, and long-context pressure. These are not
# platform-equivalent, but they separate decode throughput, concurrency, and
# prefill/long-context pressure well enough to locate the next bottleneck.
BENCH_MATRIX="${BENCH_MATRIX:-decode,32,1024,512,1;concurrent,64,4096,256,8;longctx,16,32768,256,4}"

DEFAULT_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.70 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --dtype bfloat16"
SERVER_ARGS="${SERVER_ARGS:-}"

usage() {
    sed -n '1,34p' "$0" >&2
    cat >&2 <<EOF

Usage:
  bash scripts/w4_scale_recalib_ablation.sh \\
    --artifact /root/autodl-fs/zyn/models/last7attn-quantized \\
    --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\
    --variant submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120 \\
    --eval

Useful env:
  BENCH_MATRIX='decode,32,1024,512,1;concurrent,64,4096,256,8;longctx,16,32768,256,4'
  SERVER_ARGS='... explicit launch args ...'
  MODULE_REGEX='model.layers.(2[4-9]|3[0-1]).*self_attn'
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --artifact) SOURCE_ARTIFACT="$2"; shift 2 ;;
        --base) BASE_MODEL="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --out-artifact) OUT_ARTIFACT="$2"; shift 2 ;;
        --variant) VARIANT_DIR="$2"; shift 2 ;;
        --server-args) SERVER_ARGS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --eval) RUN_EVAL=1; shift ;;
        --eval-samples) EVAL_SAMPLES="$2"; shift 2 ;;
        --module-regex) MODULE_REGEX="$2"; shift 2 ;;
        --limit) LIMIT_MODULES="$2"; shift 2 ;;
        --skip-copy) SKIP_COPY=1; shift ;;
        --skip-recal) SKIP_RECAL=1; shift ;;
        --skip-bench) SKIP_BENCH=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [ -z "${SOURCE_ARTIFACT}" ]; then
    echo "error: --artifact is required" >&2
    exit 2
fi
if [ ! -d "${SOURCE_ARTIFACT}" ]; then
    echo "error: artifact dir not found: ${SOURCE_ARTIFACT}" >&2
    exit 2
fi
if [ ! -d "${BASE_MODEL}" ]; then
    echo "error: base model dir not found: ${BASE_MODEL}" >&2
    exit 2
fi

if [ -z "${SERVER_ARGS}" ] && [ -n "${VARIANT_DIR}" ]; then
    PREPARE_ENV="${REPO_ROOT}/${VARIANT_DIR}/prepare_env.sh"
    if [ ! -f "${PREPARE_ENV}" ]; then
        echo "error: variant prepare_env.sh not found: ${PREPARE_ENV}" >&2
        exit 2
    fi
    SERVER_ARGS="$(grep -E '^export SGLANG_SERVER_ARGS=' "${PREPARE_ENV}" | sed -E 's/^export SGLANG_SERVER_ARGS="//; s/"$//')"
fi
if [ -z "${SERVER_ARGS}" ]; then
    SERVER_ARGS="${DEFAULT_SERVER_ARGS}"
fi

mkdir -p "${WORK_DIR}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"
OUT_ARTIFACT="${OUT_ARTIFACT:-${WORK_DIR}/$(basename "${SOURCE_ARTIFACT}")-w4scale}"
SUMMARY_CSV="${WORK_DIR}/bench_summary.csv"
printf 'tag,profile,num_prompts,input_len,output_len,concurrency,output_tok_s,input_tok_s,mean_ttft_ms,mean_tpot_ms,mean_itl_ms,duration_s,jsonl,stdout\n' > "${SUMMARY_CSV}"

echo "=================================================="
echo " W4 scale recalibration ablation"
echo "=================================================="
echo " repo:            ${REPO_ROOT}"
echo " base model:      ${BASE_MODEL}"
echo " source artifact: ${SOURCE_ARTIFACT}"
echo " out artifact:    ${OUT_ARTIFACT}"
echo " work dir:        ${WORK_DIR}"
echo " server args:     ${SERVER_ARGS}"
echo " bench matrix:    ${BENCH_MATRIX}"
echo " run eval:        ${RUN_EVAL}"
echo "=================================================="

if [ "${SKIP_COPY}" -eq 0 ]; then
    rm -rf "${OUT_ARTIFACT}"
    echo "[copy] copying artifact to recalibration workspace"
    if command -v rsync >/dev/null 2>&1; then
        mkdir -p "${OUT_ARTIFACT}"
        rsync -a --delete "${SOURCE_ARTIFACT}/" "${OUT_ARTIFACT}/"
    else
        cp -a "${SOURCE_ARTIFACT}" "${OUT_ARTIFACT}"
    fi
else
    echo "[copy] skipped; using existing ${OUT_ARTIFACT}"
fi

RECAL_ARGS=(--artifact "${OUT_ARTIFACT}" --base "${BASE_MODEL}" --report "${WORK_DIR}/w4_scale_recalib_apply.json")
DRY_ARGS=(--artifact "${OUT_ARTIFACT}" --base "${BASE_MODEL}" --report "${WORK_DIR}/w4_scale_recalib_dryrun.json" --dry-run)
if [ -n "${MODULE_REGEX}" ]; then
    RECAL_ARGS+=(--module-regex "${MODULE_REGEX}")
    DRY_ARGS+=(--module-regex "${MODULE_REGEX}")
fi
if [ -n "${LIMIT_MODULES}" ]; then
    RECAL_ARGS+=(--limit "${LIMIT_MODULES}")
    DRY_ARGS+=(--limit "${LIMIT_MODULES}")
fi

if [ "${SKIP_RECAL}" -eq 0 ]; then
    echo "[recal] dry-run report"
    python3 "${REPO_ROOT}/tools/recalibrate_gptq_w4_scales.py" "${DRY_ARGS[@]}" | tee "${WORK_DIR}/w4_scale_recalib_dryrun.log"
    echo "[recal] applying scale rewrite to copied artifact"
    python3 "${REPO_ROOT}/tools/recalibrate_gptq_w4_scales.py" "${RECAL_ARGS[@]}" | tee "${WORK_DIR}/w4_scale_recalib_apply.log"
else
    echo "[recal] skipped"
fi

echo "[verify] dequant smoke on recalibrated artifact"
python3 "${REPO_ROOT}/tools/verify_artifact_dequant.py" \
    --artifact "${SOURCE_ARTIFACT}" \
    --base "${BASE_MODEL}" \
    --fast | tee "${WORK_DIR}/verify_original_fast.log"
python3 "${REPO_ROOT}/tools/verify_artifact_dequant.py" \
    --artifact "${OUT_ARTIFACT}" \
    --base "${BASE_MODEL}" \
    --fast | tee "${WORK_DIR}/verify_recalibrated_fast.log"

server_pid=""
monitor_pid=""

port_free() {
    python3 - "$HOST" "$PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
try:
    s.bind((host, port))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

stop_server() {
    if [ -n "${server_pid}" ] && kill -0 "${server_pid}" 2>/dev/null; then
        echo "[server] stopping pid=${server_pid}"
        kill -TERM "${server_pid}" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "${server_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${server_pid}" 2>/dev/null || true
    fi
    server_pid=""
    if [ -n "${monitor_pid}" ] && kill -0 "${monitor_pid}" 2>/dev/null; then
        kill -TERM "${monitor_pid}" 2>/dev/null || true
    fi
    monitor_pid=""
}
trap stop_server EXIT INT TERM

start_gpu_monitor() {
    local tag="$1"
    local out="${LOG_DIR}/nvidia_${tag}.log"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw --format=csv > "${LOG_DIR}/nvidia_${tag}_before.csv" 2>&1 || true
        nvidia-smi dmon -s pucm -d 1 -o DT > "${out}" 2>&1 &
        monitor_pid="$!"
        echo "[monitor] nvidia-smi dmon pid=${monitor_pid} log=${out}"
    fi
}

start_server() {
    local tag="$1"
    local model_dir="$2"
    local log_file="${LOG_DIR}/server_${tag}.log"

    if ! port_free; then
        echo "error: ${HOST}:${PORT} is already in use" >&2
        exit 1
    fi

    export GPTQMODEL_MARLIN_USE_FP32="${GPTQMODEL_MARLIN_USE_FP32:-1}"
    echo "[server:${tag}] starting ${model_dir}"
    # shellcheck disable=SC2086
    PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}" python3 -m sglang.launch_server \
        --model-path "${model_dir}" \
        --trust-remote-code \
        --host "${HOST}" \
        --port "${PORT}" \
        --tp-size 1 \
        --max-running-requests 32 \
        --enable-metrics \
        ${SERVER_ARGS} \
        > "${log_file}" 2>&1 &
    server_pid="$!"

    local elapsed=0
    while [ "${elapsed}" -lt "${STARTUP_TIMEOUT}" ]; do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "[server:${tag}] died during startup; tail:" >&2
            tail -80 "${log_file}" >&2 || true
            exit 1
        fi
        if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
            echo "[server:${tag}] ready after ${elapsed}s"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    echo "[server:${tag}] startup timeout; tail:" >&2
    tail -80 "${log_file}" >&2 || true
    exit 1
}

append_bench_summary() {
    local tag="$1" profile="$2" num="$3" in_len="$4" out_len="$5" conc="$6" jsonl="$7" stdout="$8"
    python3 - "$tag" "$profile" "$num" "$in_len" "$out_len" "$conc" "$jsonl" "$stdout" "$SUMMARY_CSV" <<'PY'
import json, sys
tag, profile, num, in_len, out_len, conc, jsonl, stdout, csv_path = sys.argv[1:]
row = {}
try:
    lines = [x for x in open(jsonl, encoding="utf-8").read().splitlines() if x.strip()]
    row = json.loads(lines[-1]) if lines else {}
except Exception:
    row = {}
def val(key):
    x = row.get(key, "-")
    return f"{x:.3f}" if isinstance(x, float) else str(x)
fields = [
    tag, profile, num, in_len, out_len, conc,
    val("output_throughput"),
    val("input_throughput"),
    val("mean_ttft_ms"),
    val("mean_tpot_ms"),
    val("mean_itl_ms"),
    val("duration"),
    jsonl,
    stdout,
]
with open(csv_path, "a", encoding="utf-8") as f:
    f.write(",".join(fields) + "\n")
print(f"[bench:{tag}:{profile}] output_tok_s={fields[6]} input_tok_s={fields[7]} ttft_ms={fields[8]} tpot_ms={fields[9]} duration={fields[11]}")
PY
}

run_bench_matrix() {
    local tag="$1"
    [ "${SKIP_BENCH}" -eq 0 ] || return 0
    IFS=';' read -r -a profiles <<< "${BENCH_MATRIX}"
    for spec in "${profiles[@]}"; do
        IFS=',' read -r profile num_prompts input_len output_len conc <<< "${spec}"
        local out_json="${LOG_DIR}/bench_${tag}_${profile}.jsonl"
        local out_log="${LOG_DIR}/bench_${tag}_${profile}.stdout"
        echo "[bench:${tag}:${profile}] prompts=${num_prompts} in=${input_len} out=${output_len} conc=${conc}"
        python3 -m sglang.bench_serving \
            --backend sglang \
            --host "${HOST}" \
            --port "${PORT}" \
            --dataset-name random \
            --num-prompts "${num_prompts}" \
            --random-input-len "${input_len}" \
            --random-output-len "${output_len}" \
            --random-range-ratio 1.0 \
            --max-concurrency "${conc}" \
            --output-file "${out_json}" \
            > "${out_log}" 2>&1
        append_bench_summary "${tag}" "${profile}" "${num_prompts}" "${input_len}" "${output_len}" "${conc}" "${out_json}" "${out_log}"
    done
}

run_eval() {
    local tag="$1"
    local model_dir="$2"
    [ "${RUN_EVAL}" -eq 1 ] || return 0
    if [ ! -f "${EVAL_SCRIPT}" ] || [ ! -f "${EVAL_DATA}" ]; then
        echo "[eval:${tag}] missing eval script/data; set EVAL_SCRIPT and EVAL_DATA" >&2
        return 1
    fi
    local help_log="${LOG_DIR}/eval_help_${tag}.log"
    python3 "${EVAL_SCRIPT}" --help > "${help_log}" 2>&1 || true
    local missing_flags=()
    for flag in --api_base --model_path --data_path --concurrency --num_samples; do
        if ! grep -q -- "${flag}" "${help_log}"; then
            missing_flags+=("${flag}")
        fi
    done
    if [ "${#missing_flags[@]}" -gt 0 ]; then
        echo "[eval:${tag}] EVAL_SCRIPT help did not advertise required flags: ${missing_flags[*]}" >&2
        echo "[eval:${tag}] help log: ${help_log}" >&2
        sed -n '1,120p' "${help_log}" >&2 || true
        return 1
    fi
    local out_log="${LOG_DIR}/eval_${tag}.log"
    echo "[eval:${tag}] samples=${EVAL_SAMPLES} concurrency=${EVAL_CONCURRENCY}"
    python3 "${EVAL_SCRIPT}" \
        --api_base "http://${HOST}:${PORT}" \
        --model_path "${model_dir}" \
        --data_path "${EVAL_DATA}" \
        --concurrency "${EVAL_CONCURRENCY}" \
        --num_samples "${EVAL_SAMPLES}" \
        2>&1 | tee "${out_log}"
}

run_one_model() {
    local tag="$1"
    local model_dir="$2"
    start_gpu_monitor "${tag}"
    start_server "${tag}" "${model_dir}"
    run_bench_matrix "${tag}"
    run_eval "${tag}" "${model_dir}"
    stop_server
}

run_one_model "orig" "${SOURCE_ARTIFACT}"
run_one_model "recal" "${OUT_ARTIFACT}"

python3 "${REPO_ROOT}/tools/summarize_w4_scale_ablation.py" \
    --work-dir "${WORK_DIR}" \
    --output "${WORK_DIR}/summary.md"

echo "=================================================="
echo " W4 scale ablation done"
echo " work dir:      ${WORK_DIR}"
echo " recal artifact:${OUT_ARTIFACT}"
echo " bench summary: ${SUMMARY_CSV}"
echo " summary:       ${WORK_DIR}/summary.md"
echo " reports:       ${WORK_DIR}/w4_scale_recalib_*.json"
echo "=================================================="
