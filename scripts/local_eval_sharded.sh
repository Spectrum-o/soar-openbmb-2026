#!/usr/bin/env bash
# Sharded local correctness eval for SOAR variants.
#
# Difference from local_eval.sh:
#   - Quantizes once.
#   - Starts one SGLang server.
#   - Splits the eval JSONL into non-overlapping shards.
#   - Runs eval_model.py shard by shard, printing shard and cumulative acc
#     after every shard so a bad run can be stopped early.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
RESULTS_CSV="${REPO_ROOT}/scripts/eval_shards.csv"
CSV_HEADER="timestamp,variant,quant_model,shard_index,shard_start,shard_end,shard_samples,cumulative_samples,concurrency,shard_acc,cumulative_acc,shard_duration_s,total_duration_s,server_log,eval_log,predictions_path"

VARIANT_DIR=""
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
QUANT_BASE="${QUANT_BASE:-/root/autodl-fs/zyn/models}"
TOOLKIT_DIR="${TOOLKIT_DIR:-/root/autodl-fs/zyn/soar_toolkit}"
EVAL_SCRIPT="${TOOLKIT_DIR}/eval_model.py"
EVAL_DATA="${TOOLKIT_DIR}/perf_public_set.jsonl"
PORT="${PORT:-31111}"
MAX_SAMPLES="${MAX_SAMPLES:-150}"
SHARD_SIZE="${SHARD_SIZE:-30}"
CONCURRENCY="${CONCURRENCY:-32}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-240}"
CHECKPOINT_AFTER_SHARD="${CHECKPOINT_AFTER_SHARD:-1}"
CHECKPOINT_SCRIPT="${CHECKPOINT_SCRIPT:-${REPO_ROOT}/scripts/checkpoint_full_w4a16.sh}"
FORCE_REQUANT=0
SKIP_QUANT=0

ensure_csv_header() {
    if [ ! -f "${RESULTS_CSV}" ]; then
        echo "${CSV_HEADER}" > "${RESULTS_CSV}"
    fi
}

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Usage:
  bash scripts/local_eval_sharded.sh --variant submission_gptqmodel_full_w4a16
  bash scripts/local_eval_sharded.sh --variant submission_gptqmodel_full_w4a16 --max-samples 150 --shard-size 30 --force-requant
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --variant)         VARIANT_DIR="$2"; shift 2 ;;
        --model-path)      MODEL_PATH="$2"; shift 2 ;;
        --quant-out)       QUANT_OUT="$2"; shift 2 ;;
        --eval-script)     EVAL_SCRIPT="$2"; shift 2 ;;
        --eval-data)       EVAL_DATA="$2"; shift 2 ;;
        --max-samples)     MAX_SAMPLES="$2"; shift 2 ;;
        --shard-size)      SHARD_SIZE="$2"; shift 2 ;;
        --concurrency)     CONCURRENCY="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
        --no-checkpoint)   CHECKPOINT_AFTER_SHARD=0; shift ;;
        --force-requant)   FORCE_REQUANT=1; shift ;;
        --skip-quant)      SKIP_QUANT=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "${VARIANT_DIR}" ]; then
    echo "error: --variant is required" >&2
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
if [ "${MAX_SAMPLES}" -le 0 ] || [ "${SHARD_SIZE}" -le 0 ]; then
    echo "error: --max-samples and --shard-size must be positive" >&2
    exit 2
fi

VARIANT_NAME="$(basename "${VARIANT_DIR}")"
QUANT_OUT="${QUANT_OUT:-${QUANT_BASE}/${VARIANT_NAME}-quantized}"
PREPARE_ENV="${REPO_ROOT}/${VARIANT_DIR}/prepare_env.sh"
PREPARE_MODEL="${REPO_ROOT}/${VARIANT_DIR}/prepare_model.sh"
if [ ! -f "${PREPARE_ENV}" ]; then
    echo "error: ${PREPARE_ENV} missing" >&2
    exit 2
fi
if [ -f "${PREPARE_MODEL}" ]; then
    QUANT_SCRIPT=""
else
    QUANT_SCRIPT="$(find "${REPO_ROOT}/${VARIANT_DIR}" -maxdepth 1 -name 'quantize_*.py' | sort | head -1)"
fi

SGLANG_SERVER_ARGS="$(grep -E '^export SGLANG_SERVER_ARGS=' "${PREPARE_ENV}" \
    | sed -E 's/^export SGLANG_SERVER_ARGS="//; s/"$//')"

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
            echo "[sharded_eval] sed-patched $(basename "${pyfile}") for fp16 quant run"
        fi
    done
elif echo "${SGLANG_SERVER_ARGS}" | grep -q "gptq_marlin"; then
    echo "[sharded_eval] gptq_marlin + --dtype bfloat16 detected; skipping fp16 sed-patch"
fi

LOG_DIR="${REPO_ROOT}/scripts/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%s)"
SERVER_LOG="${LOG_DIR}/sharded_server_${VARIANT_NAME}_${STAMP}.log"
QUANT_LOG="${LOG_DIR}/sharded_quant_${VARIANT_NAME}_${STAMP}.log"
SHARD_DIR="${LOG_DIR}/shards_${VARIANT_NAME}_${STAMP}"
mkdir -p "${SHARD_DIR}"

echo "=================================================="
echo " local_eval_sharded"
echo "=================================================="
echo " variant:       ${VARIANT_NAME}"
echo " source model:  ${MODEL_PATH}"
echo " quant cache:   ${QUANT_OUT}"
echo " eval script:   ${EVAL_SCRIPT}"
echo " eval data:     ${EVAL_DATA}"
echo " max samples:   ${MAX_SAMPLES}"
echo " shard size:    ${SHARD_SIZE}"
echo " concurrency:   ${CONCURRENCY}"
echo " checkpoint:    ${CHECKPOINT_AFTER_SHARD}"
echo " sglang args:   ${SGLANG_SERVER_ARGS}"
echo " shard csv:     ${RESULTS_CSV}"
echo "=================================================="

START_TS=$(date +%s)

if [ "${SKIP_QUANT}" -eq 1 ]; then
    echo "[1/4] quantize: SKIPPED via --skip-quant"
elif [ "${FORCE_REQUANT}" -eq 0 ] && [ -d "${QUANT_OUT}" ] && [ -f "${QUANT_OUT}/config.json" ]; then
    echo "[1/4] quantize: CACHED at ${QUANT_OUT} (use --force-requant to rebuild)"
else
    rm -rf "${QUANT_OUT}"
    mkdir -p "${QUANT_OUT}"
    if [ -z "${QUANT_SCRIPT}" ]; then
        echo "[1/4] quantize: running ${PREPARE_MODEL}..."
        bash "${PREPARE_MODEL}" \
            --input "${MODEL_PATH}" \
            --output "${QUANT_OUT}" \
            > "${QUANT_LOG}" 2>&1 || {
            echo "[1/4] quantize FAILED. tail of log:" >&2
            tail -50 "${QUANT_LOG}" >&2
            exit 1
        }
    else
        echo "[1/4] quantize: running ${QUANT_SCRIPT}..."
        python3 "${QUANT_SCRIPT}" \
            --input "${MODEL_PATH}" \
            --output "${QUANT_OUT}" \
            > "${QUANT_LOG}" 2>&1 || {
            echo "[1/4] quantize FAILED. tail of log:" >&2
            tail -30 "${QUANT_LOG}" >&2
            exit 1
        }
    fi
    echo "  quantize done. log: ${QUANT_LOG}"
fi

mkdir -p "${TOOLKIT_DIR}"
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "[2/4] downloading eval_model.py..."
    curl -fsSL -o "${EVAL_SCRIPT}" \
        "https://raw.githubusercontent.com/OpenBMB/SOAR-Toolkit/main/eval_model.py"
fi
if [ ! -f "${EVAL_DATA}" ]; then
    echo "[2/4] downloading perf_public_set.jsonl..."
    curl -fsSL -o "${EVAL_DATA}" \
        "https://raw.githubusercontent.com/OpenBMB/SOAR-Toolkit/main/eval_dataset/perf_public_set.jsonl"
fi

echo "[2/4] writing non-overlapping eval shards..."
mapfile -t SHARDS < <(
    python3 - "${EVAL_DATA}" "${SHARD_DIR}" "${MAX_SAMPLES}" "${SHARD_SIZE}" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
max_samples = int(sys.argv[3])
shard_size = int(sys.argv[4])

rows = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
rows = rows[:max_samples]
for shard_idx, start in enumerate(range(0, len(rows), shard_size), 1):
    shard_rows = rows[start:start + shard_size]
    if not shard_rows:
        continue
    end = start + len(shard_rows) - 1
    path = out_dir / f"shard_{shard_idx:03d}_{start:03d}_{end:03d}.jsonl"
    path.write_text("\n".join(shard_rows) + "\n", encoding="utf-8")
    print(f"{shard_idx}\t{start}\t{end}\t{len(shard_rows)}\t{path}")
PY
)
if [ "${#SHARDS[@]}" -eq 0 ]; then
    echo "error: no shards produced from ${EVAL_DATA}" >&2
    exit 1
fi
echo "  wrote ${#SHARDS[@]} shard(s) under ${SHARD_DIR}"

echo "[3/4] launching sglang server..."
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
    for pyfile in "${SED_PATCH_FILES[@]}"; do
        if [ -f "${pyfile}.local_eval.bak" ]; then
            mv "${pyfile}.local_eval.bak" "${pyfile}"
            echo "  reverted fp16 sed patch on $(basename "${pyfile}")"
        fi
    done
}
trap cleanup EXIT INT TERM

echo "  waiting for /health (max ${STARTUP_TIMEOUT}s, log: ${SERVER_LOG})..."
ELAPSED=0
while [ "${ELAPSED}" -lt "${STARTUP_TIMEOUT}" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "  ! server died. tail of log:" >&2
        tail -40 "${SERVER_LOG}" >&2
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "  server up after ${ELAPSED}s"
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

echo "[4/4] running shards..."
ensure_csv_header
CUM_SCORE="0"
CUM_COUNT=0

for shard in "${SHARDS[@]}"; do
    IFS=$'\t' read -r SHARD_INDEX SHARD_START SHARD_END SHARD_COUNT SHARD_PATH <<<"${shard}"
    EVAL_LOG="${LOG_DIR}/sharded_eval_${VARIANT_NAME}_${STAMP}_${SHARD_INDEX}.log"
    SHARD_START_TS=$(date +%s)
    echo "--------------------------------------------------"
    echo "[shard ${SHARD_INDEX}/${#SHARDS[@]}] rows ${SHARD_START}-${SHARD_END} (${SHARD_COUNT} samples)"

    set +e
    python3 "${EVAL_SCRIPT}" \
        --api_base "http://127.0.0.1:${PORT}" \
        --model_path "${QUANT_OUT}" \
        --data_path "${SHARD_PATH}" \
        --concurrency "${CONCURRENCY}" \
        2>&1 | tee "${EVAL_LOG}"
    EVAL_EXIT=${PIPESTATUS[0]}
    set -e

    if [ "${EVAL_EXIT}" -ne 0 ]; then
        echo "  ! eval_model.py exited ${EVAL_EXIT}. tail of log:" >&2
        tail -20 "${EVAL_LOG}" >&2
        exit "${EVAL_EXIT}"
    fi

    METRICS="$(
        python3 - "${REPO_ROOT}" "${EVAL_LOG}" <<'PY'
import json
import re
import sys
from pathlib import Path

repo = Path(sys.argv[1])
log = Path(sys.argv[2])
text = log.read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"Detailed results saved to\s+(\S+)", text)
if not matches:
    raise SystemExit("predictions path not found in eval log")
pred = Path(matches[-1])
if not pred.is_absolute():
    pred = repo / pred
if pred.is_dir():
    pred = pred / "predictions.jsonl"

score_sum = 0.0
count = 0
with pred.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        score_sum += score
        count += 1
acc = (score_sum / count * 100.0) if count else 0.0
print(f"{pred}\t{count}\t{score_sum:.8f}\t{acc:.2f}")
PY
    )"
    IFS=$'\t' read -r PRED_PATH SCORE_COUNT SCORE_SUM SHARD_ACC <<<"${METRICS}"
    ARCHIVED_PRED="${LOG_DIR}/sharded_predictions_${VARIANT_NAME}_${STAMP}_${SHARD_INDEX}.jsonl"
    cp "${PRED_PATH}" "${ARCHIVED_PRED}"
    PRED_PATH="${ARCHIVED_PRED}"
    CUM_SCORE="$(python3 - "${CUM_SCORE}" "${SCORE_SUM}" <<'PY'
import sys
print(f"{float(sys.argv[1]) + float(sys.argv[2]):.8f}")
PY
)"
    CUM_COUNT=$((CUM_COUNT + SCORE_COUNT))
    CUM_ACC="$(python3 - "${CUM_SCORE}" "${CUM_COUNT}" <<'PY'
import sys
score = float(sys.argv[1])
count = int(sys.argv[2])
print(f"{(score / count * 100.0) if count else 0.0:.2f}")
PY
)"
    SHARD_END_TS=$(date +%s)
    SHARD_DURATION=$((SHARD_END_TS - SHARD_START_TS))
    TOTAL_DURATION=$((SHARD_END_TS - START_TS))

    echo "[shard ${SHARD_INDEX}] shard_acc=${SHARD_ACC} cumulative_acc=${CUM_ACC} cumulative_samples=${CUM_COUNT}"
    echo "$(date '+%F %T'),${VARIANT_NAME},${QUANT_OUT},${SHARD_INDEX},${SHARD_START},${SHARD_END},${SCORE_COUNT},${CUM_COUNT},${CONCURRENCY},${SHARD_ACC},${CUM_ACC},${SHARD_DURATION},${TOTAL_DURATION},${SERVER_LOG},${EVAL_LOG},${PRED_PATH}" >> "${RESULTS_CSV}"

    if [ "${CHECKPOINT_AFTER_SHARD}" = "1" ] && [ -x "${CHECKPOINT_SCRIPT}" ]; then
        echo "[shard ${SHARD_INDEX}] checkpointing partial results"
        if ! RUN_CHECKS=0 MESSAGE="checkpoint full w4a16 shard ${SHARD_INDEX} cumulative ${CUM_ACC}" \
                bash "${CHECKPOINT_SCRIPT}"; then
            echo "[shard ${SHARD_INDEX}] WARNING: checkpoint failed; continuing eval" >&2
        fi
    fi
done

echo "=================================================="
echo " SHARDED RESULT"
echo "   cumulative acc_ori: ${CUM_ACC}"
echo "   cumulative samples: ${CUM_COUNT}"
echo "   results csv:        ${RESULTS_CSV}"
echo "   server log:         ${SERVER_LOG}"
echo "=================================================="
