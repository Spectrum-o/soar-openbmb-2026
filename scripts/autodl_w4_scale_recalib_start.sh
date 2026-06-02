#!/usr/bin/env bash
# AutoDL launcher for the W4 scale recalibration A/B run.
#
# This is a thin wrapper around scripts/w4_scale_recalib_ablation.sh. It chooses
# the known-good non-FP8 W4A16 artifact when --artifact is omitted, records the
# machine/git state, and starts the run either in foreground or via nohup.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
SOURCE_ARTIFACT="${SOURCE_ARTIFACT:-}"
WORK_DIR="${WORK_DIR:-/root/autodl-fs/zyn/w4_scale_recalib_$(date '+%Y%m%d_%H%M%S')}"
VARIANT_DIR="${VARIANT_DIR:-submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120}"
PORT="${PORT:-31111}"
MODULE_REGEX="${MODULE_REGEX:-}"
MODE="fast"
FOREGROUND=0
DRY_RUN=0
RUN_EVAL=1

FAST_EVAL_SAMPLES="${FAST_EVAL_SAMPLES:-60}"
FAST_BENCH_MATRIX="${FAST_BENCH_MATRIX:-decode,16,1024,256,1;concurrent,32,4096,128,8}"
FULL_EVAL_SAMPLES="${FULL_EVAL_SAMPLES:-150}"
FULL_BENCH_MATRIX="${FULL_BENCH_MATRIX:-decode,32,1024,512,1;concurrent,64,4096,256,8;longctx,16,32768,256,4}"

usage() {
    cat >&2 <<EOF
Usage:
  bash scripts/autodl_w4_scale_recalib_start.sh [options]

Options:
  --artifact PATH       Quantized W4A16 artifact. If omitted, auto-detects under /root/autodl-{fs,tmp}.
  --base PATH           BF16 MiniCPM-SALA base model. Default: ${BASE_MODEL}
  --work-dir PATH       Output work dir. Default: ${WORK_DIR}
  --variant DIR         Submission variant used for server args. Default: ${VARIANT_DIR}
  --module-regex REGEX  Recalibrate only matching GPTQ modules.
  --port PORT           SGLang server port. Default: ${PORT}
  --fast                Short direction check. Default.
  --full                Full bench matrix and 150-sample public eval.
  --no-eval             Skip public-set eval and run benchmark only.
  --foreground          Run in the current shell instead of nohup background.
  --dry-run             Print selection/command without launching.
  -h, --help            Show this help.

Typical:
  bash scripts/autodl_w4_scale_recalib_start.sh --fast
  bash scripts/autodl_w4_scale_recalib_start.sh --full --foreground
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --artifact) SOURCE_ARTIFACT="$2"; shift 2 ;;
        --base) BASE_MODEL="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --variant) VARIANT_DIR="$2"; shift 2 ;;
        --module-regex) MODULE_REGEX="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --fast) MODE="fast"; shift ;;
        --full) MODE="full"; shift ;;
        --no-eval) RUN_EVAL=0; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

pick_newest() {
    awk '
      NF >= 2 {
        ts = $1
        sub(/^[^ ]+ /, "", $0)
        if (ts > best_ts) {
          best_ts = ts
          best_path = $0
        }
      }
      END {
        if (best_path != "") print best_path
      }
    '
}

auto_detect_artifact() {
    local candidates preferred
    candidates="$(
        find /root/autodl-fs /root/autodl-tmp -maxdepth 6 -type d \
            -name '*last7attn*rmsopfusion*calib300*quantized*' \
            -printf '%T@ %p\n' 2>/dev/null \
        | grep -vi 'fp8kv' \
        | grep -vi 'descact' \
        | grep -vi 'accmax' \
        || true
    )"
    if [ -z "${candidates}" ]; then
        return 1
    fi
    preferred="$(printf '%s\n' "${candidates}" | grep -i 'multi120' | pick_newest || true)"
    if [ -n "${preferred}" ]; then
        printf '%s\n' "${preferred}"
        return 0
    fi
    printf '%s\n' "${candidates}" | pick_newest
}

if [ -z "${SOURCE_ARTIFACT}" ]; then
    if ! SOURCE_ARTIFACT="$(auto_detect_artifact)"; then
        echo "error: could not auto-detect the best W4A16 artifact" >&2
        echo "hint: pass --artifact /root/autodl-fs/zyn/models/<artifact>" >&2
        exit 2
    fi
fi

if [ ! -d "${SOURCE_ARTIFACT}" ]; then
    echo "error: artifact dir not found: ${SOURCE_ARTIFACT}" >&2
    exit 2
fi
if [ ! -d "${BASE_MODEL}" ]; then
    echo "error: base model dir not found: ${BASE_MODEL}" >&2
    exit 2
fi
if [ ! -f "${REPO_ROOT}/${VARIANT_DIR}/prepare_env.sh" ]; then
    echo "error: variant prepare_env.sh not found: ${REPO_ROOT}/${VARIANT_DIR}/prepare_env.sh" >&2
    exit 2
fi

if [ "${MODE}" = "fast" ]; then
    EVAL_SAMPLES="${EVAL_SAMPLES:-${FAST_EVAL_SAMPLES}}"
    BENCH_MATRIX="${BENCH_MATRIX:-${FAST_BENCH_MATRIX}}"
else
    EVAL_SAMPLES="${EVAL_SAMPLES:-${FULL_EVAL_SAMPLES}}"
    BENCH_MATRIX="${BENCH_MATRIX:-${FULL_BENCH_MATRIX}}"
fi

mkdir -p "${WORK_DIR}"
RUNNER="${WORK_DIR}/run_ablation.sh"
ENV_SNAPSHOT="${WORK_DIR}/env_snapshot.txt"
DRIVER_LOG="${WORK_DIR}/driver.log"

{
    echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "repo=${REPO_ROOT}"
    echo "branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
    echo "commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
    echo "status_short_begin"
    git -C "${REPO_ROOT}" status --short 2>/dev/null || true
    echo "status_short_end"
    echo "mode=${MODE}"
    echo "artifact=${SOURCE_ARTIFACT}"
    echo "base_model=${BASE_MODEL}"
    echo "variant=${VARIANT_DIR}"
    echo "work_dir=${WORK_DIR}"
    echo "port=${PORT}"
    echo "run_eval=${RUN_EVAL}"
    echo "eval_samples=${EVAL_SAMPLES}"
    echo "bench_matrix=${BENCH_MATRIX}"
    echo "module_regex=${MODULE_REGEX}"
    echo "python=$(command -v python3 || true)"
    python3 --version 2>&1 || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia_smi_begin"
        nvidia-smi || true
        echo "nvidia_smi_end"
    else
        echo "nvidia_smi=missing"
    fi
} > "${ENV_SNAPSHOT}"

AB_ARGS=(
    "${REPO_ROOT}/scripts/w4_scale_recalib_ablation.sh"
    --artifact "${SOURCE_ARTIFACT}"
    --base "${BASE_MODEL}"
    --work-dir "${WORK_DIR}"
    --variant "${VARIANT_DIR}"
    --port "${PORT}"
)
if [ "${RUN_EVAL}" -eq 1 ]; then
    AB_ARGS+=(--eval)
fi
if [ -n "${MODULE_REGEX}" ]; then
    AB_ARGS+=(--module-regex "${MODULE_REGEX}")
fi

{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "${REPO_ROOT}"
    printf 'export EVAL_SAMPLES=%q\n' "${EVAL_SAMPLES}"
    printf 'export BENCH_MATRIX=%q\n' "${BENCH_MATRIX}"
    printf 'exec bash'
    for arg in "${AB_ARGS[@]}"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
} > "${RUNNER}"
chmod +x "${RUNNER}"

echo "=================================================="
echo " W4 scale recalibration AutoDL launcher"
echo "=================================================="
echo " mode:       ${MODE}"
echo " artifact:   ${SOURCE_ARTIFACT}"
echo " base:       ${BASE_MODEL}"
echo " variant:    ${VARIANT_DIR}"
echo " work dir:   ${WORK_DIR}"
echo " eval:       ${RUN_EVAL} samples=${EVAL_SAMPLES}"
echo " bench:      ${BENCH_MATRIX}"
echo " runner:     ${RUNNER}"
echo " env:        ${ENV_SNAPSHOT}"
echo "=================================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[dry-run] launch command:"
    printf '%q ' "${RUNNER}"
    printf '\n'
    exit 0
fi

if [ "${FOREGROUND}" -eq 1 ]; then
    exec "${RUNNER}"
fi

nohup "${RUNNER}" > "${DRIVER_LOG}" 2>&1 &
PID="$!"
echo "${PID}" > "${WORK_DIR}/driver.pid"
echo "[launch] started pid=${PID}"
echo "[launch] log: ${DRIVER_LOG}"
echo "[launch] tail with:"
echo "  tail -f ${DRIVER_LOG}"
