#!/usr/bin/env bash
# scripts/gpu_session_next.sh
#
# One-shot orchestrator for the next GPU session. Runs Stage 1 (startup
# verifications) immediately, then kicks off Stage 2 (v5j smoke) in the
# background so it can run unattended.
#
# Designed to be invoked when you first SSH onto the GPU box. Stages 3
# and 4 are documented in experiments/GPU_NEXT_SESSION_RUNBOOK.md — they're
# conditional on Stage 2's result and can't be fully scripted.
#
# Usage:
#   cd /root/autodl-tmp/zyn/sglang   # server path
#   git pull origin quant/w4a16
#   source sglang_minicpm_sala_env/bin/activate
#   bash scripts/gpu_session_next.sh
#
# Logs everything to /root/autodl-fs/zyn/logs/gpu_session_<TS>/ for NAS
# persistence (survives instance reboot).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
NAS_LOG_DIR="${NAS_LOG_DIR:-/root/autodl-fs/zyn/logs/gpu_session_${TS}}"
mkdir -p "$NAS_LOG_DIR" 2>/dev/null || {
    NAS_LOG_DIR="/tmp/gpu_session_${TS}"
    mkdir -p "$NAS_LOG_DIR"
    echo "[gpu_session] WARN: NAS not writable, using /tmp" >&2
}

MASTER_LOG="$NAS_LOG_DIR/master.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=========================================="
echo "GPU session next — $TS"
echo "log dir: $NAS_LOG_DIR"
echo "=========================================="
echo

# =============================================================================
# STAGE 1 — Cheap startup verifications (no eval, ~5 min)
# =============================================================================
echo "===== STAGE 1: startup verifications ====="

verify_args() {
    local name="$1"
    local dir="$2"
    local must_contain="$3"   # comma-separated tokens that MUST be in SGLANG_SERVER_ARGS
    local must_not_contain="$4"  # comma-separated tokens that must NOT be in args

    local args
    args=$(grep "export SGLANG_SERVER_ARGS" "$dir/prepare_env.sh" 2>/dev/null | head -1 || echo "")
    if [ -z "$args" ]; then
        echo "  ✗ $name: SGLANG_SERVER_ARGS not found in $dir/prepare_env.sh"
        return 1
    fi
    local ok=1
    IFS=',' read -ra MUST <<< "$must_contain"
    for tok in "${MUST[@]}"; do
        if [[ "$args" != *"$tok"* ]]; then
            echo "  ✗ $name: missing required token '$tok'"
            ok=0
        fi
    done
    IFS=',' read -ra MUSTNOT <<< "$must_not_contain"
    for tok in "${MUSTNOT[@]}"; do
        if [[ "$args" == *"$tok"* ]]; then
            echo "  ✗ $name: forbidden token '$tok' present"
            ok=0
        fi
    done
    if [ "$ok" = "1" ]; then
        echo "  ✓ $name: config OK"
        return 0
    fi
    return 1
}

# 1a. v5h: must have 0.65 mem-fraction + bfloat16 (the OOM-and-acc fix)
verify_args "v5h (bf16_native_chunk32k)" \
    "submission_bf16_native_chunk32k" \
    "mem-fraction-static 0.65,chunked-prefill-size 32768,--dtype bfloat16" \
    "--dtype float16"

# 1b. v5j: must have 0.80 mem-fraction + W4A16
verify_args "v5j (gptqmodel_no_fp16_patch)" \
    "submission_gptqmodel_no_fp16_patch" \
    "mem-fraction-static 0.80,chunked-prefill-size 8192,gptq_marlin,--dtype float16" \
    "65536"

# 1c. v5j_dtype_bf16: must have --dtype bfloat16 not float16
verify_args "v5j_dtype_bf16" \
    "submission_gptqmodel_no_fp16_patch_dtype_bf16" \
    "mem-fraction-static 0.80,gptq_marlin,--dtype bfloat16" \
    "--dtype float16"

# 1d. Examine why AWQ P2 smoke failed rc=1 — read the recent log
echo
echo "----- AWQ P2 failure investigation -----"
P2_LOG=$(ls -t scripts/logs/5h_*_full/P2*.log 2>/dev/null | head -1)
if [ -n "$P2_LOG" ]; then
    echo "log: $P2_LOG"
    echo "last 30 lines:"
    tail -30 "$P2_LOG" | sed 's/^/  /'
    echo
    echo "Traceback/Error grep:"
    grep -iE "traceback|error|fail|crash|exception" "$P2_LOG" 2>/dev/null | head -10 | sed 's/^/  /'
else
    echo "  (no scripts/logs/5h_*_full/P2*.log found — AWQ P2 hasn't run on this instance)"
fi

# =============================================================================
# STAGE 2 — v5j 30-sample smoke (background, ~50 min)
# =============================================================================
echo
echo "===== STAGE 2: v5j smoke (background) ====="

SMOKE_LOG="$NAS_LOG_DIR/v5j_smoke.log"
echo "Launching v5j smoke (30 samples, --force-requant)..."
echo "  log: $SMOKE_LOG"
echo "  monitor: tail -f $SMOKE_LOG"

nohup bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch \
    --num-samples 30 \
    --force-requant \
    > "$SMOKE_LOG" 2>&1 &
SMOKE_PID=$!
echo "$SMOKE_PID" > "$NAS_LOG_DIR/v5j_smoke.pid"
echo "  v5j smoke PID: $SMOKE_PID (saved to $NAS_LOG_DIR/v5j_smoke.pid)"

# =============================================================================
# STAGE 3+4 are MANUAL — see runbook
# =============================================================================
echo
echo "===== Next steps (manual, see runbook) ====="
echo
echo "While v5j smoke runs (~50 min), you can:"
echo "  1. tail -f $SMOKE_LOG       # watch progress"
echo "  2. cat experiments/GPU_NEXT_SESSION_RUNBOOK.md  # decision tree"
echo
echo "When v5j smoke finishes:"
echo "  bash scripts/gpu_session_decide_next.sh   # auto-parses result + recommends next"
echo "  OR manually grep acc_ori in $SMOKE_LOG and consult the runbook."
echo
echo "Master log: $MASTER_LOG"
echo "All artifacts: $NAS_LOG_DIR"
echo "=========================================="
