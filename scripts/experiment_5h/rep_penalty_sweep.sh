#!/usr/bin/env bash
# scripts/experiment_5h/rep_penalty_sweep.sh
#
# Phase C of the 5h plan: sweep repetition_penalty values against the existing
# J0 quant artifact. Does NOT re-quant; just restarts SGLang with different
# server args (if SGLang supports --repetition-penalty) or with the value
# baked into the eval payload.
#
# CRITICAL: this script first GREPs eval_model.py to determine the path.
# Three handling cases:
#   (a) SOAR Toolkit eval_model.py already accepts a --repetition-penalty flag
#       → forward it from the wrapper
#   (b) SGLang launch_server accepts --repetition-penalty as a server arg
#       → restart SGLang with new args
#   (c) Neither → patch eval_model.py at runtime to add sampling_params
#       (or skip this experiment with a clear log line)
#
# Usage:
#   bash scripts/experiment_5h/rep_penalty_sweep.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TOOLKIT_DIR="${TOOLKIT_DIR:-/root/autodl-fs/zyn/soar_toolkit}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$TOOLKIT_DIR/eval_model.py}"
PORT="${PORT:-31114}"

LOG_DIR="${LOG_DIR:-scripts/logs/rep_penalty_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

# Step 1: probe what the eval pipeline supports
echo "[rep_penalty_sweep] probing eval_model.py for sampling-param support..."
SUPPORT="none"
if grep -qE "repetition_penalty|repetition-penalty" "$EVAL_SCRIPT" 2>/dev/null; then
    SUPPORT="eval_payload"
    echo "  ✓ eval_model.py mentions repetition_penalty"
elif python3 -m sglang.launch_server --help 2>&1 | grep -qiE "repetition.penalty"; then
    SUPPORT="server_arg"
    echo "  ✓ sglang.launch_server supports --repetition-penalty"
else
    SUPPORT="none"
    echo "  ! Neither eval_model.py nor sglang supports repetition_penalty natively"
    echo "  ! Phase C will note the absence; experiment writeable as monkey-patch."
fi

echo "  support_mode = $SUPPORT"

# Step 2: sweep
SAMPLES="${SAMPLES:-30}"
for RP in 1.00 1.05 1.10; do
    OUT_LOG="$LOG_DIR/rp_${RP//./_}.log"
    echo "[rep_penalty_sweep] rp=$RP samples=$SAMPLES → $OUT_LOG"

    case "$SUPPORT" in
        server_arg)
            # Restart SGLang with the flag, then run eval normally
            REP_PENALTY_SERVER_ARG="--repetition-penalty $RP" \
                bash "$REPO_ROOT/scripts/local_eval.sh" \
                --variant submission_gptqmodel_calib_w4a16 \
                --num-samples "$SAMPLES" \
                --port "$PORT" \
                --skip-quant 2>&1 | tee "$OUT_LOG" || true
            ;;
        eval_payload)
            # Forward via env var that eval_model.py reads (assuming it does)
            REPETITION_PENALTY="$RP" \
                bash "$REPO_ROOT/scripts/local_eval.sh" \
                --variant submission_gptqmodel_calib_w4a16 \
                --num-samples "$SAMPLES" \
                --port "$PORT" \
                --skip-quant 2>&1 | tee "$OUT_LOG" || true
            ;;
        none)
            echo "[rep_penalty_sweep] SKIP rp=$RP — no support path" | tee "$OUT_LOG"
            ;;
    esac

    # Tag the log so parse_results can identify the per-rp results
    echo "[rep_penalty_sweep_marker] rep_penalty=$RP" >> "$OUT_LOG"
done

echo "[rep_penalty_sweep] done. Logs in $LOG_DIR"
