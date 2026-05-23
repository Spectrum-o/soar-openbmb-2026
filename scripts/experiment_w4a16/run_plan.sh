#!/usr/bin/env bash
# scripts/experiment_w4a16/run_plan.sh
#
# UNATTENDED W4A16 MIGRATION VALIDATOR
#
# Two-phase local validation for tonight's W4A16 migration decision.
# Runs sequentially on a single GPU; logs everything; writes a decision
# file when complete. Commits + pushes after each phase so progress
# survives session loss.
#
# Phases:
#   Phase 1 (control): submission_bf16_official_args
#     - v5g (acc_ori=83.04 verified) + official 65K/0.80 serving args
#     - Tests: do the official serving args preserve acc on bf16?
#     - No quantization, fast (~15-20 min for 200 samples)
#
#   Phase 2 (main):    submission_awq_official_args
#     - AWQ via llm-compressor + compressed-tensors loader + official args
#     - Tests: does the W4A16 official path clear platform-equivalent acc?
#     - Quant ~1h (cached on rerun) + eval ~20 min
#
# Each phase writes acc_ori to scripts/experiment_w4a16/results/<session>.txt.
# After both complete, decision.sh recommends which submission to pack.
#
# Usage on AutoDL GPU box:
#   cd /root/autodl-tmp/zyn/sglang
#   git pull origin quant/w4a16
#   source sglang_minicpm_sala_env/bin/activate
#
#   # Quick smoke first (~15-20 min): tiny calib + 5 samples each
#   bash scripts/experiment_w4a16/run_plan.sh --smoke
#
#   # Then long autonomous run (~2-3h): full eval
#   nohup bash scripts/experiment_w4a16/run_plan.sh \
#       > /tmp/w4a16_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Resume support:
#   bash scripts/experiment_w4a16/run_plan.sh --from 2   # skip phase 1

set -uo pipefail   # NOT -e: don't bail mid-phase

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ----- Args ----------------------------------------------------------------
SMOKE=0
START_PHASE=1
NUM_SAMPLES_OVERRIDE=""
NO_COMMIT=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --smoke)         SMOKE=1; shift ;;
        --from)          START_PHASE="$2"; shift 2 ;;
        --num-samples)   NUM_SAMPLES_OVERRIDE="$2"; shift 2 ;;
        --no-commit)     NO_COMMIT=1; shift ;;
        -h|--help)
            sed -n 's/^# \?//p' "$0" | head -40
            exit 0
            ;;
        *)
            echo "unknown: $1" >&2
            exit 1
            ;;
    esac
done

# ----- Session setup -------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TAG="$( [ "$SMOKE" = 1 ] && echo smoke || echo full )"
SESSION_TAG="${TIMESTAMP}_${TAG}"
LOG_DIR="$REPO_ROOT/scripts/logs/w4a16_${SESSION_TAG}"
RESULT_DIR="$REPO_ROOT/scripts/experiment_w4a16/results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"
MASTER_LOG="$LOG_DIR/master.log"
RESULT_FILE="$RESULT_DIR/${SESSION_TAG}.txt"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG" >&2; }

# Sample / calib counts depending on mode
if [ "$SMOKE" = 1 ]; then
    NUM_SAMPLES="${NUM_SAMPLES_OVERRIDE:-5}"
    NUM_CALIB=8        # AWQ calib samples
    MAX_CALIB_LEN=2048
else
    NUM_SAMPLES="${NUM_SAMPLES_OVERRIDE:-200}"
    NUM_CALIB=256      # AWQ calib samples (matches AWQ variant default)
    MAX_CALIB_LEN=8192
fi

log "======================================================================="
log "W4A16 migration validator starting"
log "  mode      : $( [ "$SMOKE" = 1 ] && echo SMOKE || echo FULL )"
log "  session   : $SESSION_TAG"
log "  log_dir   : $LOG_DIR"
log "  results   : $RESULT_FILE"
log "  samples   : $NUM_SAMPLES per phase (calib=$NUM_CALIB len=$MAX_CALIB_LEN)"
log "  from_phase: $START_PHASE"
log "======================================================================="

# Initialize results file
cat > "$RESULT_FILE" <<EOF
# W4A16 migration validation — session $SESSION_TAG
# mode=$TAG samples=$NUM_SAMPLES calib=$NUM_CALIB
# baseline reference: v5g platform acc_ori=83.04 (chunked-prefill 8192)
# local↔platform delta historically ~2-3 pp on this dataset

EOF

# ----- Helper: run one phase ------------------------------------------------
record_result() {
    local phase="$1"
    local variant="$2"
    local acc_ori="$3"
    local duration_s="$4"
    local note="$5"
    {
        echo "phase=$phase"
        echo "  variant     = $variant"
        echo "  acc_ori     = $acc_ori"
        echo "  duration_s  = $duration_s"
        echo "  note        = $note"
        echo
    } >> "$RESULT_FILE"
}

parse_acc_ori_from_log() {
    local log_file="$1"
    # Try several patterns. parse_results.py logic is similar but simpler here.
    local acc
    # 1. ori_accuracy: NN.NN
    acc=$(grep -oE 'ori_accuracy[^0-9-]*[0-9]+(\.[0-9]+)?' "$log_file" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
    [ -n "$acc" ] && { echo "$acc"; return; }
    # 2. "acc_ori": NN.NN
    acc=$(grep -oE '"acc_ori"[[:space:]]*:[[:space:]]*[0-9]+(\.[0-9]+)?' "$log_file" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
    [ -n "$acc" ] && { echo "$acc"; return; }
    # 3. Average Score: NN.NN%
    acc=$(grep -oE 'Average Score[^0-9-]*[0-9]+(\.[0-9]+)?' "$log_file" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
    [ -n "$acc" ] && { echo "$acc"; return; }
    echo "?"
}

commit_progress() {
    local msg="$1"
    if [ "$NO_COMMIT" = "1" ]; then
        log "  (skipping commit/push: --no-commit set)"
        return
    fi
    if [ -f "$REPO_ROOT/scripts/experiment_5h/commit_push.sh" ]; then
        bash "$REPO_ROOT/scripts/experiment_5h/commit_push.sh" \
            "w4a16 $msg ($SESSION_TAG)" 2>&1 | tee -a "$MASTER_LOG" || true
    else
        log "  (commit_push.sh missing; skipping)"
    fi
}

run_phase() {
    local phase_num="$1"
    local phase_name="$2"
    local variant="$3"
    local extra_env_vars="$4"   # e.g. "NUM_CALIB=8 MAX_CALIB_LEN=2048"
    local extra_eval_args=("${@:5}")

    local phase_log="$LOG_DIR/phase${phase_num}_${phase_name}.log"

    log "----- Phase $phase_num : $phase_name -----"
    log "  variant: $variant"
    log "  log:     $phase_log"
    if [ -n "$extra_env_vars" ]; then
        log "  env:     $extra_env_vars"
    fi
    log "  eval args: --num-samples $NUM_SAMPLES ${extra_eval_args[*]}"

    local start=$(date +%s)
    # Run local_eval.sh with the variant's env vars
    (
        # shellcheck disable=SC2086
        eval "export $extra_env_vars"
        bash "$REPO_ROOT/scripts/local_eval.sh" \
            --variant "$variant" \
            --num-samples "$NUM_SAMPLES" \
            "${extra_eval_args[@]}"
    ) > "$phase_log" 2>&1
    local rc=$?
    local end=$(date +%s)
    local duration=$((end - start))

    log "  exit code: $rc, duration: ${duration}s"

    local acc_ori
    acc_ori=$(parse_acc_ori_from_log "$phase_log")

    if [ "$rc" -eq 0 ] && [ "$acc_ori" != "?" ]; then
        log "  ✓ Phase $phase_num OK: acc_ori=$acc_ori"
        record_result "$phase_num" "$variant" "$acc_ori" "$duration" "ok"
    elif [ "$rc" -eq 0 ]; then
        log "  ⚠ Phase $phase_num completed but acc_ori not extracted"
        record_result "$phase_num" "$variant" "?" "$duration" "completed_no_acc"
    else
        log "  ✗ Phase $phase_num FAILED (rc=$rc). tail of log:"
        tail -30 "$phase_log" | tee -a "$MASTER_LOG"
        record_result "$phase_num" "$variant" "?" "$duration" "FAILED rc=$rc"
    fi

    commit_progress "phase $phase_num $phase_name acc_ori=$acc_ori"
}

# ============================================================================
# Phase 1 — BF16 + official serving args (control)
# ============================================================================
if [ "$START_PHASE" -le 1 ]; then
    # Force a rebuild of the bf16-config-fixed model dir each time (cheap, <5s).
    # --force-requant on a no-quant prepare_model.sh just re-runs the config.json
    # rewrite, which is what we want to ensure cleanliness.
    run_phase 1 "bf16_official_args" \
        "submission_bf16_official_args" \
        "" \
        --force-requant
fi

# ============================================================================
# Phase 2 — AWQ via llm-compressor + official serving args (W4A16 main)
# ============================================================================
if [ "$START_PHASE" -le 2 ]; then
    # NUM_CALIB / MAX_CALIB_LEN are consumed by prepare_model.sh (AWQ variant).
    # MLP_ONLY is left at its default (1) for safety; if Phase 2 acc is great,
    # follow-up could try MLP_ONLY=0. AWQ_SCHEME stays at the variant default
    # (W4A16_ASYM).
    if [ "$SMOKE" = 1 ]; then
        # Smoke: tiny calib, force a fresh quant so the smoke validates the
        # whole pipeline, not a cached artifact from a different run.
        run_phase 2 "awq_official_args" \
            "submission_awq_official_args" \
            "NUM_CALIB=$NUM_CALIB MAX_CALIB_LEN=$MAX_CALIB_LEN" \
            --force-requant
    else
        # Full: reuse cached quant if present (saves 1h on re-run). The
        # quant dir is hashed by variant name, so the smoke artifact (tiny
        # calib) IS different from the full artifact (256 calib). Force a
        # rebuild for full mode to avoid using the smoke artifact.
        run_phase 2 "awq_official_args" \
            "submission_awq_official_args" \
            "NUM_CALIB=$NUM_CALIB MAX_CALIB_LEN=$MAX_CALIB_LEN" \
            --force-requant
    fi
fi

# ============================================================================
# Decision
# ============================================================================
log "======================================================================="
log "All phases complete. Computing recommendation..."

PHASE1_ACC=$(grep -A2 'phase=1$' "$RESULT_FILE" | grep 'acc_ori' | awk '{print $3}' | head -1)
PHASE2_ACC=$(grep -A2 'phase=2$' "$RESULT_FILE" | grep 'acc_ori' | awk '{print $3}' | head -1)

PHASE1_ACC="${PHASE1_ACC:-?}"
PHASE2_ACC="${PHASE2_ACC:-?}"

decide() {
    # Numeric comparison helper
    python3 - "$PHASE1_ACC" "$PHASE2_ACC" <<'PY'
import sys
p1 = sys.argv[1]
p2 = sys.argv[2]

def to_float(x):
    try: return float(x)
    except: return None

f1, f2 = to_float(p1), to_float(p2)

# Decision tree:
#   IF phase1 < 78 -> official args are BROKEN; submit v5h (chunked-prefill 32K) instead
#   ELSE IF phase2 >= 78 -> W4A16 + official args (highest EV)
#   ELSE IF phase2 70..77 -> judgment call; default to BF16 + official args (safer)
#   ELSE phase2 < 70 -> W4A16 broken; submit BF16 + official args

if f1 is None and f2 is None:
    print("UNDECIDED")
    print("  Both phases failed to parse acc_ori. Read the logs manually.")
elif f1 is None or f1 < 78:
    print("SUBMIT: submission_bf16_native_chunk32k (v5h fallback)")
    print(f"  Reason: phase1 acc={p1} < 78, official serving args BREAK acc")
    print(f"          v5h (chunked-prefill 32K) is the safer step from v5g")
elif f2 is not None and f2 >= 78:
    print("SUBMIT: submission_awq_official_args (W4A16 + official, MAIN target)")
    print(f"  Reason: phase1={p1} (control OK) AND phase2={p2} >= 78")
    print(f"          W4A16 path validated; expected platform acc_ori ~{f2-2.5:.1f}")
elif f2 is not None and f2 >= 70:
    print("SUBMIT: submission_bf16_official_args (safe upgrade from v5g)")
    print(f"  Reason: phase1={p1} OK, phase2={p2} marginal (70-78)")
    print(f"          BF16+official args = guaranteed acc-preserving throughput bump")
elif f2 is not None:
    print("SUBMIT: submission_bf16_official_args")
    print(f"  Reason: phase1={p1} OK but phase2={p2} < 70 — W4A16 path BROKE acc")
    print(f"          Stick with BF16 family; build GPTQ-via-llmcompressor next time")
else:
    print("SUBMIT: submission_bf16_official_args")
    print(f"  Reason: phase1={p1} OK but phase2 failed to run/parse")
    print(f"          Default to safe BF16+official args")
PY
}

DECISION=$(decide)
log "$DECISION"

{
    echo
    echo "DECISION"
    echo "========"
    echo "$DECISION"
    echo
    echo "Local validation thresholds reminder:"
    echo "  acc_ori >= 78 (local) → expected acc_ori ~75 (platform; -2.5pp slack)"
    echo "  v5g (platform anchor) = acc_ori 83.04"
} >> "$RESULT_FILE"

log "======================================================================="
log "Results:    $RESULT_FILE"
log "Master log: $MASTER_LOG"
log "======================================================================="

# Final commit
commit_progress "FINAL: $DECISION"

# Cat the result file at the very end for convenience
echo
echo "=================== RESULT FILE ==================="
cat "$RESULT_FILE"
echo "==================================================="
