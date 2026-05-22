#!/usr/bin/env bash
# scripts/experiment_5h/run_plan.sh
#
# Master orchestration for the 5-hour unattended W4A16 experiment plan.
# Designed to run sequentially on a single GPU, log each step, append a row
# to scripts/eval_results.csv after each experiment, and auto-commit + push
# so results survive the user being asleep.
#
# Usage on AutoDL GPU box:
#   cd /root/autodl-tmp/zyn/sglang
#   git pull origin quant/w4a16
#   source sglang_minicpm_sala_env/bin/activate
#   bash scripts/experiment_5h/run_plan.sh           # full 5h run
#   bash scripts/experiment_5h/run_plan.sh --smoke   # 2-min smoke per experiment
#   bash scripts/experiment_5h/run_plan.sh --from E  # resume from phase E
#
# Plan: see experiments/UNATTENDED_5H_PLAN.md for full design rationale.
#   Phase A — J0 baseline (50 min)
#   Phase B — 5/14 llm-compressor artifact test (15-45 min)
#   Phase C — repetition_penalty sweep, no requant (20 min)
#   Phase D — single-knob ablations × 4 (100 min)
#   Phase E — winner combo full eval (50 min)
#   Phase F — conditional fill-in (25-50 min)

set -uo pipefail   # NOT -e: continue on individual experiment failures

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SMOKE=0
START_PHASE=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --smoke)   SMOKE=1; shift ;;
        --from)    START_PHASE="$2"; shift 2 ;;
        -h|--help) sed -n 's/^# \?//p' "$0" | head -25; exit 0 ;;
        *) echo "unknown: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_TAG="${TIMESTAMP}_$( [ "$SMOKE" = 1 ] && echo smoke || echo full )"
LOG_DIR="$REPO_ROOT/scripts/logs/5h_${SESSION_TAG}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG" >&2; }

log "===================================================================="
log "5h experiment plan starting"
log "  mode      : $( [ "$SMOKE" = 1 ] && echo SMOKE || echo FULL )"
log "  session   : $SESSION_TAG"
log "  log_dir   : $LOG_DIR"
log "  results   : $REPO_ROOT/scripts/eval_results.csv"
log "  start_at  : ${START_PHASE:-A}"
log "===================================================================="

# Smoke samples (low) vs full samples
if [ "$SMOKE" = 1 ]; then
    SAMPLES_FULL=3       # eval samples
    SAMPLES_SMOKE=3
    NUM_CALIB_EXP=8      # quant calibration samples
else
    SAMPLES_FULL=150
    SAMPLES_SMOKE=60
    NUM_CALIB_EXP=256
fi

# ============================================================================
# Helper: run one experiment, log, append CSV, commit+push
# ============================================================================
run_exp() {
    local name="$1"
    local desc="$2"
    shift 2
    local exp_log="$LOG_DIR/${name}.log"

    log "----- ${name} : ${desc} -----"
    log "cmd: $*"

    local start=$(date +%s)
    if [ "$SMOKE" = 1 ]; then
        # In smoke mode, run with a 120s wall-clock cap. Goal is "does it START"
        # not "does it finish". Kill is success-equivalent for smoke.
        timeout --preserve-status 120 "$@" >"$exp_log" 2>&1 || true
        log "  smoke duration: $(( $(date +%s) - start ))s (capped at 120s)"
    else
        if "$@" >"$exp_log" 2>&1; then
            log "  exp $name OK in $(( $(date +%s) - start ))s"
        else
            log "  exp $name FAILED in $(( $(date +%s) - start ))s — continuing"
        fi
    fi

    # Parse + append CSV (best effort, may fail if log has no metrics yet)
    python3 "$REPO_ROOT/scripts/experiment_5h/parse_results.py" \
        --log "$exp_log" --exp "$name" --desc "$desc" \
        --csv "$REPO_ROOT/scripts/eval_results.csv" 2>&1 | tee -a "$MASTER_LOG" || true

    # Commit + push (best effort)
    bash "$REPO_ROOT/scripts/experiment_5h/commit_push.sh" \
        "5h exp $name ($SESSION_TAG)" 2>&1 | tee -a "$MASTER_LOG" || true
}

# Helper: should this phase run? (--from support)
should_run() {
    local phase="$1"
    if [ -z "$START_PHASE" ]; then return 0; fi
    [ "$phase" \> "$START_PHASE" ] || [ "$phase" = "$START_PHASE" ]
}

# ============================================================================
# Phase A — Baseline (J0) — establishes local↔platform anchor
# ============================================================================
if should_run "A"; then
    run_exp "A_J0_baseline" "current 1849 recipe, no env overrides" \
        bash "$REPO_ROOT/scripts/local_eval.sh" \
            --variant submission_gptqmodel_calib_w4a16 \
            --num-samples "$SAMPLES_FULL" \
            --force-requant
fi

# ============================================================================
# Phase B — 5/14 llm-compressor artifact test (zero-quant, eval only)
# ============================================================================
if should_run "B"; then
    if [ -f "$REPO_ROOT/scripts/test_llmcompressor_artifact.sh" ]; then
        run_exp "B_5_14_smoke" "5/14 llm-compressor artifact, 30 samples" \
            bash "$REPO_ROOT/scripts/test_llmcompressor_artifact.sh" \
                --num-samples 30 --port 31112
        # Auto-decide full eval (only in non-smoke mode)
        if [ "$SMOKE" = 0 ]; then
            B_ACC=$(grep -oE "acc[_a-z]*[[:space:]]*[=:][[:space:]]*[0-9.]+" \
                "$LOG_DIR/B_5_14_smoke.log" | head -1 | grep -oE "[0-9.]+" || echo "0")
            if (( $(echo "$B_ACC >= 30" | bc -l 2>/dev/null || echo 0) )); then
                run_exp "B_5_14_full" "5/14 artifact, 150 samples + per-task" \
                    bash "$REPO_ROOT/scripts/test_llmcompressor_artifact.sh" \
                        --num-samples 150 --port 31112
            else
                log "  Phase B smoke = $B_ACC < 30, skipping full"
            fi
        fi
    else
        log "  scripts/test_llmcompressor_artifact.sh missing — skipping Phase B"
        log "  (lives on parallel/non-gptqmodel-paths branch — cherry-pick it)"
    fi
fi

# ============================================================================
# Phase C — repetition_penalty sweep (no requant, reuses J0 artifact)
# ============================================================================
if should_run "C"; then
    if [ -f "$REPO_ROOT/scripts/experiment_5h/rep_penalty_sweep.sh" ]; then
        run_exp "C_rep_pen_sweep" "rep_pen 1.0 / 1.05 / 1.10 sweep on J0 artifact" \
            bash "$REPO_ROOT/scripts/experiment_5h/rep_penalty_sweep.sh"
    else
        log "  rep_penalty_sweep.sh missing — skipping Phase C"
    fi
fi

# ============================================================================
# Phase D — Single-knob ablations (60-sample smoke each, FULL re-quant)
# ============================================================================
# D1: sym=False (asymmetric, production standard)
if should_run "D"; then
    GPTQ_SYM=False run_exp "D1_sym_false" "sym=False (asymmetric)" \
        bash "$REPO_ROOT/scripts/local_eval.sh" \
            --variant submission_gptqmodel_calib_w4a16 \
            --num-samples "$SAMPLES_SMOKE" \
            --force-requant

    # D2: dampening_frac=0.1 (production uses 0.1 not 0.01)
    GPTQ_DAMPENING_FRAC=0.1 run_exp "D2_damp_01" "dampening_frac=0.1" \
        bash "$REPO_ROOT/scripts/local_eval.sh" \
            --variant submission_gptqmodel_calib_w4a16 \
            --num-samples "$SAMPLES_SMOKE" \
            --force-requant

    # D3: num_calib=512 (intermediate — bigger calib than 256, fits in 5h)
    NUM_CALIB=512 run_exp "D3_calib_512" "num_calib=512" \
        bash "$REPO_ROOT/scripts/local_eval.sh" \
            --variant submission_gptqmodel_calib_w4a16 \
            --num-samples "$SAMPLES_SMOKE" \
            --force-requant

    # D4: max_calib_len=16384 (longer per-prompt window)
    MAX_CALIB_LEN=16384 run_exp "D4_calib_len_16k" "max_calib_len=16384" \
        bash "$REPO_ROOT/scripts/local_eval.sh" \
            --variant submission_gptqmodel_calib_w4a16 \
            --num-samples "$SAMPLES_SMOKE" \
            --force-requant
fi

# ============================================================================
# Phase E — Winner combo (read Phase D CSV, stack winners, full eval)
# ============================================================================
if should_run "E"; then
    # Auto-detect winners from CSV
    WINNERS=$(python3 "$REPO_ROOT/scripts/experiment_5h/pick_winners.py" \
        --csv "$REPO_ROOT/scripts/eval_results.csv" \
        --session "$SESSION_TAG" \
        --baseline-exp "A_J0_baseline" \
        --threshold 2.0 2>/dev/null || echo "")
    if [ -n "$WINNERS" ]; then
        log "Phase E winners detected: $WINNERS"
        # WINNERS is a string of env var assignments like "GPTQ_SYM=False GPTQ_DAMPENING_FRAC=0.1"
        eval "env $WINNERS" run_exp "E_winners_combo" "stacked winners: $WINNERS" \
            bash "$REPO_ROOT/scripts/local_eval.sh" \
                --variant submission_gptqmodel_calib_w4a16 \
                --num-samples "$SAMPLES_FULL" \
                --force-requant
    else
        log "Phase E: no clear winners from D (none > baseline + 2pp). Running Phase D combo all-anyway"
        GPTQ_SYM=False GPTQ_DAMPENING_FRAC=0.1 NUM_CALIB=1024 \
            run_exp "E_fallback_all_knobs" "fallback: all 3 safe knobs stacked" \
                bash "$REPO_ROOT/scripts/local_eval.sh" \
                    --variant submission_gptqmodel_calib_w4a16 \
                    --num-samples "$SAMPLES_FULL" \
                    --force-requant
    fi
fi

# ============================================================================
# Phase F — Conditional fill-in (only if time remains)
# ============================================================================
if should_run "F"; then
    # F1: v25_calib_multi_adaptive (ready variant)
    if [ -d "$REPO_ROOT/submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive" ]; then
        run_exp "F1_multi_adaptive" "v25 multi-adaptive calibration windowing" \
            bash "$REPO_ROOT/scripts/local_eval.sh" \
                --variant submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive \
                --num-samples "$SAMPLES_SMOKE" \
                --force-requant
    fi
    # F2: v25_g64 (ready variant)
    if [ -d "$REPO_ROOT/submission_gptqmodel_calib_w4a16_v25_g64" ]; then
        run_exp "F2_g64" "v25 group_size=64" \
            bash "$REPO_ROOT/scripts/local_eval.sh" \
                --variant submission_gptqmodel_calib_w4a16_v25_g64 \
                --num-samples "$SAMPLES_SMOKE" \
                --force-requant
    fi
fi

log "===================================================================="
log "5h plan complete. Results: scripts/eval_results.csv"
log "Master log: $MASTER_LOG"
log "===================================================================="

# Final commit + push to capture the master log
bash "$REPO_ROOT/scripts/experiment_5h/commit_push.sh" \
    "5h plan COMPLETE ($SESSION_TAG)" 2>&1 | tee -a "$MASTER_LOG" || true
