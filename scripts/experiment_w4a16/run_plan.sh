#!/usr/bin/env bash
# scripts/experiment_w4a16/run_plan.sh
#
# UNATTENDED W4A16 MIGRATION VALIDATOR — 6-PHASE EXTENDED EDITION
#
# Phases (all sequential on single GPU):
#
#   P1 (control)  : submission_bf16_official_args
#                   v5g + official 65K/0.80 serving args (no quant)
#                   ~25 min. Tests: do official serving args preserve acc?
#
#   P2 (main)     : submission_awq_official_args
#                   AWQ via llmcompressor + compressed-tensors + 65K args
#                   ~1h25. PRIMARY W4A16 candidate for today's submission.
#
#   P3 (ablation) : submission_gptq_official_args
#                   GPTQ via llmcompressor + compressed-tensors + 65K args
#                   ~1h25. Tests: official W4A16_README's GPTQ vs our AWQ choice.
#
#   P4 (calib)    : submission_awq_official_args with NUM_CALIB=512
#                   ~1h35. Tests: does more calibration help (F1's +7.5pp hint)?
#                   Reuses AWQ variant, different QUANT_OUT cache.
#
#   P5 (attn)     : submission_awq_official_args with MLP_ONLY=0
#                   ~1h55. Tests: can lightning attention survive 4-bit quant?
#                   Reuses AWQ variant, different QUANT_OUT cache.
#
#   P6 (overlay)  : apply_lightning_skip_overlay.py on P2 artifact
#                   ~30 min (no requant). Tests: lightning-MLPs back to BF16
#                   on top of P2 — best-of-both-worlds attempt.
#
# Smoke mode (--smoke) runs only P1+P2 (~15 min) to validate pipeline.
# P3-P6 require full quant artifacts; smoke would waste them.
#
# Each phase's acc_ori written to scripts/experiment_w4a16/results/<session>.txt.
# After all phases, decide() recommends which variant to pack.
#
# Usage:
#   bash scripts/experiment_w4a16/run_plan.sh                 # full 6-phase ~6h35
#   bash scripts/experiment_w4a16/run_plan.sh --smoke         # pipeline smoke ~15 min
#   bash scripts/experiment_w4a16/run_plan.sh --from 3        # skip P1+P2, start at P3
#   bash scripts/experiment_w4a16/run_plan.sh --num-samples 100 # smaller eval
#
# Resume support: --from N (lowest N that hasn't run)

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
            sed -n 's/^# \?//p' "$0" | head -55
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
    NUM_CALIB=8
    MAX_CALIB_LEN=2048
else
    NUM_SAMPLES="${NUM_SAMPLES_OVERRIDE:-200}"
    NUM_CALIB=256
    MAX_CALIB_LEN=8192
fi

# Quant cache base (each phase that reuses AWQ variant uses a distinct path)
QUANT_BASE="${QUANT_BASE:-/root/autodl-fs/zyn/models}"
BF16_MODEL="${MODEL_PATH:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"

log "======================================================================="
log "W4A16 migration validator (6-phase) starting"
log "  mode      : $( [ "$SMOKE" = 1 ] && echo SMOKE || echo FULL )"
log "  session   : $SESSION_TAG"
log "  log_dir   : $LOG_DIR"
log "  results   : $RESULT_FILE"
log "  samples   : $NUM_SAMPLES per phase (calib=$NUM_CALIB len=$MAX_CALIB_LEN)"
log "  from_phase: $START_PHASE"
log "  smoke mode runs P1+P2 only; full mode runs all 6 phases."
log "======================================================================="

cat > "$RESULT_FILE" <<EOF
# W4A16 migration validation — session $SESSION_TAG
# mode=$TAG samples=$NUM_SAMPLES calib=$NUM_CALIB
# baseline: v5g platform acc_ori=83.04 (chunked-prefill 8192)
# local↔platform delta historically ~-3pp (local UNDERESTIMATES)
# threshold: local acc_ori >= 75 ≈ platform >= 78 (gate-clearing margin)

EOF

# ----- Helpers ------------------------------------------------------------
record_result() {
    local phase="$1" variant="$2" acc_ori="$3" duration_s="$4" note="$5"
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
    local acc
    acc=$(grep -oE 'ori_accuracy[^0-9-]*[0-9]+(\.[0-9]+)?' "$log_file" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
    [ -n "$acc" ] && { echo "$acc"; return; }
    acc=$(grep -oE '"acc_ori"[[:space:]]*:[[:space:]]*[0-9]+(\.[0-9]+)?' "$log_file" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
    [ -n "$acc" ] && { echo "$acc"; return; }
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

# Generic phase runner — runs local_eval.sh and parses acc_ori
run_phase() {
    local phase_num="$1" phase_name="$2" variant="$3" extra_env_vars="$4"
    local extra_eval_args=("${@:5}")
    local phase_log="$LOG_DIR/phase${phase_num}_${phase_name}.log"

    log "----- Phase $phase_num : $phase_name -----"
    log "  variant: $variant"
    log "  log:     $phase_log"
    [ -n "$extra_env_vars" ] && log "  env:     $extra_env_vars"
    log "  eval args: --num-samples $NUM_SAMPLES ${extra_eval_args[*]}"

    local start=$(date +%s)
    (
        # shellcheck disable=SC2086
        eval "export $extra_env_vars"
        bash "$REPO_ROOT/scripts/local_eval.sh" \
            --variant "$variant" \
            --num-samples "$NUM_SAMPLES" \
            "${extra_eval_args[@]}"
    ) > "$phase_log" 2>&1
    local rc=$?
    local duration=$(( $(date +%s) - start ))

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

# Phase 6 — lightning-skip overlay (special: no quant, post-processes P2 artifact)
run_phase_6_lightning_skip() {
    local phase_log="$LOG_DIR/phase6_lightning_skip.log"
    local p2_cache="$QUANT_BASE/submission_awq_official_args-quantized"
    local p6_cache="$QUANT_BASE/submission_awq_official_args_LIGHTNING_SKIP-quantized"

    log "----- Phase 6 : lightning_skip (overlay on P2 artifact) -----"
    log "  source quant: $p2_cache"
    log "  overlay out:  $p6_cache"

    if [ ! -f "$p2_cache/config.json" ]; then
        log "  ✗ Phase 6 SKIPPED: P2 artifact missing at $p2_cache"
        record_result 6 "lightning_skip" "?" "0" "skipped: no P2 artifact"
        commit_progress "phase 6 lightning_skip skipped (no P2)"
        return
    fi
    if [ ! -d "$BF16_MODEL" ]; then
        log "  ✗ Phase 6 SKIPPED: BF16 source missing at $BF16_MODEL"
        record_result 6 "lightning_skip" "?" "0" "skipped: no BF16 model"
        commit_progress "phase 6 lightning_skip skipped (no BF16)"
        return
    fi

    local start=$(date +%s)

    # Step 1: Copy P2 artifact -> P6 cache (overlay modifies in-place)
    log "  [1/3] copying P2 artifact -> P6 cache (overlay modifies in-place)"
    rm -rf "$p6_cache"
    if ! cp -r "$p2_cache" "$p6_cache" > "$phase_log" 2>&1; then
        log "  ✗ copy failed"
        record_result 6 "lightning_skip" "?" "$(( $(date +%s) - start ))" "FAILED at copy"
        commit_progress "phase 6 lightning_skip FAILED at copy"
        return
    fi

    # Step 2: Apply overlay
    log "  [2/3] applying lightning-skip overlay"
    if ! python3 "$REPO_ROOT/tools/apply_lightning_skip_overlay.py" \
        --quantized-dir "$p6_cache" \
        --bf16-dir "$BF16_MODEL" \
        >> "$phase_log" 2>&1; then
        log "  ✗ overlay failed. tail of log:"
        tail -20 "$phase_log" | tee -a "$MASTER_LOG"
        record_result 6 "lightning_skip" "?" "$(( $(date +%s) - start ))" "FAILED at overlay"
        commit_progress "phase 6 lightning_skip FAILED at overlay"
        return
    fi

    # Step 3: Eval — reuse AWQ variant's SGLANG_SERVER_ARGS but with custom QUANT_OUT
    log "  [3/3] eval with AWQ serving args + overlayed model"
    (
        export QUANT_OUT="$p6_cache"
        bash "$REPO_ROOT/scripts/local_eval.sh" \
            --variant submission_awq_official_args \
            --num-samples "$NUM_SAMPLES" \
            --skip-quant
    ) >> "$phase_log" 2>&1
    local rc=$?
    local duration=$(( $(date +%s) - start ))

    local acc_ori
    acc_ori=$(parse_acc_ori_from_log "$phase_log")

    if [ "$rc" -eq 0 ] && [ "$acc_ori" != "?" ]; then
        log "  ✓ Phase 6 OK: acc_ori=$acc_ori"
        record_result 6 "lightning_skip" "$acc_ori" "$duration" "ok"
    elif [ "$rc" -eq 0 ]; then
        log "  ⚠ Phase 6 completed but acc_ori not extracted"
        record_result 6 "lightning_skip" "?" "$duration" "completed_no_acc"
    else
        log "  ✗ Phase 6 FAILED (rc=$rc)"
        record_result 6 "lightning_skip" "?" "$duration" "FAILED rc=$rc"
    fi

    commit_progress "phase 6 lightning_skip acc_ori=$acc_ori"
}

# ============================================================================
# Phase execution
# ============================================================================

# Phase 1: BF16 + official args (always runs unless --from > 1)
if [ "$START_PHASE" -le 1 ]; then
    run_phase 1 "bf16_official_args" \
        "submission_bf16_official_args" \
        "" \
        --force-requant
fi

# Phase 2: AWQ MLP-only + official args (always runs unless --from > 2)
if [ "$START_PHASE" -le 2 ]; then
    run_phase 2 "awq_official_args" \
        "submission_awq_official_args" \
        "NUM_CALIB=$NUM_CALIB MAX_CALIB_LEN=$MAX_CALIB_LEN" \
        --force-requant
fi

# Smoke mode stops here.
if [ "$SMOKE" = 1 ]; then
    log "----- Smoke mode: skipping P3-P6 -----"
else
    # Phase 3: GPTQ via llmcompressor + official args (AWQ vs GPTQ A/B)
    if [ "$START_PHASE" -le 3 ]; then
        run_phase 3 "gptq_official_args" \
            "submission_gptq_official_args" \
            "NUM_CALIB=$NUM_CALIB MAX_CALIB_LEN=$MAX_CALIB_LEN" \
            --force-requant
    fi

    # Phase 4: AWQ + NUM_CALIB=512 (calibration size ablation)
    if [ "$START_PHASE" -le 4 ]; then
        P4_QUANT_OUT="$QUANT_BASE/submission_awq_official_args_CALIB512-quantized"
        run_phase 4 "awq_calib512" \
            "submission_awq_official_args" \
            "NUM_CALIB=512 MAX_CALIB_LEN=$MAX_CALIB_LEN QUANT_OUT=$P4_QUANT_OUT" \
            --force-requant
    fi

    # Phase 5: AWQ + MLP_ONLY=0 (full attention quant ablation)
    if [ "$START_PHASE" -le 5 ]; then
        P5_QUANT_OUT="$QUANT_BASE/submission_awq_official_args_FULL_ATTN-quantized"
        run_phase 5 "awq_full_attn" \
            "submission_awq_official_args" \
            "MLP_ONLY=0 NUM_CALIB=$NUM_CALIB MAX_CALIB_LEN=$MAX_CALIB_LEN QUANT_OUT=$P5_QUANT_OUT" \
            --force-requant
    fi

    # Phase 6: lightning-skip overlay on P2 artifact (special handler)
    if [ "$START_PHASE" -le 6 ]; then
        run_phase_6_lightning_skip
    fi
fi

# ============================================================================
# Decision
# ============================================================================
log "======================================================================="
log "All phases complete. Computing recommendation..."

extract_acc() {
    local phase_num="$1"
    grep -A2 "^phase=${phase_num}\$" "$RESULT_FILE" | grep 'acc_ori' | awk '{print $3}' | head -1
}

P1_ACC=$(extract_acc 1); P1_ACC="${P1_ACC:-?}"
P2_ACC=$(extract_acc 2); P2_ACC="${P2_ACC:-?}"
P3_ACC=$(extract_acc 3); P3_ACC="${P3_ACC:-?}"
P4_ACC=$(extract_acc 4); P4_ACC="${P4_ACC:-?}"
P5_ACC=$(extract_acc 5); P5_ACC="${P5_ACC:-?}"
P6_ACC=$(extract_acc 6); P6_ACC="${P6_ACC:-?}"

decide() {
    python3 - "$P1_ACC" "$P2_ACC" "$P3_ACC" "$P4_ACC" "$P5_ACC" "$P6_ACC" <<'PY'
import sys
p1, p2, p3, p4, p5, p6 = sys.argv[1:7]

def f(x):
    try: return float(x)
    except: return None

f1, f2, f3, f4, f5, f6 = map(f, (p1, p2, p3, p4, p5, p6))

# Map each phase to its corresponding submission variant
PHASE_VARIANTS = {
    1: ("submission_bf16_official_args", "BF16 + 65K official args"),
    2: ("submission_awq_official_args", "AWQ MLP-only + 65K"),
    3: ("submission_gptq_official_args", "GPTQ MLP-only + 65K"),
    4: ("submission_awq_official_args (with NUM_CALIB=512 quant cache)", "AWQ calib=512"),
    5: ("submission_awq_official_args (with MLP_ONLY=0 quant cache)", "AWQ full attention"),
    6: ("submission_awq_official_args + lightning-skip overlay", "AWQ + lightning skip"),
}

quant_phases = [(2, f2), (3, f3), (4, f4), (5, f5), (6, f6)]
valid_quant = [(p, v) for p, v in quant_phases if v is not None]

# Gate check on Phase 1
if f1 is None:
    print("UNDECIDED")
    print("  Phase 1 (BF16 control) failed or didn't run. Check logs.")
elif f1 < 75:
    print("SUBMIT: submission_bf16_native_chunk32k (v5h fallback)")
    print(f"  Reason: P1={p1} < 75 — official serving args BROKE bf16 acc")
    print(f"          Stick with v5g + 32K chunked-prefill (already-packed v5h)")
elif not valid_quant:
    print("SUBMIT: submission_bf16_official_args (BF16 + official, safe upgrade)")
    print(f"  Reason: P1={p1} OK ({p1} >= 75) but no W4A16 phase succeeded")
else:
    best_phase, best_acc = max(valid_quant, key=lambda x: x[1])
    if best_acc >= 75:
        variant, desc = PHASE_VARIANTS[best_phase]
        print(f"SUBMIT: {variant}")
        print(f"  Reason: best quant phase = P{best_phase} ({desc}) acc_ori={best_acc:.2f}")
        print(f"          P1={p1}, P2={p2}, P3={p3}, P4={p4}, P5={p5}, P6={p6}")
        print(f"          Expected platform acc_ori ~{best_acc + 3:.1f} (local underestimates ~3pp)")
    elif best_acc >= 70:
        variant, desc = PHASE_VARIANTS[best_phase]
        print(f"SUBMIT: submission_bf16_official_args (W4A16 marginal)")
        print(f"  Reason: best quant = P{best_phase} ({desc}) acc_ori={best_acc:.2f}, BELOW safe 75")
        print(f"          BF16+official guaranteed to clear gate")
        print(f"          P1={p1}, P2={p2}, P3={p3}, P4={p4}, P5={p5}, P6={p6}")
    else:
        print(f"SUBMIT: submission_bf16_official_args (W4A16 all failed)")
        print(f"  Reason: best quant acc_ori={best_acc:.2f} < 70 — W4A16 path can't keep acc on SALA")
        print(f"          P1={p1}, P2={p2}, P3={p3}, P4={p4}, P5={p5}, P6={p6}")
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
    echo "Phase summary:"
    echo "  P1 (BF16 + 65K official args)      acc_ori = $P1_ACC"
    echo "  P2 (AWQ MLP-only + 65K)            acc_ori = $P2_ACC"
    echo "  P3 (GPTQ MLP-only + 65K)           acc_ori = $P3_ACC"
    echo "  P4 (AWQ MLP-only + 65K + calib512) acc_ori = $P4_ACC"
    echo "  P5 (AWQ FULL ATTN + 65K)           acc_ori = $P5_ACC"
    echo "  P6 (AWQ + lightning-skip overlay)  acc_ori = $P6_ACC"
    echo
    echo "Reminder: v5g platform anchor = 83.04 acc_ori, final_score 15.76"
    echo "          local→platform delta ~+3pp (local underestimates)"
} >> "$RESULT_FILE"

log "======================================================================="
log "Results:    $RESULT_FILE"
log "Master log: $MASTER_LOG"
log "======================================================================="

# Final commit
commit_progress "FINAL: $(echo "$DECISION" | head -1)"

# Display final result file
echo
echo "=================== RESULT FILE ==================="
cat "$RESULT_FILE"
echo "==================================================="
