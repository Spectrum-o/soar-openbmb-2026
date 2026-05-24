#!/usr/bin/env bash
# scripts/gpu_smoke_remaining.sh
#
# Autonomous orchestrator for the 3 variants remaining after the
# 2026-05-24 chunk32k_safe + fp8kv runs:
#   1. lightning_skip_bf16  (acc-axis; overlay × bf16 path)
#   2. chunk32k_opfusion    (decode perf; op-fusion overlay)
#   3. chunk65k_safe        (chunk32k_safe + bigger chunk)
#
# Bakes in 4 lessons from earlier sessions today:
#   - Must `source sglang_minicpm_sala_env/bin/activate` (script does it)
#   - NO_PROXY=127.0.0.1,localhost for /health (AutoDL turbo proxy)
#   - --startup-timeout 1800 (cold JIT compile takes ~19 min first time)
#   - Reuse chunk32k_safe artifact via symlink where compatible
#     (opfusion + chunk65k_safe — same quantize_gptqmodel_w4a16.py).
#     lightning_skip_bf16 needs its own quant + overlay → --force-requant.
#
# After each variant: stage all relevant outputs, pull --rebase, push.
# On per-variant failure: continue queue (don't block). Final SUMMARY.md
# commit at end.
#
# Run autonomously (Claude unavailable):
#   cd /root/autodl-tmp/zyn/sglang
#   nohup bash scripts/gpu_smoke_remaining.sh > /tmp/remaining_smoke.log 2>&1 &
#   echo "orchestrator PID: $!"
#
# Total wall time estimate: ~80 min for the 2 cached variants, ~50 min for
# lightning_skip_bf16's full requant + overlay = ~2h serial.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate venv. If already activated (VIRTUAL_ENV set), skip.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/sglang_minicpm_sala_env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/sglang_minicpm_sala_env/bin/activate"
fi

# AutoDL turbo proxy intercepts localhost HTTP. /health and eval_model.py
# both need direct localhost access. Scoped to this script's children.
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

TS="$(date +%Y%m%d_%H%M%S)"
NAS_DIR="/root/autodl-fs/zyn/logs/gpu_remaining_${TS}"
mkdir -p "$NAS_DIR" 2>/dev/null || NAS_DIR="/tmp/gpu_remaining_${TS}"
mkdir -p "$NAS_DIR"

MASTER_LOG="$NAS_DIR/master.log"
SUMMARY_MD="$NAS_DIR/SUMMARY.md"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "============================================================"
echo "GPU smoke remaining — $TS"
echo "log dir: $NAS_DIR"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-(none)}"
echo "NO_PROXY=${NO_PROXY}"
echo "============================================================"

NUM_SAMPLES="${NUM_SAMPLES:-30}"
SHARED_QUANT_DIR="/root/autodl-fs/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized"

# Variants in priority order. Each entry:
#   variant_dir | reuse_artifact (yes/no) | timeout | description
VARIANTS=(
    "submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk65k_safe|yes|1800|chunk65k+0.70 (verify mem-frac+bigger chunk)"
    "submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion|yes|1800|chunk32k+op-fusion overlay (decode perf)"
    "submission_w4a16_lightning_skip_bf16|no|1800|lightning_skip overlay × bf16 path (acc-axis; needs full requant + overlay)"
)

# Initialize SUMMARY.md
{
    echo "# GPU Remaining Smoke — $TS"
    echo ""
    echo "Autonomous run of 3 variants left after chunk32k_safe + fp8kv (2026-05-24)."
    echo ""
    echo "- Baseline: v5j_dtype_bf16 platform acc_ori = 82.18, final_score = 23.18"
    echo "- chunk32k_safe local smoke (this session): acc_ori = 78.67"
    echo "- chunk32k_fp8kv: CRASH (FlashAttention rejects fp8 K/V; Path D insufficient)"
    echo ""
    echo "## Results"
    echo ""
    echo "| Variant | Status | acc_ori | duration | Notes |"
    echo "|---|---|---|---|---|"
} > "$SUMMARY_MD"

# ----------------------------------------------------------------------
# Helper: stage + pull --rebase + push. Patterned after
# scripts/experiment_5h/commit_push.sh (which handles the same divergence
# problem we just fixed in scripts/watchdog_commit.sh).
# ----------------------------------------------------------------------
commit_and_push() {
    local msg="$1"
    # Stage everything that might have changed for this smoke. Be explicit;
    # never use `git add -A` (would pick up tarballs, quant artifacts, etc.).
    git add -f scripts/eval_results.csv 2>/dev/null || true
    git add -f scripts/logs/local_eval_*.log 2>/dev/null || true
    git add -f outputs/ 2>/dev/null || true
    git add -f "$SUMMARY_MD" 2>/dev/null || true

    if git diff --cached --quiet 2>/dev/null; then
        echo "[commit_and_push] nothing to commit"
        return 0
    fi

    git commit --no-verify -m "${msg}

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3

    # Pull --rebase BEFORE push to survive non-fast-forward (origin advances
    # if a parallel session pushed). Abort rebase on conflict (leaves tree
    # usable; next call retries).
    if ! git pull --rebase --no-edit 2>/tmp/.remaining_push_err; then
        git rebase --abort 2>/dev/null || true
        echo "[commit_and_push] PULL --REBASE FAILED (conflict?). commit stays local: $(tail -1 /tmp/.remaining_push_err)"
        return 1
    fi
    if git push 2>/tmp/.remaining_push_err; then
        echo "[commit_and_push] pushed: ${msg}"
        return 0
    else
        echo "[commit_and_push] PUSH FAILED: $(tail -1 /tmp/.remaining_push_err)"
        return 1
    fi
}

# ----------------------------------------------------------------------
# Per-variant runner
# ----------------------------------------------------------------------
run_one() {
    local variant="$1"
    local reuse_artifact="$2"
    local timeout="$3"
    local desc="$4"
    local short="${variant#submission_}"
    local variant_log="$NAS_DIR/${short}.log"

    echo ""
    echo "============================================================"
    echo "[$(date '+%F %T')] SMOKE: $short"
    echo "  desc:     $desc"
    echo "  reuse:    $reuse_artifact"
    echo "  timeout:  ${timeout}s"
    echo "  log:      $variant_log"
    echo "============================================================"

    if [ ! -d "$REPO_ROOT/$variant" ]; then
        echo "  [SKIP] variant dir not found"
        echo "| $short | SKIP (no dir) | — | — | dir missing |" >> "$SUMMARY_MD"
        commit_and_push "smoke remaining: $short SKIPPED (no dir)" || true
        return 0
    fi

    # Artifact reuse via symlink (saves ~30min of redundant requant for the
    # variants that share submission_gptqmodel_calib_w4a16/quantize_*.py).
    local quant_target="/root/autodl-fs/zyn/models/${variant}-quantized"
    local force_arg=""
    if [ "$reuse_artifact" = "yes" ]; then
        if [ -d "$SHARED_QUANT_DIR" ] && [ -f "$SHARED_QUANT_DIR/config.json" ]; then
            # Replace any stale dir/symlink with a fresh symlink to chunk32k_safe's artifact.
            rm -rf "$quant_target"
            ln -sfn "$SHARED_QUANT_DIR" "$quant_target"
            echo "  [reuse] symlinked $quant_target -> $SHARED_QUANT_DIR"
        else
            echo "  [reuse] WARN: shared artifact missing; falling back to --force-requant"
            force_arg="--force-requant"
        fi
    else
        force_arg="--force-requant"
        echo "  [requant] forcing requant for $short"
    fi

    local start
    start=$(date +%s)
    # Don't crash master on smoke failure (set -e is off here anyway, but
    # explicit || true to be safe).
    bash "$REPO_ROOT/scripts/local_eval.sh" \
        --variant "$variant" \
        --num-samples "$NUM_SAMPLES" \
        --startup-timeout "$timeout" \
        $force_arg > "$variant_log" 2>&1 || true
    local dur=$(( $(date +%s) - start ))

    # Mirror the variant log to repo so it can be committed.
    mkdir -p "$REPO_ROOT/scripts/logs/remaining_${TS}"
    cp -f "$variant_log" "$REPO_ROOT/scripts/logs/remaining_${TS}/${short}.log" 2>/dev/null || true

    # Parse outcome
    local status="UNKNOWN" acc_ori="" note=""
    acc_ori=$(grep -oE 'acc_ori \(raw on dataset\):[[:space:]]+[0-9]+\.[0-9]+' "$variant_log" \
              | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)
    [ -z "$acc_ori" ] && acc_ori=$(grep -oE 'Average Score:[[:space:]]+[0-9]+\.[0-9]+' "$variant_log" \
              | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)

    if grep -qE "FlashAttention only support fp16 and bf16|RuntimeError.*dtype|dtype mismatch" "$variant_log"; then
        status="CRASH_dtype"
        note="dtype mismatch (FA)"
    elif grep -qE "CUDA out of memory|OutOfMemoryError" "$variant_log"; then
        status="CRASH_oom"
        note="CUDA OOM"
    elif grep -qE "startup timeout" "$variant_log"; then
        status="CRASH_startup"
        note="server didn't reach /health in time"
    elif grep -qiE "Traceback|^Error|FATAL|Killed " "$variant_log"; then
        status="CRASH_other"
        note=$(grep -E "Error|FATAL|Exception" "$variant_log" | tail -1 | cut -c-80 | tr -d '|')
    elif [ -n "$acc_ori" ]; then
        if awk "BEGIN { exit !($acc_ori >= 78) }"; then
            status="OK"
        elif awk "BEGIN { exit !($acc_ori >= 60) }"; then
            status="OK_acc_low"
            note="below v5j_dtype_bf16 baseline (82.18)"
        else
            status="acc_BAD"
            note="acc far below baseline"
        fi
    else
        status="UNKNOWN"
        note="no acc + no clear error"
    fi

    echo "  done: status=$status acc_ori=${acc_ori:-—} duration=${dur}s"
    echo "| $short | $status | ${acc_ori:-—} | ${dur}s | $note |" >> "$SUMMARY_MD"

    commit_and_push "smoke remaining: $short ($status, acc_ori=${acc_ori:-—})" || true
}

# ----------------------------------------------------------------------
# Run queue
# ----------------------------------------------------------------------
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r variant reuse_artifact timeout desc <<< "$entry"
    run_one "$variant" "$reuse_artifact" "$timeout" "$desc"
done

# ----------------------------------------------------------------------
# Final SUMMARY enrichment + commit
# ----------------------------------------------------------------------
{
    echo ""
    echo "## Recommendations"
    echo ""
    # Walk results, find best OK acc_ori
    best_acc="0"
    best_name=""
    while IFS='|' read -r _ short status acc_col _ _ _; do
        short="$(echo "$short" | xargs)"
        status="$(echo "$status" | xargs)"
        acc_col="$(echo "$acc_col" | xargs)"
        if [[ "$status" == OK* ]] && [ "$acc_col" != "—" ] && [ -n "$acc_col" ]; then
            if awk "BEGIN { exit !($acc_col > $best_acc) }"; then
                best_acc="$acc_col"
                best_name="$short"
            fi
        fi
    done < <(grep "^| " "$SUMMARY_MD" | tail -n +3)

    if [ -n "$best_name" ]; then
        echo "Best smoke result: \`$best_name\` (acc_ori $best_acc)."
        echo "Combined with chunk32k_safe (78.67), the safest platform candidate is"
        echo "whichever of {chunk32k_safe, $best_name} has the higher acc."
    else
        echo "No variant passed smoke cleanly. Submit \`chunk32k_safe\` (78.67) blindly."
    fi
} >> "$SUMMARY_MD"

echo ""
echo "============================================================"
echo "ALL SMOKES COMPLETE — $(date '+%F %T')"
echo "SUMMARY: $SUMMARY_MD"
echo "============================================================"
cat "$SUMMARY_MD"

# Mirror SUMMARY into repo for git tracking
mkdir -p "$REPO_ROOT/scripts/logs/remaining_${TS}"
cp -f "$SUMMARY_MD" "$REPO_ROOT/scripts/logs/remaining_${TS}/SUMMARY.md" 2>/dev/null || true
git add -f "$REPO_ROOT/scripts/logs/remaining_${TS}/" 2>/dev/null || true
commit_and_push "smoke remaining COMPLETE — SUMMARY.md ($TS)" || true
