#!/usr/bin/env bash
# scripts/gpu_smoke_all_pending.sh
#
# Master orchestrator for evening 2026-05-24 GPU session.
# Sequentially smokes all pending variants (30 samples each), records
# acc + crash status, mirrors logs to NAS, generates SUMMARY.md.
#
# Run order (by EV / smoke priority):
#   1. chunk32k_safe       (verify 0.70 mem-frac + 32K combo; 90% expected OK)
#   2. chunk32k_fp8kv      (high-risk FP8 KV via Path B patch; 50% expected OK)
#   3. lightning_skip_bf16 (overlay × bf16; 75% expected OK)
#   4. chunk32k_opfusion   (GPU op-fusion correctness; 70% expected OK)
#   5. chunk65k_safe       (verify 0.70 mem-frac + 65K combo; 80% expected OK)
#
# Total wall time: ~4-5 hours serial. Run nohup so it survives SSH disconnect.
#
# Usage on AutoDL GPU box:
#   cd /root/autodl-tmp/zyn/sglang
#   git pull origin quant/w4a16
#   source sglang_minicpm_sala_env/bin/activate
#   nohup bash scripts/gpu_smoke_all_pending.sh > /tmp/master_smoke.log 2>&1 &
#   echo "master PID: $!"
#   tail -f /tmp/master_smoke.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
NAS_DIR="${NAS_DIR:-/root/autodl-fs/zyn/logs/gpu_session_evening_${TS}}"
mkdir -p "$NAS_DIR" 2>/dev/null || NAS_DIR="/tmp/gpu_session_evening_${TS}"
mkdir -p "$NAS_DIR"

MASTER_LOG="$NAS_DIR/master.log"
SUMMARY_MD="$NAS_DIR/SUMMARY.md"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "============================================================"
echo "GPU smoke all pending — $TS"
echo "log dir: $NAS_DIR"
echo "============================================================"

# Variants to smoke, in priority order. Each entry: VARIANT_DIR:DESCRIPTION
#
# 2026-05-24 v2: chunk32k_fp8kv RE-ADDED after SOAR official toolkit
# guidance (soar.openbmb.cn/toolkit) confirmed FP8 KV is the canonical
# recommended stack. Previous v1 (fp8_e4m3 + Path B minicpm_backend.py
# patch) was wrong on dtype + wrong patch site. v2 uses:
#   - fp8_e5m2 (official spec; ±57344 range vs e4m3's ±448)
#   - Path D patch on gptq.py (3-line add RadixAttention → BaseKVCacheMethod)
#   - --dense-as-sparse already in args → dense_len=0 → no compressed_k path
VARIANTS=(
    "submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe:chunk32k+0.70 (the SAFEST option to validate today)"
    "submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv:chunk32k+0.70+FP8 KV via Path D patch (official-recommended; highest EV)"
    "submission_w4a16_lightning_skip_bf16:lightning_skip on bf16 path (acc-axis bet)"
    "submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion:chunk32k+op-fusion overlay (decode +5-10%)"
    "submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk65k_safe:chunk65k+0.70 (after chunk32k_safe validates)"
)

NUM_SAMPLES="${NUM_SAMPLES:-30}"

# Initialize SUMMARY.md
{
    echo "# GPU Smoke Session — $TS"
    echo ""
    echo "All pending variants smoked at $NUM_SAMPLES samples each."
    echo "Log dir: \`$NAS_DIR\`"
    echo ""
    echo "## Results"
    echo ""
    echo "| Variant | Status | acc_ori | acc | duration | Notes |"
    echo "|---|---|---|---|---|---|"
} > "$SUMMARY_MD"

# ----------------------------------------------------------------------
# Per-variant smoke
# ----------------------------------------------------------------------
run_smoke() {
    local variant="$1"
    local desc="$2"
    local short_name="${variant#submission_}"
    local exp_log="$NAS_DIR/${short_name}.log"

    echo ""
    echo "============================================================"
    echo "SMOKE: $short_name"
    echo "  desc: $desc"
    echo "  log:  $exp_log"
    echo "  start: $(date '+%F %T')"
    echo "============================================================"

    if [ ! -d "$REPO_ROOT/$variant" ]; then
        echo "  [SKIP] variant dir not found: $variant"
        echo "| $short_name | SKIP (no dir) | — | — | — | dir missing |" >> "$SUMMARY_MD"
        return 0
    fi

    local start=$(date +%s)
    # Run smoke. Don't crash master on individual smoke failure.
    bash "$REPO_ROOT/scripts/local_eval.sh" \
        --variant "$variant" \
        --num-samples "$NUM_SAMPLES" \
        --force-requant > "$exp_log" 2>&1 || true
    local dur=$(( $(date +%s) - start ))

    # Parse outcome
    local status="UNKNOWN"
    local acc_ori=""
    local acc=""
    local note=""

    if grep -qiE "Traceback|Error|FATAL|Killed" "$exp_log"; then
        if grep -qE "RuntimeError.*dtype|dtype mismatch" "$exp_log"; then
            status="CRASH_dtype"
            note="dtype mismatch crash"
        elif grep -qE "CUDA out of memory|OutOfMemoryError" "$exp_log"; then
            status="CRASH_oom"
            note="CUDA OOM"
        elif grep -qE "AssertionError|FlashAttention only support" "$exp_log"; then
            status="CRASH_assert"
            note="assertion failure"
        else
            status="CRASH_other"
            note=$(grep -E "Error|FATAL" "$exp_log" | tail -1 | cut -c-80)
        fi
    fi

    # Try to extract acc even if it eventually crashed (some samples may have completed)
    acc_ori=$(grep -iE '"acc_ori"|acc_ori\s*[=:]' "$exp_log" | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "")
    acc=$(grep -iE '"acc":|^\s*acc\s*[=:]' "$exp_log" | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "")

    if [ -n "$acc_ori" ] && [ "$status" = "UNKNOWN" ]; then
        # Compare to working baseline of v5j_dtype_bf16 (82.18)
        if (( $(echo "$acc_ori >= 78" | bc -l 2>/dev/null || echo 0) )); then
            status="OK"
        elif (( $(echo "$acc_ori >= 60" | bc -l 2>/dev/null || echo 0) )); then
            status="OK_acc_low"
            note="acc below v5j_dtype_bf16 baseline"
        else
            status="acc_BAD"
            note="acc much lower than baseline"
        fi
    fi

    if [ "$status" = "UNKNOWN" ]; then
        status="UNKNOWN"
        note="no acc + no error pattern"
    fi

    echo "  end:   $(date '+%F %T')   duration: ${dur}s"
    echo "  STATUS: $status  acc_ori=$acc_ori  acc=$acc  note=$note"

    # Append to SUMMARY
    echo "| $short_name | $status | $acc_ori | $acc | ${dur}s | $note |" >> "$SUMMARY_MD"

    # commit + push intermediate state so results survive instance restart
    if [ -x "$REPO_ROOT/scripts/experiment_5h/commit_push.sh" ]; then
        bash "$REPO_ROOT/scripts/experiment_5h/commit_push.sh" \
            "gpu evening smoke: $short_name ($status)" 2>&1 | tail -3 || true
    fi
}

# ----------------------------------------------------------------------
# Run all variants in order
# ----------------------------------------------------------------------
for entry in "${VARIANTS[@]}"; do
    variant="${entry%%:*}"
    desc="${entry#*:}"
    run_smoke "$variant" "$desc"
done

# ----------------------------------------------------------------------
# Final SUMMARY enrichment
# ----------------------------------------------------------------------
{
    echo ""
    echo "## Recommendations"
    echo ""
    echo "(Filled below based on result table — read top-to-bottom)"
    echo ""
    echo "### Next platform submission"
    echo ""

    # Find best-acc variant that didn't crash
    best_acc="0"
    best_name=""
    while IFS='|' read -r _ name status acc_ori_col _ _ _; do
        name="$(echo "$name" | xargs)"
        status="$(echo "$status" | xargs)"
        acc_ori_col="$(echo "$acc_ori_col" | xargs)"
        if [[ "$status" =~ ^OK ]] && [ -n "$acc_ori_col" ] && [ "$acc_ori_col" != "—" ]; then
            if (( $(echo "$acc_ori_col > $best_acc" | bc -l 2>/dev/null || echo 0) )); then
                best_acc="$acc_ori_col"
                best_name="$name"
            fi
        fi
    done < <(grep "^| " "$SUMMARY_MD" | tail -n +3)

    if [ -n "$best_name" ]; then
        echo "**Highest-acc variant that passed smoke: \`$best_name\` (acc_ori $best_acc)**"
        echo ""
        echo "Recommend submitting this to platform next."
    else
        echo "**No variant passed smoke cleanly.** Review individual logs."
        echo "Fall back: submit \`chunk32k_safe\` blind (lowest variance)."
    fi
    echo ""
    echo "### Logs"
    echo ""
    for entry in "${VARIANTS[@]}"; do
        variant="${entry%%:*}"
        short_name="${variant#submission_}"
        echo "- [\`$short_name\`](./${short_name}.log)"
    done
} >> "$SUMMARY_MD"

echo ""
echo "============================================================"
echo "ALL SMOKES COMPLETE"
echo "  total time: $(($(date +%s) - $(stat -c %Y "$NAS_DIR")))s"
echo "  summary:    $SUMMARY_MD"
echo "  copy this back:"
echo "    scp \$AUTODL:$SUMMARY_MD /tmp/  # from dev mirror"
echo "============================================================"
cat "$SUMMARY_MD"

# Final commit+push of SUMMARY
mkdir -p "$REPO_ROOT/scripts/logs/$(basename $NAS_DIR)"
cp "$SUMMARY_MD" "$REPO_ROOT/scripts/logs/$(basename $NAS_DIR)/SUMMARY.md" 2>/dev/null || true
if [ -x "$REPO_ROOT/scripts/experiment_5h/commit_push.sh" ]; then
    bash "$REPO_ROOT/scripts/experiment_5h/commit_push.sh" \
        "gpu evening smoke COMPLETE — see SUMMARY.md" 2>&1 | tail -5 || true
fi
