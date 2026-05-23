#!/usr/bin/env bash
# scripts/gpu_session_decide_next.sh
#
# Reads the most recent v5j smoke log, classifies the outcome, and prints
# the recommended next experiment per the runbook decision tree.
#
# Run this AFTER gpu_session_next.sh's Stage 2 finishes.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Find most recent v5j smoke log (try NAS first, then /tmp)
SMOKE_LOG=""
for cand in $(ls -t /root/autodl-fs/zyn/logs/gpu_session_*/v5j_smoke.log /tmp/gpu_session_*/v5j_smoke.log 2>/dev/null); do
    SMOKE_LOG="$cand"
    break
done

if [ -z "$SMOKE_LOG" ] || [ ! -f "$SMOKE_LOG" ]; then
    echo "[decide_next] no v5j smoke log found"
    exit 2
fi

echo "[decide_next] reading: $SMOKE_LOG"
echo

# Classify outcome
ACC_ORI=""
CRASHED=0

if grep -qE "RuntimeError.*dtype|AssertionError.*dtype|dtype mismatch" "$SMOKE_LOG"; then
    CRASHED=1
    echo "## OUTCOME: dtype mismatch crash"
elif grep -qE "Traceback|CUDA out of memory|RuntimeError" "$SMOKE_LOG"; then
    CRASHED=2
    echo "## OUTCOME: crashed (non-dtype)"
fi

# Try to extract acc_ori
for pat in '"acc_ori"\s*:\s*([0-9.]+)' '^\s*acc_ori\s*[=:]\s*([0-9.]+)' '\bacc_ori\s*[=:]\s*([0-9.]+)'; do
    m=$(grep -oE "$pat" "$SMOKE_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+" || true)
    if [ -n "$m" ]; then
        ACC_ORI="$m"
        break
    fi
done

echo
if [ "$CRASHED" -ne 0 ]; then
    echo "================================================================"
    echo "## v5j CRASHED — Path 1 (gptqmodel + Marlin + dtype fp16) DEAD."
    echo "================================================================"
    if [ "$CRASHED" = "1" ]; then
        echo
        echo "Specifically dtype mismatch. SGLang's auto-cast at Marlin→sparse"
        echo "boundary is NOT working. Try v5j_dtype_bf16 (which uses --dtype"
        echo "bfloat16 explicitly), it might sidestep the boundary issue:"
        echo
        echo "  bash scripts/local_eval.sh \\"
        echo "      --variant submission_gptqmodel_no_fp16_patch_dtype_bf16 \\"
        echo "      --num-samples 30 --force-requant"
    else
        echo
        echo "Some other crash. Read $SMOKE_LOG tail to identify."
        echo "Probably the bug is in our quant-prep, not the dtype path."
    fi
    exit 1
fi

if [ -z "$ACC_ORI" ]; then
    echo "[decide_next] could not extract acc_ori from log. Manual review needed."
    echo "Last 30 lines:"
    tail -30 "$SMOKE_LOG" | sed 's/^/  /'
    exit 2
fi

echo "Detected acc_ori = $ACC_ORI"
echo

# Decision tree per runbook
threshold_high=70
threshold_partial_lo=50
threshold_partial_hi=70

if (( $(echo "$ACC_ORI >= $threshold_high" | bc -l 2>/dev/null || echo 0) )); then
    echo "================================================================"
    echo "## v5j WORKS (acc_ori >= 70) — Path 1 CONFIRMED"
    echo "================================================================"
    echo
    echo "Action:"
    echo "  1. Pack v5j as platform submission:"
    echo "       python3 tools/pack_submission.py \\"
    echo "           --variant submission_gptqmodel_no_fp16_patch \\"
    echo "           --suffix _v5j --output-dir ."
    echo "  2. ALSO build a v5j_chunk32k variant (v5j + 32K chunked-prefill)"
    echo "     for the next slot. v5j proved bf16 works, now stack throughput."
    echo "  3. Submit v5j to platform with confidence."
    echo
    echo "Expected platform: acc_ori ~83 (matches v5g), final_score 20-25"
    echo "(slightly higher than v5g because W4A16 is 4× decode bandwidth save)."
elif (( $(echo "$ACC_ORI >= $threshold_partial_lo" | bc -l 2>/dev/null || echo 0) )); then
    echo "================================================================"
    echo "## v5j PARTIAL (acc_ori in [50, 70]) — sed-patch is most of"
    echo "## the problem, but --dtype float16 still hurts"
    echo "================================================================"
    echo
    echo "Action: smoke v5j_dtype_bf16 (drops both sed-patch AND switches dtype):"
    echo
    echo "  bash scripts/local_eval.sh \\"
    echo "      --variant submission_gptqmodel_no_fp16_patch_dtype_bf16 \\"
    echo "      --num-samples 30 --force-requant > /tmp/v5j_bf16_smoke.log 2>&1 &"
    echo
    echo "Expected: acc_ori >= 75 if --dtype bfloat16 closes the remaining gap."
else
    echo "================================================================"
    echo "## v5j FAILED (acc_ori < 50) — Path 1 (W4A16 Marlin) is dead end"
    echo "================================================================"
    echo
    echo "Even removing sed-patch doesn't help; --dtype float16 in Marlin"
    echo "internals is enough to kill acc. Next options:"
    echo
    echo "  1. Try v5j_dtype_bf16 (low confidence but cheap):"
    echo "       bash scripts/local_eval.sh \\"
    echo "           --variant submission_gptqmodel_no_fp16_patch_dtype_bf16 \\"
    echo "           --num-samples 30 --force-requant"
    echo
    echo "  2. Switch to Path 2 (gptq_official_args via compressed-tensors)"
    echo "     after debugging AWQ P2 failure first."
    echo
    echo "  3. Path 3: modify SGLang source to cast Marlin output to bf16."
    echo
    echo "  See experiments/GPU_NEXT_SESSION_RUNBOOK.md Stage 3 for full"
    echo "  decision tree."
fi
