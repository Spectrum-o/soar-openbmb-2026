#!/usr/bin/env bash
# scripts/gpu_smoke_phase3.sh
#
# Tonight's phase 3 autonomous orchestrator. Runs three independent
# experiments serially, each isolated (apply patch → smoke → revert)
# so a failure in one doesn't poison the next.
#
# Experiments (in order):
#   E1 op-fusion REAL  — actually apply submission_*_opfusion/overlay/minicpm.py
#                        before smoke (last session's smoke "OK 82.33" did NOT
#                        apply the overlay because local_eval skips prepare_env;
#                        result was equivalent to chunk32k baseline).
#   E2 fp8kv Path E    — patch minicpm_backend.py with fp8→bf16 dequant fallback
#                        right after get_kv_buffer in both forward_extend and
#                        forward_decode. Side-steps the scale-plumbing question
#                        and proves fp8 KV pool can work on SALA hybrid backend.
#                        Loses ~50% bandwidth win but unblocks FA.
#   E3 full-attn-bf16  — fork chunk32k_safe with quantize script expanded to
#                        quantize self_attn.{q,k,v,o}_proj on minicpm4 dense
#                        layers (NOT MLP-only). lightning layers' o_gate /
#                        z_proj / o_norm / q_norm / k_norm stay skipped via
#                        dynamic. High risk: gptqmodel may crash on lightning
#                        layers that lack self_attn (no graceful skip).
#
# Each experiment: apply patches → local_eval smoke (NO_PROXY, --startup-timeout
# 1800, reuse cached artifact where possible) → revert patches via git checkout →
# parse acc → commit + pull-rebase + push → continue.
#
# Run autonomously:
#   cd /root/autodl-tmp/zyn/sglang
#   nohup bash scripts/gpu_smoke_phase3.sh > /tmp/phase3.log 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/sglang_minicpm_sala_env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/sglang_minicpm_sala_env/bin/activate"
fi

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

TS="$(date +%Y%m%d_%H%M%S)"
NAS_DIR="/root/autodl-fs/zyn/logs/phase3_${TS}"
mkdir -p "$NAS_DIR" 2>/dev/null || NAS_DIR="/tmp/phase3_${TS}"
mkdir -p "$NAS_DIR"

MASTER_LOG="$NAS_DIR/master.log"
SUMMARY_MD="$NAS_DIR/SUMMARY.md"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "============================================================"
echo "Phase 3 orchestrator — $TS"
echo "log dir: $NAS_DIR"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-(none)}"
echo "============================================================"

NUM_SAMPLES="${NUM_SAMPLES:-30}"
SHARED_QUANT_DIR="/root/autodl-fs/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized"

{
    echo "# Phase 3 Smokes — $TS"
    echo ""
    echo "After chunk32k_safe platform success (acc_ori=80.31, final_score=22.9)."
    echo ""
    echo "## Results"
    echo ""
    echo "| Experiment | Status | acc_ori | duration | Notes |"
    echo "|---|---|---|---|---|"
} > "$SUMMARY_MD"

# ----------------------------------------------------------------------
# git commit helper (pull-rebase before push)
# ----------------------------------------------------------------------
commit_and_push() {
    local msg="$1"
    git add -f scripts/eval_results.csv 2>/dev/null || true
    git add -f scripts/logs/local_eval_*.log 2>/dev/null || true
    git add -f outputs/ 2>/dev/null || true
    git add -f "$SUMMARY_MD" 2>/dev/null || true

    if git diff --cached --quiet 2>/dev/null; then
        echo "[git] nothing to commit"
        return 0
    fi
    git commit --no-verify -m "${msg}

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
    if ! git pull --rebase --no-edit 2>/tmp/.phase3_push_err; then
        git rebase --abort 2>/dev/null || true
        echo "[git] PULL --REBASE FAILED: $(tail -1 /tmp/.phase3_push_err)"
        return 1
    fi
    git push 2>&1 | tail -3
}

# ----------------------------------------------------------------------
# Smoke runner + outcome parser
# ----------------------------------------------------------------------
run_smoke_variant() {
    local variant="$1"
    local quant_target="/root/autodl-fs/zyn/models/${variant}-quantized"
    local exp_log="$2"
    local force_arg="$3"   # "yes" or "no"
    local local_force=""
    [ "$force_arg" = "yes" ] && local_force="--force-requant"

    bash "$REPO_ROOT/scripts/local_eval.sh" \
        --variant "$variant" \
        --num-samples "$NUM_SAMPLES" \
        --startup-timeout 1800 \
        $local_force > "$exp_log" 2>&1 || true
}

parse_acc() {
    local log="$1"
    grep -oE 'acc_ori \(raw on dataset\):[[:space:]]+[0-9]+\.[0-9]+' "$log" \
        | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true
}

# ----------------------------------------------------------------------
# E1 — op-fusion REAL (apply overlay, smoke, revert)
# ----------------------------------------------------------------------
run_e1_opfusion() {
    local exp_name="E1_opfusion_real"
    local exp_log="$NAS_DIR/${exp_name}.log"
    local overlay_src="$REPO_ROOT/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion/overlay/minicpm.py"
    local target="$REPO_ROOT/python/sglang/srt/models/minicpm.py"
    local variant="submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion"

    echo ""; echo "===== E1 opfusion REAL ====="
    if [ ! -f "$overlay_src" ]; then
        echo "  [SKIP] overlay not found at $overlay_src"
        echo "| $exp_name | SKIP | — | — | overlay missing |" >> "$SUMMARY_MD"
        return 0
    fi

    # Apply overlay (cp; revert via git checkout)
    cp -f "$overlay_src" "$target"
    echo "  [apply] copied overlay → $target"

    # Verify import doesn't break
    if ! python3 -c "import sglang.srt.models.minicpm" >/dev/null 2>&1; then
        echo "  [FAIL] overlay broke sglang.srt.models.minicpm import; reverting + skipping"
        git checkout -- "$target"
        echo "| $exp_name | SKIP | — | — | overlay import broken |" >> "$SUMMARY_MD"
        commit_and_push "phase3 E1: opfusion overlay import broken (skipped)" || true
        return 0
    fi

    # Symlink artifact (reuse chunk32k_safe)
    rm -rf "/root/autodl-fs/zyn/models/${variant}-quantized"
    ln -sfn "$SHARED_QUANT_DIR" "/root/autodl-fs/zyn/models/${variant}-quantized"

    local t0=$(date +%s)
    run_smoke_variant "$variant" "$exp_log" "no"
    local dur=$(( $(date +%s) - t0 ))
    local acc=$(parse_acc "$exp_log")

    # Revert overlay
    git checkout -- "$target"
    echo "  [revert] $target restored"
    rm -f "/root/autodl-fs/zyn/models/${variant}-quantized"

    local status="UNKNOWN" note=""
    if [ -n "$acc" ]; then
        if awk "BEGIN{exit !($acc >= 78)}"; then status="OK"
        elif awk "BEGIN{exit !($acc >= 60)}"; then status="OK_acc_low"
        else status="acc_BAD"; fi
    elif grep -qE "FlashAttention only support fp16|RuntimeError|Traceback" "$exp_log"; then
        status="CRASH"
        note=$(grep -E "Error|Exception" "$exp_log" | tail -1 | cut -c-80 | tr -d '|')
    fi

    echo "  done: status=$status acc_ori=${acc:-—} dur=${dur}s"
    echo "| $exp_name | $status | ${acc:-—} | ${dur}s | overlay applied; ${note} |" >> "$SUMMARY_MD"
    cp -f "$exp_log" "$REPO_ROOT/scripts/logs/phase3_${TS}_E1.log" 2>/dev/null || true
    commit_and_push "phase3 E1 opfusion REAL: $status acc_ori=${acc:-—}" || true
}

# ----------------------------------------------------------------------
# E2 — fp8kv Path E (dequant fallback inside minicpm_backend.py)
# ----------------------------------------------------------------------
run_e2_path_e() {
    local exp_name="E2_fp8kv_path_e"
    local exp_log="$NAS_DIR/${exp_name}.log"
    local variant="submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv"
    local backend_src="$REPO_ROOT/python/sglang/srt/layers/attention/minicpm_backend.py"
    local patch_tool="$REPO_ROOT/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv/apply_gptq_marlin_kv_method_patch.py"
    local gptq_target="$REPO_ROOT/python/sglang/srt/layers/quantization/gptq.py"

    echo ""; echo "===== E2 fp8kv Path E (dequant fallback) ====="

    # Apply Path D first (gives layer.k_scale attached, defensive)
    if [ -f "$patch_tool" ]; then
        python3 "$patch_tool" "$gptq_target" >/dev/null 2>&1 || true
        echo "  [apply] Path D patch on gptq.py"
    fi

    # Apply Path E inline patch to minicpm_backend.py — insert dequant right
    # after both get_kv_buffer calls (forward_extend ~line 1030, forward_decode ~line 1158).
    # Use python to be robust against line drift.
    python3 - <<'PYEOF' || { echo "  [FAIL] Path E patch failed"; git checkout -- "$backend_src" "$gptq_target"; return 1; }
import re
from pathlib import Path
p = Path("/root/autodl-tmp/zyn/sglang/python/sglang/srt/layers/attention/minicpm_backend.py")
src = p.read_text()
marker = "# PATH_E_DEQUANT_FALLBACK"
if marker in src:
    print("[path_e] already applied")
    raise SystemExit(0)

# Find each "get_kv_buffer(" call and inject dequant after the assignment block.
# Match pattern: key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(...)
# (allowing for multi-line). Inject after the closing paren of the call.
pattern = re.compile(
    r"(key_cache,\s*value_cache\s*=\s*forward_batch\.token_to_kv_pool\.get_kv_buffer\([^)]*\)\s*\n)",
    re.MULTILINE,
)
insert = """        # PATH_E_DEQUANT_FALLBACK (2026-05-24 experiment): if KV pool is fp8,
        # dequant K/V to bf16 in-place before downstream FA/sparse kernels.
        # Side-steps scale-plumbing (layer.k_scale path); loses ~50% of fp8
        # bandwidth win but proves SALA hybrid backend tolerates fp8 KV pool.
        if str(self.kv_cache_dtype).startswith("torch.float8"):
            key_cache = key_cache.to(torch.bfloat16)
            value_cache = value_cache.to(torch.bfloat16)
"""
new_src, n = pattern.subn(lambda m: m.group(1) + insert, src)
if n == 0:
    print("[path_e] FATAL: no get_kv_buffer call site found"); raise SystemExit(1)
p.write_text(new_src)
print(f"[path_e] injected dequant after {n} get_kv_buffer site(s)")
PYEOF

    if ! python3 -c "import sglang.srt.layers.attention.minicpm_backend" >/dev/null 2>&1; then
        echo "  [FAIL] Path E broke import; reverting + skipping"
        git checkout -- "$backend_src" "$gptq_target"
        echo "| $exp_name | SKIP | — | — | Path E broke import |" >> "$SUMMARY_MD"
        commit_and_push "phase3 E2 Path E: import broken (skipped)" || true
        return 0
    fi

    # Symlink artifact (reuse chunk32k_safe — same W4A16 weights)
    rm -rf "/root/autodl-fs/zyn/models/${variant}-quantized"
    ln -sfn "$SHARED_QUANT_DIR" "/root/autodl-fs/zyn/models/${variant}-quantized"

    local t0=$(date +%s)
    run_smoke_variant "$variant" "$exp_log" "no"
    local dur=$(( $(date +%s) - t0 ))
    local acc=$(parse_acc "$exp_log")

    # Revert all source patches
    git checkout -- "$backend_src" "$gptq_target"
    echo "  [revert] backend + gptq restored"
    rm -f "/root/autodl-fs/zyn/models/${variant}-quantized"

    local status="UNKNOWN" note=""
    if [ -n "$acc" ]; then
        if awk "BEGIN{exit !($acc >= 78)}"; then status="OK"
        elif awk "BEGIN{exit !($acc >= 60)}"; then status="OK_acc_low"
        else status="acc_BAD"; fi
    elif grep -qE "FlashAttention only support fp16|RuntimeError" "$exp_log"; then
        status="CRASH_FA"
        note="FA still rejects (Path E insufficient)"
    elif grep -qE "Traceback|Exception" "$exp_log"; then
        status="CRASH_other"
        note=$(grep -E "Error|Exception" "$exp_log" | tail -1 | cut -c-80 | tr -d '|')
    fi

    echo "  done: status=$status acc_ori=${acc:-—} dur=${dur}s"
    echo "| $exp_name | $status | ${acc:-—} | ${dur}s | Path E dequant fallback; $note |" >> "$SUMMARY_MD"
    cp -f "$exp_log" "$REPO_ROOT/scripts/logs/phase3_${TS}_E2.log" 2>/dev/null || true
    commit_and_push "phase3 E2 fp8kv Path E: $status acc_ori=${acc:-—}" || true
}

# ----------------------------------------------------------------------
# E3 — full-attn quant on bf16 path (fork variant + smoke)
# ----------------------------------------------------------------------
run_e3_full_attn() {
    local exp_name="E3_full_attn_bf16"
    local exp_log="$NAS_DIR/${exp_name}.log"
    local base="$REPO_ROOT/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe"
    local fork="$REPO_ROOT/submission_phase3_full_attn_bf16"

    echo ""; echo "===== E3 full-attn-bf16 ====="

    # Build the fork by copying chunk32k_safe + editing quantize script.
    rm -rf "$fork"
    mkdir -p "$fork"
    for f in prepare_env.sh prepare_model.sh flash_attn-*.whl; do
        cp -f "$base/$f" "$fork/" 2>/dev/null || cp -fL "$base/$f" "$fork/" 2>/dev/null || true
    done
    ln -sfn "$base/sglang" "$fork/sglang" 2>/dev/null || cp -fr "$base/sglang" "$fork/" 2>/dev/null || true
    cp -fL "$base/quantize_gptqmodel_w4a16.py" "$fork/quantize_gptqmodel_w4a16.py"

    # Patch the fork's quantize script: expand layer_modules to include
    # self_attn projections, AND remove the `-:.*self_attn.*` dynamic skip.
    # Other SALA-specific skips (o_gate/z_proj/o_norm/q_norm/k_norm) stay because
    # gptqmodel crashes if it sees them on layers that don't have them.
    python3 - <<'PYEOF' || { echo "  [FAIL] could not patch quantize script"; return 1; }
from pathlib import Path
p = Path("/root/autodl-tmp/zyn/sglang/submission_phase3_full_attn_bf16/quantize_gptqmodel_w4a16.py")
src = p.read_text()
# Expand layer_modules
old_lm = """        layer_modules = [
            ["mlp.gate_proj", "mlp.up_proj"],
            ["mlp.down_proj"],
        ]"""
new_lm = """        # Phase 3 E3: include self_attn projections (q/k/v fused, o separate).
        # Lightning layers lack self_attn entirely — gptqmodel will see "module
        # not found" on those layers; this experiment validates whether gptqmodel
        # 7.0 silently skips or fatally crashes. If it crashes, layer-type-aware
        # dispatch is needed (out of scope tonight).
        layer_modules = [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["mlp.gate_proj", "mlp.up_proj"],
            ["mlp.down_proj"],
        ]"""
if old_lm not in src:
    print("[e3] FATAL: layer_modules marker not found"); raise SystemExit(1)
src = src.replace(old_lm, new_lm)
# Remove the self_attn skip from dynamic (keep the other 5)
old_dyn_line = '            "-:.*self_attn.*": True,  # keep all sparse/linear attention projections BF16'
if old_dyn_line not in src:
    print("[e3] FATAL: self_attn dynamic skip line not found"); raise SystemExit(1)
src = src.replace(old_dyn_line + "\n", "")
p.write_text(src)
print("[e3] quantize script patched for full-attn")
PYEOF

    # Run smoke with --force-requant (own artifact; different quant config)
    local t0=$(date +%s)
    bash "$REPO_ROOT/scripts/local_eval.sh" \
        --variant submission_phase3_full_attn_bf16 \
        --num-samples "$NUM_SAMPLES" \
        --startup-timeout 1800 \
        --force-requant > "$exp_log" 2>&1 || true
    local dur=$(( $(date +%s) - t0 ))
    local acc=$(parse_acc "$exp_log")

    local status="UNKNOWN" note=""
    if [ -n "$acc" ]; then
        if awk "BEGIN{exit !($acc >= 78)}"; then status="OK"
        elif awk "BEGIN{exit !($acc >= 60)}"; then status="OK_acc_low"
        else status="acc_BAD"; fi
    elif grep -qiE "self_attn|module not found|AttributeError" "$exp_log"; then
        status="CRASH_quant"
        note="gptqmodel crashed on missing self_attn in lightning layers"
    elif grep -qE "FlashAttention only support fp16|RuntimeError" "$exp_log"; then
        status="CRASH_FA"
    elif grep -qE "Traceback|Exception" "$exp_log"; then
        status="CRASH_other"
        note=$(grep -E "Error|Exception" "$exp_log" | tail -1 | cut -c-80 | tr -d '|')
    fi

    echo "  done: status=$status acc_ori=${acc:-—} dur=${dur}s"
    echo "| $exp_name | $status | ${acc:-—} | ${dur}s | full-attn q/k/v/o quantized; $note |" >> "$SUMMARY_MD"
    cp -f "$exp_log" "$REPO_ROOT/scripts/logs/phase3_${TS}_E3.log" 2>/dev/null || true
    # NOTE: deliberately NOT staging the fork variant dir — it's experimental,
    # don't pollute submission_* namespace. The log + result is enough record.
    commit_and_push "phase3 E3 full-attn bf16: $status acc_ori=${acc:-—}" || true
}

# ----------------------------------------------------------------------
# Run all
# ----------------------------------------------------------------------
run_e1_opfusion
run_e2_path_e
run_e3_full_attn

# Final SUMMARY
echo ""
echo "============================================================"
echo "PHASE 3 COMPLETE — $(date '+%F %T')"
echo "SUMMARY: $SUMMARY_MD"
echo "============================================================"
cat "$SUMMARY_MD"

mkdir -p "$REPO_ROOT/scripts/logs/phase3_${TS}"
cp -f "$SUMMARY_MD" "$REPO_ROOT/scripts/logs/phase3_${TS}/SUMMARY.md"
git add -f "$REPO_ROOT/scripts/logs/phase3_${TS}/" 2>/dev/null || true
commit_and_push "phase3 COMPLETE — SUMMARY ($TS)" || true
