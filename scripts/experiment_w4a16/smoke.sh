#!/usr/bin/env bash
# scripts/experiment_w4a16/smoke.sh
#
# 30-second smoke check before launching the long autonomous W4A16 run.
# No GPU work — just verifies the env, paths, scripts, imports are healthy.
# If this fails, the long run will fail too; if this passes, the long run
# has a chance.
#
# Usage:
#   bash scripts/experiment_w4a16/smoke.sh
#   bash scripts/experiment_w4a16/smoke.sh --verbose   # show all output

set -u   # NOT -e: collect all failures, exit non-zero if any failed
VERBOSE=0
[ "${1:-}" = "--verbose" ] && VERBOSE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PASS=0
FAIL=0
FAILED_CHECKS=()

check() {
    local label="$1"
    local cmd="$2"
    local out
    out=$(eval "$cmd" 2>&1)
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        printf "  \e[32m✓\e[0m %s\n" "$label"
        [ "$VERBOSE" = 1 ] && [ -n "$out" ] && echo "    $out" | head -3
        PASS=$((PASS + 1))
    else
        printf "  \e[31m✗\e[0m %s\n" "$label"
        echo "    error: $out" | head -5
        FAIL=$((FAIL + 1))
        FAILED_CHECKS+=("$label")
    fi
}

echo "==================================================="
echo " W4A16 migration: pre-flight smoke check"
echo " $(date '+%F %T') on $(hostname)"
echo "==================================================="

echo "[1] Filesystem & repo state"
check "in repo root with .git" \
    "test -d '$REPO_ROOT/.git'"
check "submission_bf16_official_args/ exists" \
    "test -d '$REPO_ROOT/submission_bf16_official_args'"
check "submission_awq_official_args/ exists" \
    "test -d '$REPO_ROOT/submission_awq_official_args'"
check "submission_bf16_official_args/prepare_env.sh parses" \
    "bash -n '$REPO_ROOT/submission_bf16_official_args/prepare_env.sh'"
check "submission_awq_official_args/prepare_env.sh parses" \
    "bash -n '$REPO_ROOT/submission_awq_official_args/prepare_env.sh'"
check "submission_bf16_official_args/prepare_model.sh exists" \
    "test -f '$REPO_ROOT/submission_bf16_official_args/prepare_model.sh'"
check "submission_awq_official_args/prepare_model.sh resolves (symlink)" \
    "test -e '$REPO_ROOT/submission_awq_official_args/prepare_model.sh'"
check "submission_awq_official_args/quantize_llmcompressor_awq.py resolves" \
    "test -e '$REPO_ROOT/submission_awq_official_args/quantize_llmcompressor_awq.py'"
check "scripts/local_eval.sh executable" \
    "test -x '$REPO_ROOT/scripts/local_eval.sh'"
check "scripts/experiment_w4a16/run_plan.sh executable" \
    "test -x '$REPO_ROOT/scripts/experiment_w4a16/run_plan.sh'"

echo "[2] SGLANG_SERVER_ARGS contain official W4A16 recipe"
BF16_ARGS=$(grep -E '^export SGLANG_SERVER_ARGS=' "$REPO_ROOT/submission_bf16_official_args/prepare_env.sh" 2>/dev/null | head -1)
AWQ_ARGS=$(grep -E '^export SGLANG_SERVER_ARGS=' "$REPO_ROOT/submission_awq_official_args/prepare_env.sh" 2>/dev/null | head -1)
check "bf16 variant has chunked-prefill 65536" \
    "echo '$BF16_ARGS' | grep -q 'chunked-prefill-size 65536'"
check "bf16 variant has mem-fraction-static 0.80" \
    "echo '$BF16_ARGS' | grep -q 'mem-fraction-static 0.80'"
check "bf16 variant has --dtype bfloat16" \
    "echo '$BF16_ARGS' | grep -q -- '--dtype bfloat16'"
check "bf16 variant does NOT have gptq_marlin (no fp16 sed-patch wanted)" \
    "! echo '$BF16_ARGS' | grep -q 'gptq_marlin'"
check "awq variant has chunked-prefill 65536" \
    "echo '$AWQ_ARGS' | grep -q 'chunked-prefill-size 65536'"
check "awq variant has compressed-tensors loader" \
    "echo '$AWQ_ARGS' | grep -q 'compressed-tensors'"
check "awq variant does NOT have gptq_marlin" \
    "! echo '$AWQ_ARGS' | grep -q 'gptq_marlin'"

echo "[3] Python env (this venv)"
check "uv on PATH" \
    "command -v uv"
check "python3 importable: transformers" \
    "python3 -c 'import transformers; print(transformers.__version__)'"
check "python3 importable: llmcompressor" \
    "python3 -c 'import llmcompressor; print(llmcompressor.__version__)'"
check "python3 importable: flash_attn" \
    "python3 -c 'import flash_attn; print(flash_attn.__version__)'"
check "python3 importable: sglang" \
    "python3 -c 'import sglang'"

echo "[4] Data + model paths (AutoDL conventions)"
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
check "BF16 source model: $MODEL_PATH" \
    "test -d '$MODEL_PATH' && test -f '$MODEL_PATH/config.json'"
TOOLKIT_DIR="${TOOLKIT_DIR:-/root/autodl-fs/zyn/soar_toolkit}"
check "SOAR toolkit dir: $TOOLKIT_DIR" \
    "test -d '$TOOLKIT_DIR'"
check "eval_model.py present" \
    "test -f '$TOOLKIT_DIR/eval_model.py'"
check "perf_public_set.jsonl present" \
    "test -f '$TOOLKIT_DIR/perf_public_set.jsonl'"

echo "[5] Sed-patch leak check (would corrupt bf16 runs)"
BACKEND_DIR="$REPO_ROOT/python/sglang/srt/layers/attention"
check "minicpm_backend.py still has torch.bfloat16 (no fp16 leak)" \
    "test ! -f '$BACKEND_DIR/minicpm_backend.py' || grep -q 'torch.bfloat16' '$BACKEND_DIR/minicpm_backend.py'"
check "minicpm_sparse_utils.py still has torch.bfloat16 (no fp16 leak)" \
    "test ! -f '$BACKEND_DIR/minicpm_sparse_utils.py' || grep -q 'torch.bfloat16' '$BACKEND_DIR/minicpm_sparse_utils.py'"

echo "[6] GPU availability"
check "nvidia-smi accessible" \
    "command -v nvidia-smi"
check "at least one GPU visible" \
    "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -q '.'"
check "GPU has free memory (>= 30 GB)" \
    "python3 -c 'import subprocess as s; r=s.run([\"nvidia-smi\",\"--query-gpu=memory.free\",\"--format=csv,nounits,noheader\"],capture_output=True,text=True); free=[int(x.strip()) for x in r.stdout.split() if x.strip().isdigit()][0]; exit(0 if free >= 30000 else 1)'"

echo "==================================================="
if [ "$FAIL" -eq 0 ]; then
    printf " \e[32mALL %d CHECKS PASSED\e[0m\n" "$PASS"
    echo
    echo " Ready to launch long autonomous run:"
    echo "   nohup bash scripts/experiment_w4a16/run_plan.sh \\"
    echo "       > /tmp/w4a16_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
    echo
    echo " Or first try a 15-min smoke pipeline test:"
    echo "   bash scripts/experiment_w4a16/run_plan.sh --smoke"
    echo "==================================================="
    exit 0
else
    printf " \e[31m%d CHECKS FAILED (%d passed)\e[0m\n" "$FAIL" "$PASS"
    echo " Failed checks:"
    for c in "${FAILED_CHECKS[@]}"; do
        echo "   - $c"
    done
    echo
    echo " Do NOT launch the long run until these are fixed."
    echo "==================================================="
    exit 1
fi
