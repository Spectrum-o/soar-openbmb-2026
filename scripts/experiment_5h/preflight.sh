#!/usr/bin/env bash
# scripts/experiment_5h/preflight.sh
#
# Verify the 5h plan can run on this machine. Checks:
#   - Right python env (transformers 4.57.1, gptqmodel 7.0.0, hub <1.0)
#   - Input model exists
#   - SOAR Toolkit eval script + dataset exist
#   - Scripts are executable
#   - test_llmcompressor_artifact.sh exists (Phase B prerequisite)
#   - 5/14 llm-compressor artifact exists (Phase B prerequisite, optional)
#
# Usage:
#   bash scripts/experiment_5h/preflight.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

FAIL=0
ok()   { echo "  ✓ $*"; }
warn() { echo "  ! $*" >&2; }
fail() { echo "  ✗ $*" >&2; FAIL=1; }

echo "=== 5h plan preflight ==="

# 1. Python env
echo "[1/6] python env"
python3 - <<'PY' || FAIL=1
import importlib.metadata as m
import sys
def v(p):
    try: return m.version(p)
    except m.PackageNotFoundError: return None
print(f"  python                = {sys.version.split()[0]}")
for pkg in ("transformers", "gptqmodel", "huggingface-hub", "tokenizers", "flash-attn", "flash-linear-attention", "torch"):
    ver = v(pkg)
    print(f"  {pkg:24s} = {ver}")
    if pkg == "transformers" and ver != "4.57.1":
        print(f"  ! transformers must be 4.57.1, got {ver}")
        sys.exit(1)
    if pkg == "gptqmodel" and ver != "7.0.0":
        print(f"  ! gptqmodel must be 7.0.0, got {ver}")
        sys.exit(1)
    if pkg == "huggingface-hub" and ver and int(ver.split(".")[0]) >= 1:
        print(f"  ! huggingface-hub must be <1.0, got {ver}")
        sys.exit(1)
PY
[ "$FAIL" = 0 ] && ok "python env OK"

# 2. Input model
echo "[2/6] BF16 source model"
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/OpenBMB/MiniCPM-SALA}"
[ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ] \
    && ok "model at $MODEL_PATH" \
    || fail "model not found at $MODEL_PATH (set MODEL_PATH env var)"

# 3. SOAR Toolkit
echo "[3/6] SOAR Toolkit"
TOOLKIT_DIR="${TOOLKIT_DIR:-/root/autodl-fs/zyn/soar_toolkit}"
[ -f "$TOOLKIT_DIR/eval_model.py" ] \
    && ok "eval_model.py at $TOOLKIT_DIR" \
    || fail "eval_model.py not at $TOOLKIT_DIR/eval_model.py"
[ -f "$TOOLKIT_DIR/perf_public_set.jsonl" ] \
    && ok "perf_public_set.jsonl at $TOOLKIT_DIR" \
    || fail "perf_public_set.jsonl not at $TOOLKIT_DIR"

# 4. Scripts executable
echo "[4/6] runner scripts"
for s in scripts/local_eval.sh \
         scripts/experiment_5h/run_plan.sh \
         scripts/experiment_5h/commit_push.sh \
         scripts/experiment_5h/rep_penalty_sweep.sh; do
    [ -x "$s" ] && ok "$s exec" || fail "$s NOT executable (chmod +x)"
done

for p in scripts/experiment_5h/parse_results.py \
         scripts/experiment_5h/pick_winners.py; do
    [ -f "$p" ] && ok "$p exists" || fail "$p missing"
done

# 5. Phase B prereq (test_llmcompressor_artifact.sh — lives on parallel branch)
echo "[5/6] Phase B prereqs (llm-compressor smoke)"
if [ -f "scripts/test_llmcompressor_artifact.sh" ]; then
    ok "test_llmcompressor_artifact.sh exists"
else
    warn "test_llmcompressor_artifact.sh missing — cherry-pick from parallel/non-gptqmodel-paths"
    warn "  git fetch origin parallel/non-gptqmodel-paths"
    warn "  git checkout origin/parallel/non-gptqmodel-paths -- scripts/test_llmcompressor_artifact.sh"
fi

LLMC_ARTIFACT="${LLMC_ARTIFACT:-/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16}"
if [ -d "$LLMC_ARTIFACT" ]; then
    ok "5/14 llmcompressor artifact at $LLMC_ARTIFACT"
else
    warn "5/14 llmcompressor artifact NOT at $LLMC_ARTIFACT — Phase B will be skipped"
fi

# 6. git tracking
echo "[6/6] git tracking"
[ -d ".git" ] && ok "git repo" || fail "not a git repo"
git remote -v 2>/dev/null | grep -q origin && ok "origin remote configured" || fail "no origin remote"

echo
if [ "$FAIL" = 0 ]; then
    echo "=== PREFLIGHT PASSED. Safe to run: ==="
    echo "  bash scripts/experiment_5h/run_plan.sh --smoke   # quick 2-min-per-exp boot test"
    echo "  bash scripts/experiment_5h/run_plan.sh           # full 5h run"
else
    echo "=== PREFLIGHT FAILED — fix issues above before running plan ===" >&2
    exit 1
fi
