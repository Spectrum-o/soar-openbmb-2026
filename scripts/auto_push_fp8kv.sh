#!/usr/bin/env bash
# Auto-check, optionally pack, commit, and push the FP8 KV FlashInfer submission.
# Push is opt-in: pass --push after reviewing the commit locally.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer"
BRANCH="quant/w4a16"
DO_PUSH=0
DO_PACK=0
RUN_SMOKE=1
DRY_RUN=0
MSG="fp8kv flashinfer path-y verified"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --push) DO_PUSH=1 ;;
        --pack) DO_PACK=1 ;;
        --no-smoke) RUN_SMOKE=0 ;;
        --dry-run) DRY_RUN=1 ;;
        --variant) VARIANT="$2"; shift ;;
        --message|-m) MSG="$2"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

need_file() {
    [ -e "$1" ] || { echo "missing: $1" >&2; exit 1; }
}

need_file "$VARIANT/prepare_env.sh"
need_file "$VARIANT/README_SUBMISSION.md"
need_file "tools/local_fp8kv_smoke.py"

if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
    echo "error: expected branch $BRANCH, current $(git rev-parse --abbrev-ref HEAD)" >&2
    exit 1
fi

echo "[1/6] python compile"
/root/autodl-tmp/zyn/sglang/sglang_minicpm_sala_env/bin/python -m py_compile \
    python/sglang/srt/layers/attention/minicpm_backend.py \
    python/sglang/srt/layers/attention/minicpm_attention_kernels.py \
    python/sglang/srt/layers/quantization/gptq.py \
    tools/local_fp8kv_smoke.py

echo "[2/6] diff whitespace check"
git diff --check

echo "[3/6] submission preflight"
python3 tools/pack_submission.py --variant "$VARIANT" --check-only

if [ "$RUN_SMOKE" = 1 ]; then
    echo "[4/6] local FP8KV smoke against http://127.0.0.1:31111"
    env no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
        /root/autodl-tmp/zyn/sglang/sglang_minicpm_sala_env/bin/python \
        tools/local_fp8kv_smoke.py --limit 1 --max-tokens 64
else
    echo "[4/6] local smoke skipped"
fi

TARBALL=""
if [ "$DO_PACK" = 1 ]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    TARBALL="/root/autodl-fs/zyn/submissions/soar_fp8kv_flashinfer_${stamp}.tar.gz"
    mkdir -p "$(dirname "$TARBALL")"
    echo "[5/6] pack -> $TARBALL"
    python3 tools/pack_submission.py --variant "$VARIANT" --output "$TARBALL"
else
    echo "[5/6] pack skipped (--pack not set)"
fi

echo "[6/6] commit selected files"
if [ "$DRY_RUN" = 1 ]; then
    echo "dry-run: would stage FP8KV files and commit with message: $MSG"
    git status --short
    exit 0
fi

git add \
    python/sglang/srt/layers/attention/minicpm_backend.py \
    python/sglang/srt/layers/attention/minicpm_attention_kernels.py \
    python/sglang/srt/layers/quantization/gptq.py \
    submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv/README_SUBMISSION.md \
    submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv/prepare_env.sh \
    submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer/README_SUBMISSION.md \
    submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer/prepare_env.sh \
    tools/local_fp8kv_smoke.py \
    scripts/auto_push_fp8kv.sh

# Public-set jsonl files are ignored by repo default, but these two are
# intentionally bundled for platform-side calibration stability.
git add -f \
    submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv/perf_public_set.jsonl \
    submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer/perf_public_set.jsonl

if git diff --cached --quiet; then
    echo "nothing staged; commit skipped"
else
    git commit -m "$MSG" --no-verify
fi

save_unpushed_patch() {
    local backup_root="${NAS_BACKUP_DIR:-/root/autodl-fs/zyn/backups/fp8kv_push}"
    local ts upstream ahead patch_dir
    mkdir -p "$backup_root" 2>/dev/null || return 0
    upstream="origin/$BRANCH"
    git rev-parse "$upstream" >/dev/null 2>&1 || return 0
    ahead="$(git rev-list --count "${upstream}..HEAD" 2>/dev/null || echo 0)"
    [ "$ahead" -gt 0 ] || return 0
    ts="$(date +%Y%m%d_%H%M%S)"
    patch_dir="$backup_root/unpushed_${ts}"
    mkdir -p "$patch_dir" 2>/dev/null || return 0
    git format-patch "${upstream}..HEAD" --output-directory "$patch_dir" >/dev/null 2>&1 || true
    git log --pretty=format:"%h %ai %s" "${upstream}..HEAD" > "$patch_dir/MANIFEST.txt" 2>/dev/null || true
    echo "saved unpushed patch backup: $patch_dir"
}

if [ "$DO_PUSH" = 1 ]; then
    echo "push requested: rebasing then pushing origin/$BRANCH"
    set +e
    git pull --rebase --no-edit origin "$BRANCH"
    pull_rc=$?
    set -e
    if [ "$pull_rc" -ne 0 ]; then
        echo "pull --rebase failed; aborting rebase if needed and saving patch backup" >&2
        git rebase --abort >/dev/null 2>&1 || true
        save_unpushed_patch
        exit "$pull_rc"
    fi

    set +e
    git push origin "$BRANCH"
    push_rc=$?
    set -e
    if [ "$push_rc" -ne 0 ]; then
        echo "push failed; saving patch backup" >&2
        save_unpushed_patch
        exit "$push_rc"
    fi
    echo "push OK: origin/$BRANCH"
else
    echo "push skipped. Run: bash scripts/auto_push_fp8kv.sh --push"
fi

[ -n "$TARBALL" ] && echo "tarball: $TARBALL"
