#!/usr/bin/env bash
# Commit and push the current full-W4A16 checkpoint.
#
# This intentionally stages only files owned by the full-W4A16 experiment and
# its validation tooling. It also stages shard CSV/log outputs when present, so
# partial eval evidence can be pushed before a long run finishes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-exp/full-w4a16}"
MESSAGE="${MESSAGE:-checkpoint full w4a16 accuracy pass}"
RUN_CHECKS="${RUN_CHECKS:-1}"
PUSH="${PUSH:-1}"

echo "============================================================"
echo " checkpoint_full_w4a16"
echo "============================================================"
echo "remote:     ${REMOTE}"
echo "branch:     ${BRANCH}"
echo "message:    ${MESSAGE}"
echo "run checks: ${RUN_CHECKS}"
echo "push:       ${PUSH}"
echo "============================================================"

if [ "${RUN_CHECKS}" = "1" ]; then
    echo "[1/4] lightweight checks"
    python3 -m py_compile \
        submission_gptqmodel_full_w4a16/quantize_gptqmodel_w4a16.py \
        tools/calib_set_preview.py \
        scripts/full_w4a16_decide_after_eval.py \
        tools/pack_submission.py \
        tools/hard_constraints_lint.py
    bash -n \
        submission_gptqmodel_full_w4a16/prepare_env.sh \
        submission_gptqmodel_full_w4a16/prepare_model.sh \
        scripts/local_eval.sh \
        scripts/local_eval_sharded.sh \
        scripts/run_full_w4a16_platform_acc_now.sh \
        scripts/run_full_w4a16_platform_acc_when_idle.sh \
        scripts/full_preflight.sh \
        scripts/checkpoint_full_w4a16.sh
    git diff --check
fi

echo "[2/4] staging scoped files"
git add \
    docs/FULL_W4A16_EXPERIMENT.md \
    scripts/local_eval.sh \
    scripts/local_eval_sharded.sh \
    scripts/full_w4a16_decide_after_eval.py \
    scripts/run_full_w4a16_platform_acc_now.sh \
    scripts/run_full_w4a16_platform_acc_when_idle.sh \
    scripts/checkpoint_full_w4a16.sh \
    submission_gptqmodel_full_w4a16/README_SUBMISSION.md \
    submission_gptqmodel_full_w4a16/prepare_env.sh \
    submission_gptqmodel_full_w4a16/prepare_model.sh \
    submission_gptqmodel_full_w4a16/quantize_gptqmodel_w4a16.py \
    tests/test_calibration_tools.py \
    tests/test_full_w4a16_variant.py \
    tests/test_hard_constraints_lint.py \
    tests/test_pack_submission_validate.py \
    tools/calib_set_preview.py \
    tools/hard_constraints_lint.py \
    tools/pack_submission.py

# Stage the flash-attn wheel symlink replacement explicitly. Use the variant
# directory path rather than a shell glob so deletions of old wheel symlinks
# are included even after the file no longer exists.
git add -A -- submission_gptqmodel_full_w4a16

# Partial eval evidence is useful during long runs. These globs may not exist.
git add -f scripts/eval_shards.csv 2>/dev/null || true
git add -f scripts/logs/sharded_*.log 2>/dev/null || true
git add -f scripts/logs/shards_*/*.jsonl 2>/dev/null || true

echo "[3/4] staged diff"
git status --short
if git diff --cached --quiet; then
    echo "nothing staged; checkpoint already up to date"
    exit 0
fi

git commit -m "${MESSAGE}"

if [ "${PUSH}" = "1" ]; then
    echo "[4/4] pushing HEAD -> ${REMOTE}/${BRANCH}"
    git push "${REMOTE}" "HEAD:refs/heads/${BRANCH}"
else
    echo "[4/4] push skipped (PUSH=${PUSH})"
fi

echo "checkpoint complete"
