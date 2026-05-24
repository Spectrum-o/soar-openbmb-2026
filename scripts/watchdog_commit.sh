#!/usr/bin/env bash
# scripts/watchdog_commit.sh
#
# Periodically snapshots experimental artifacts (logs, predictions,
# SUBMISSIONS.md, edited submission scripts) into git and pushes to
# the remote, so an interrupted GPU session doesn't lose data.
#
# Whitelisted paths only — never touches the quantized model
# (lives on /root/autodl-fs/zyn/models/, NOT in the repo), the venv,
# or the bundled `sglang/` source inside submission directories.
#
# Usage (run on the AutoDL server in another window or backgrounded):
#
#   # 5-min default, attached so you see the commit log
#   bash scripts/watchdog_commit.sh
#
#   # 2-min interval
#   INTERVAL=120 bash scripts/watchdog_commit.sh
#
#   # backgrounded, log to file, stop with `pkill -f watchdog_commit`
#   nohup bash scripts/watchdog_commit.sh > /tmp/watchdog.log 2>&1 &
#
# Notes:
#   - Pre-commit hooks are BYPASSED (`--no-verify`). The repo's
#     .pre-commit-config.yaml has `check-added-large-files` (500KB
#     default), which would block snapshots of outputs/predictions.jsonl
#     (~25MB per eval run). Bypassing is intentional for this auto-save
#     workflow; for manual commits, run without watchdog.
#   - If `git push` fails (network / auth), the commit still lands
#     locally and will be pushed on the next successful tick.
#   - All commits are tagged "auto: watchdog snapshot" so you can
#     squash them later with an interactive rebase if you want a
#     cleaner history.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${INTERVAL:-300}"

cd "$REPO"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")"

# Snapshot whitelist. Add / remove paths here, not via `git add -A`.
SNAPSHOT_PATHS=(
    "scripts/logs/"
    "outputs/"
    "scripts/eval_results.csv"
    "scripts/watchdog_commit.sh"
    "scripts/local_eval.sh"
    "SUBMISSIONS.md"
    "handoff/"
    "submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py"
    "submission_gptqmodel_calib_w4a16/prepare_model.sh"
    "submission_gptqmodel_calib_w4a16/prepare_env.sh"
    "submission_gptq_v17_minconfig/quantize_gptqmodel_w4a16.py"
    "submission_gptq_v17_minconfig/prepare_model.sh"
    "submission_gptq_v17_minconfig/prepare_env.sh"
)

echo "[watchdog] $(date '+%F %T') repo=$REPO branch=$BRANCH interval=${INTERVAL}s pid=$$"
echo "[watchdog] paths: ${SNAPSHOT_PATHS[*]}"
echo "[watchdog] stop with: kill $$ (or pkill -f watchdog_commit)"

tick() {
    local staged_any=0
    for p in "${SNAPSHOT_PATHS[@]}"; do
        if [ -e "$p" ]; then
            # `git add -f` so a misconfigured .gitignore doesn't silently drop logs.
            # Quant artifacts and the venv are NOT in the whitelist, so this is safe.
            git add -f "$p" 2>/dev/null && staged_any=1
        fi
    done

    if [ "$staged_any" -eq 0 ] || git diff --staged --quiet; then
        echo "[$(date '+%F %T')] no changes"
        return
    fi

    local summary
    summary="$(git diff --staged --stat | tail -1 | xargs)"
    local stamp
    stamp="$(date -u +%FT%TZ)"

    if git commit --no-verify -m "auto: watchdog snapshot ${stamp}

${summary}" >/dev/null 2>&1; then
        # Sync with remote before push, otherwise origin advancing (other
        # Claude session, other dev) yields a non-fast-forward reject and the
        # commit stays trapped locally forever. Abort on conflict so the tree
        # stays usable; we'll retry next tick. --no-edit keeps the rebase
        # non-interactive. Skipping --autostash on purpose: working tree
        # should be clean here (our git add + commit just landed everything
        # in the whitelist), and stashing unrelated user changes risks an
        # unstash conflict.
        if ! git pull --rebase --no-edit 2>/tmp/.watchdog_push_err; then
            git rebase --abort 2>/dev/null || true
            local err
            err="$(tail -2 /tmp/.watchdog_push_err | tr '\n' ' ')"
            echo "[$(date '+%F %T')] committed but PULL --REBASE FAILED (aborted, will retry next tick): ${err}"
            return
        fi
        if git push 2>/tmp/.watchdog_push_err; then
            echo "[$(date '+%F %T')] committed + pushed: ${summary}"
        else
            local err
            err="$(tail -2 /tmp/.watchdog_push_err | tr '\n' ' ')"
            echo "[$(date '+%F %T')] committed but PUSH FAILED: ${err}"
        fi
    else
        echo "[$(date '+%F %T')] commit failed (run \`git status\`; nothing staged or hook blocked even --no-verify?)"
    fi
}

# Initial tick so the first snapshot lands immediately, not after $INTERVAL.
tick
while sleep "$INTERVAL"; do
    tick
done
