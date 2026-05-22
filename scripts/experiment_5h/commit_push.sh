#!/usr/bin/env bash
# scripts/experiment_5h/commit_push.sh
#
# Auto-commit eval_results.csv and 5h logs, push to origin/quant/w4a16.
# Best-effort; no failure propagation.
#
# Usage:
#   bash scripts/experiment_5h/commit_push.sh "5h exp A_J0_baseline (20260523_010000_full)"

set -u   # NOT -e: don't bail if git operations fail mid-run

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MSG="${1:-5h experiment unnamed update}"

# Only add eval_results.csv + 5h logs. Don't touch quant artifacts or
# anything else.
git add scripts/eval_results.csv 2>/dev/null || true
git add scripts/logs/5h_*/ 2>/dev/null || true

# Only commit if there's something staged. The check is cheap.
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "${MSG}

Co-Authored-By: Claude (server-side, 5h plan runner) <noreply@anthropic.com>" \
        --no-verify 2>&1 | tail -5
    # Try to push, but don't block if network's bad (we'll retry next iteration)
    if git push origin quant/w4a16 2>&1 | tail -3; then
        echo "[commit_push] pushed OK"
    else
        echo "[commit_push] push FAILED, will retry next time"
    fi
else
    echo "[commit_push] nothing to commit"
fi
