#!/usr/bin/env bash
# scripts/experiment_5h/rescue_to_nas.sh
#
# EMERGENCY: copy everything we'd lose if the AutoDL instance is destroyed
# to /root/autodl-fs/zyn/ (which is on the persistent NAS, survives reboot).
#
# Background — why this exists:
#   AutoDL's /root/autodl-tmp/ is wiped when the instance is rebooted or
#   reclaimed. Anything in the git repo cloned under autodl-tmp (including
#   uncommitted-or-unpushed git history) is destroyed with it. The 5h
#   pipeline that ran 2026-05-23 01:36 silently failed to push its
#   commits (see experiments/5H_PIPELINE_HARDENING_HANDOFF.md). If that
#   instance had rebooted before we noticed, ALL of the 5h pipeline's
#   results would have been irrecoverable.
#
#   This script saves:
#     1. scripts/eval_results.csv (the acc numbers — most precious)
#     2. scripts/logs/5h_*/      (per-phase logs)
#     3. git format-patch of unpushed commits (so we can replay history)
#
#   All into a timestamped dir under /root/autodl-fs/zyn/backups/.
#
# Usage (SSH on the server, while pipeline may or may not be running):
#   cd /root/autodl-tmp/zyn/sglang        # or wherever the repo is
#   bash scripts/experiment_5h/rescue_to_nas.sh
#
# Safe to run repeatedly; each run creates a new timestamped backup dir.
# Safe to run on a live pipeline; we only READ from the repo and WRITE
# to NAS — we never touch the working tree or HEAD.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NAS_ROOT="${NAS_ROOT:-/root/autodl-fs/zyn}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${NAS_ROOT}/backups/rescue_${TIMESTAMP}"

# Resolve where we're saving
if [ ! -d "$NAS_ROOT" ]; then
    echo "[rescue] FATAL: NAS root $NAS_ROOT does not exist." >&2
    echo "[rescue]   override with NAS_ROOT=<path> bash $0" >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR"
echo "=== rescue_to_nas: saving repo state to ==="
echo "    $BACKUP_DIR"
echo

# ----------------------------------------------------------------------
# 1. eval_results.csv — the most precious file. acc numbers per phase.
# ----------------------------------------------------------------------
if [ -f "scripts/eval_results.csv" ]; then
    cp -v "scripts/eval_results.csv" "$BACKUP_DIR/eval_results.csv"
    wc -l "$BACKUP_DIR/eval_results.csv"
else
    echo "[rescue] WARN: scripts/eval_results.csv missing"
fi

# ----------------------------------------------------------------------
# 2. 5h log directories — per-phase eval/server logs
# ----------------------------------------------------------------------
if compgen -G "scripts/logs/5h_*" > /dev/null; then
    mkdir -p "$BACKUP_DIR/logs"
    # Use -r and -L to follow symlinks and preserve content; --parents
    # keeps directory structure.
    cp -rL scripts/logs/5h_*/ "$BACKUP_DIR/logs/" 2>/dev/null || true
    echo "[rescue] copied 5h log dirs:"
    ls -1 "$BACKUP_DIR/logs/"
else
    echo "[rescue] WARN: no scripts/logs/5h_*/ found"
fi

# ----------------------------------------------------------------------
# 3. Unpushed git commits — save as patches so we can apply on a fresh instance
# ----------------------------------------------------------------------
echo
echo "[rescue] checking for unpushed commits..."
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "DETACHED")"
echo "    current branch: $CURRENT_BRANCH"

# Try to find a reasonable upstream to diff against
UPSTREAM=""
for candidate in "origin/${CURRENT_BRANCH}" "origin/quant/w4a16" "origin/main"; do
    if git rev-parse "$candidate" >/dev/null 2>&1; then
        UPSTREAM="$candidate"
        break
    fi
done

if [ -n "$UPSTREAM" ]; then
    NUM_AHEAD=$(git rev-list --count "${UPSTREAM}..HEAD" 2>/dev/null || echo "0")
    echo "    commits ahead of $UPSTREAM: $NUM_AHEAD"
    if [ "$NUM_AHEAD" -gt 0 ]; then
        mkdir -p "$BACKUP_DIR/patches"
        # format-patch writes one .patch per commit into the dir
        git format-patch "${UPSTREAM}..HEAD" --output-directory "$BACKUP_DIR/patches" 2>&1 | tail -10
        echo "[rescue] saved $NUM_AHEAD commit(s) as patches:"
        ls -1 "$BACKUP_DIR/patches/"
        # Also save the commit hashes + messages as a quick-look manifest
        git log --pretty=format:"%h %ai %s" "${UPSTREAM}..HEAD" > "$BACKUP_DIR/patches/MANIFEST.txt"
        echo
        echo "    To replay on a fresh instance:"
        echo "      git fetch origin && git checkout quant/w4a16"
        echo "      git am $BACKUP_DIR/patches/*.patch"
    fi
else
    echo "[rescue] WARN: no upstream branch found to diff against"
fi

# ----------------------------------------------------------------------
# 4. git status snapshot — captures uncommitted state for the morning
# ----------------------------------------------------------------------
{
    echo "=== git status (at rescue time $(date '+%F %T')) ==="
    git status
    echo
    echo "=== git log --oneline -20 ==="
    git log --oneline -20 2>&1
    echo
    echo "=== current HEAD ==="
    git rev-parse HEAD
    echo "    branch: $CURRENT_BRANCH"
    echo "    upstream: ${UPSTREAM:-(none)}"
} > "$BACKUP_DIR/git_state.txt"
echo "[rescue] saved git state snapshot to $BACKUP_DIR/git_state.txt"

# ----------------------------------------------------------------------
# 5. Master log if it's not already on NAS (we expect it to be there)
# ----------------------------------------------------------------------
if compgen -G "scripts/logs/5h_*/master.log" > /dev/null; then
    for ml in scripts/logs/5h_*/master.log; do
        session_dir=$(basename "$(dirname "$ml")")
        cp -v "$ml" "$BACKUP_DIR/master_${session_dir}.log" 2>/dev/null || true
    done
fi

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo
echo "=========================================================="
echo "rescue complete:"
echo "  $BACKUP_DIR"
du -sh "$BACKUP_DIR" 2>/dev/null
echo
echo "Files saved:"
find "$BACKUP_DIR" -type f | head -30
echo "=========================================================="
