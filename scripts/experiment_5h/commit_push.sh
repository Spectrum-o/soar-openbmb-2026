#!/usr/bin/env bash
# scripts/experiment_5h/commit_push.sh
#
# Auto-commit eval_results.csv and 5h logs, push to origin/quant/w4a16.
# Belt-and-suspenders durability: ALSO mirror everything to NAS so results
# survive an AutoDL instance reboot (which wipes /root/autodl-tmp).
#
# Usage:
#   bash scripts/experiment_5h/commit_push.sh "5h exp A_J0_baseline (20260523_010000_full)"
#
# Env vars (all optional):
#   NAS_BACKUP_DIR     where to mirror to. Default /root/autodl-fs/zyn/backups/5h_live
#   NAS_BACKUP_DISABLE if set non-empty, skip NAS mirror (use in dev/CI)

set -u   # NOT -e: don't bail if git operations fail mid-run

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MSG="${1:-5h experiment unnamed update}"

# Resolve NAS backup target. Default works on AutoDL.
NAS_BACKUP_DIR="${NAS_BACKUP_DIR:-/root/autodl-fs/zyn/backups/5h_live}"
NAS_OK=0
if [ -z "${NAS_BACKUP_DISABLE:-}" ]; then
    if mkdir -p "$NAS_BACKUP_DIR" 2>/dev/null; then
        NAS_OK=1
    else
        echo "[commit_push] WARN: NAS dir $NAS_BACKUP_DIR not writable — backup skipped" >&2
    fi
fi

# ----------------------------------------------------------------------
# Helper: snapshot CSV + log dirs into NAS. Called BEFORE the commit
# attempt so even if commit/push fails, the raw data is durable.
# ----------------------------------------------------------------------
mirror_to_nas() {
    [ "$NAS_OK" = 1 ] || return 0
    if [ -f "scripts/eval_results.csv" ]; then
        cp -f scripts/eval_results.csv "$NAS_BACKUP_DIR/eval_results.csv" 2>/dev/null \
            && echo "[commit_push] mirrored eval_results.csv → $NAS_BACKUP_DIR/"
    fi
    # Mirror per-session log dirs (small, ~few MB each)
    for d in scripts/logs/5h_*/; do
        [ -d "$d" ] || continue
        session=$(basename "$d")
        mkdir -p "$NAS_BACKUP_DIR/$session" 2>/dev/null
        # rsync would be cleaner but may not be installed. cp -u for "update only".
        cp -ruL "$d"/* "$NAS_BACKUP_DIR/$session/" 2>/dev/null || true
    done
}

# ----------------------------------------------------------------------
# Helper: when push fails, save the unpushed commit(s) as a patch on NAS
# so they can be replayed from a different instance.
# ----------------------------------------------------------------------
save_patch_to_nas() {
    [ "$NAS_OK" = 1 ] || return 0
    local upstream="origin/quant/w4a16"
    git rev-parse "$upstream" >/dev/null 2>&1 || return 0
    local ahead
    ahead=$(git rev-list --count "${upstream}..HEAD" 2>/dev/null || echo "0")
    [ "$ahead" -gt 0 ] || return 0
    local ts patches_dir
    ts="$(date +%Y%m%d_%H%M%S)"
    patches_dir="$NAS_BACKUP_DIR/unpushed_patches/${ts}"
    mkdir -p "$patches_dir" 2>/dev/null || return 0
    git format-patch "${upstream}..HEAD" --output-directory "$patches_dir" >/dev/null 2>&1
    git log --pretty=format:"%h %ai %s" "${upstream}..HEAD" > "$patches_dir/MANIFEST.txt" 2>/dev/null
    echo "[commit_push] saved $ahead unpushed commit(s) as patches → $patches_dir"
}

# ----------------------------------------------------------------------
# Stage the things we care about. Don't touch quant artifacts.
# ----------------------------------------------------------------------
git add scripts/eval_results.csv 2>/dev/null || true
git add scripts/logs/5h_*/ 2>/dev/null || true

# Mirror to NAS BEFORE commit/push. Even if everything below fails, the
# CSV + logs are safe on persistent storage.
mirror_to_nas

# ----------------------------------------------------------------------
# Commit + push (with auto-rebase recovery)
# ----------------------------------------------------------------------
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "${MSG}

Co-Authored-By: Claude (server-side, 5h plan runner) <noreply@anthropic.com>" \
        --no-verify 2>&1 | tail -5

    # Push with auto-rebase recovery.
    #
    # CRITICAL DIVERGENCE HISTORY (2026-05-23 ~02:35): a previous 5h pipeline
    # silently lost ALL push attempts for 5.5 hours because:
    #   1. Another Claude session was pushing to origin/quant/w4a16 concurrently
    #      during 01:44-02:21 (12 commits ahead)
    #   2. This script's `git push` returned non-fast-forward rejection
    #   3. The pre-fix `if git push | tail -3` used tail's exit code, so the
    #      reject was reported as "pushed OK"
    #   4. Every subsequent phase had the same problem, building 10+ unpushed
    #      commits on the server's local branch
    #
    # Defense: detect non-fast-forward, try ONE pull --rebase + retry. We only
    # add scripts/eval_results.csv and scripts/logs/5h_*/ to commits, so a
    # rebase against the other Claude's work (in submission_*/ etc.) should
    # have no conflicts. If rebase fails, give up — NAS mirror above means
    # the data is still safe even if push fails permanently.
    push_once() {
        PUSH_OUTPUT="$(git push origin quant/w4a16 2>&1)"
        PUSH_RC=$?
        echo "$PUSH_OUTPUT" | tail -3
        return $PUSH_RC
    }

    if push_once; then
        echo "[commit_push] pushed OK"
    else
        if echo "$PUSH_OUTPUT" | grep -qiE "non-fast-forward|fetch first|rejected"; then
            echo "[commit_push] divergence detected — attempting pull --rebase"
            if git pull --rebase --no-edit origin quant/w4a16 2>&1 | tail -10; then
                echo "[commit_push] rebase OK, retrying push"
                if push_once; then
                    echo "[commit_push] pushed OK (after rebase)"
                else
                    echo "[commit_push] push FAILED even after rebase — saving patch to NAS"
                    save_patch_to_nas
                fi
            else
                # Rebase conflict — abort and leave the working tree clean
                git rebase --abort 2>/dev/null || true
                echo "[commit_push] rebase FAILED (conflicts) — saving patch to NAS"
                echo "[commit_push] resolve manually with:"
                echo "[commit_push]   git pull --rebase origin quant/w4a16   # resolve conflicts"
                echo "[commit_push]   git push origin quant/w4a16"
                save_patch_to_nas
            fi
        else
            echo "[commit_push] push FAILED (rc=$PUSH_RC, not a divergence) — saving patch to NAS"
            save_patch_to_nas
        fi
    fi
else
    echo "[commit_push] nothing to commit"
fi
