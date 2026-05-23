# 5h-pipeline hardening — emergency handoff (branch `fix/5h-run-plan-hardening`)

> Written 2026-05-23 mid-day after diagnosing why the 01:36:22 pipeline
> went silent. **READ THIS BEFORE MERGING TO `quant/w4a16`** — there's a
> deliberate "don't push yet" gate explained below.

## TL;DR of the diagnosis

The 5-hour pipeline launched on the server at 2026-05-23 01:36:22 **never
pushed a single commit to `origin/quant/w4a16`** during its run. Concrete
evidence:
  - `git log --grep="20260523_013622_full"` → 0 results
  - The 5.5-hour window 02:30→08:00 has **zero** commits to `origin/quant/w4a16`
  - The previous smoke session (00:45) pushed commits fine, proving the
    mechanism works in isolation

## Root cause

Two Claude sessions were both pushing to `origin/quant/w4a16` concurrently:

```
01:34:48  c4471a60e  ← server pipeline started from here at 01:36:22
01:44:58  0e3f8c679  hard_constraints_lint                   ┐
01:51:09  6eec1d99f  AWQ variant                             │ Other Claude
01:55:40  5cfe46276  parse_results                           │ pushed 12 commits
02:01:46  5bb888646  Phase 4 lightning_skip                  │ to origin between
02:03:23  b0843df14  preflight-hardening (my work)           │ 01:44 and 02:21
02:07:29  767773f30  Phase 5 docs                            │
02:09:35  0229ab0a8  full_preflight integration              │
02:13:10  22310f167  moonshot variant                        │
02:13:54  69cb19ab6  moonshot handoff                        │
02:16:40  fa5a7f576  SUBMISSIONS row                         │
02:19:50  770287904  lightning_skip tests                    │
02:21:24  cee60c00d  CLAUDE.md update                        ┘
~02:35              Phase A finishes → first commit_push attempt
                    `git push origin quant/w4a16` → REJECTED (non-fast-forward)
                    pre-fix `if git push | tail` used tail's exit code = 0
                    → reported "[commit_push] pushed OK"  (LIE)
~03:30              Phase B / etc → same rejection → same false "OK"
...                 (10+ unpushed commits stack on the server's local branch)
```

The script had **two design defects** that compounded:
1. **No exit-code check on `git push`** (Bug P1 below)
2. **No recovery from non-fast-forward** — script never `git pull --rebase`s

## Fixes applied on this branch

| File | Fix |
|---|---|
| `scripts/experiment_5h/commit_push.sh` | (P1) capture `git push` exit code separately from `tail`; (P1+) on non-fast-forward, try ONE `git pull --rebase` + retry; on rebase conflict, abort cleanly and tell the human exactly what to do. **(NEW) Mirrors `scripts/eval_results.csv` + `scripts/logs/5h_*/` to `/root/autodl-fs/zyn/backups/5h_live/` on every phase BEFORE the git commit, so results survive an AutoDL instance reboot (which wipes `/root/autodl-tmp/`). If push still fails after all retries, saves the unpushed commits as `git format-patch` files to NAS for replay.** |
| `scripts/experiment_5h/rescue_to_nas.sh` | **(NEW) Emergency one-shot.** SSH-and-run to snapshot the working tree's CSV + logs + unpushed commits to a timestamped dir on NAS. Read-only on the repo; safe to invoke even while a pipeline is running. Use this FIRST on any post-incident recovery. |
| `scripts/experiment_5h/pick_winners.py` | (P0) actually USE `--session`. Pre-fix: ignored the arg, would pick D1 winner from ANY earlier session (including smoke runs where 3-sample D1 had acc=0). Post-fix: refuses to look at rows outside the named session |
| `scripts/experiment_5h/parse_results.py` | (P2a) `parse_aggregate` no longer takes the LAST regex match blindly. Prefers JSON-style `"acc": N` > line-anchored `^acc = N` > inline. Plausibility-filters values to [0, 100] to reject token counts like `acc=65546`. (P2b) `find_predictions_jsonl` scopes to the experiment's log lifetime (ctime..mtime ± 60s) instead of a 12h backward window — otherwise every experiment in a 5h pipeline ends up matched to the same predictions.jsonl |
| `tests/test_experiment_5h.py` | 11 new unit tests pinning all of the above |

Full suite: **167 tests pass** (was 156 before this branch, +11 new). All
existing variant `--check-only` preflights still pass.

## CRITICAL: do not push to `quant/w4a16` until you have recovered the server pipeline's commits

**The server's local `quant/w4a16` branch likely has 8-10 unpushed commits**
with results from the 01:36 pipeline. If you push my fixes to
`origin/quant/w4a16` first, then SSH to the server and try to recover,
you'll have to merge through MY commits + the OTHER Claude's commits
before the server pipeline's commits go in. Even more divergence; same
class of bug we're trying to fix.

### Even more critical: AutoDL `/root/autodl-tmp/` is WIPED on reboot

If the AutoDL instance is rebooted or reclaimed BEFORE the unpushed commits
are recovered, **all of the 5h pipeline's results are gone forever**. The
server's local git history lives in `/root/autodl-tmp/zyn/sglang/.git` —
that whole tree gets nuked. The master log on NAS
(`/root/autodl-fs/zyn/logs/5h_full_20260523_013622.log`) survives, but
the structured CSV does not.

### Step 1 — SSH in and rescue NOW

The first thing you do after SSH is run the rescue script, BEFORE doing
anything that could disturb the working tree:

```bash
cd /root/autodl-tmp/zyn/sglang   # or actual server path
# Pull this branch first to GET the rescue script
git fetch origin fix/5h-run-plan-hardening
git checkout origin/fix/5h-run-plan-hardening -- scripts/experiment_5h/rescue_to_nas.sh

bash scripts/experiment_5h/rescue_to_nas.sh
# Writes /root/autodl-fs/zyn/backups/rescue_<TIMESTAMP>/ containing:
#   - eval_results.csv (the precious acc numbers)
#   - logs/ (per-phase log dirs)
#   - patches/*.patch (every unpushed commit as a git-replay-able file)
#   - git_state.txt (snapshot of git status + log for context)
```

After this script returns, **the data is safe even if the instance dies
in the next second**. NOW you can do the rebase recovery in step 2 without
panic.

### Step 2 — rebase the pipeline's commits onto current `origin/quant/w4a16`

```bash
git status
git log origin/quant/w4a16..HEAD --oneline    # unpushed commits (should match patches count)
tail -50 scripts/logs/5h_20260523_013622_full/master.log

git fetch origin
git rebase origin/quant/w4a16
# If conflicts: they're almost certainly in scripts/eval_results.csv
# (both sides appended different rows). Resolve by concatenating both
# blocks of rows, save, then: git add scripts/eval_results.csv && git rebase --continue
```

### Step 3 — push the recovered commits

```bash
git push origin quant/w4a16
```

If push STILL fails for whatever reason, all is not lost: the patches from
step 1 can be replayed on any other machine:

```bash
# On a fresh instance / your laptop:
git fetch origin && git checkout quant/w4a16
git am /root/autodl-fs/zyn/backups/rescue_<TIMESTAMP>/patches/*.patch
git push origin quant/w4a16
```

### Step 4 — merge this branch

```bash
git fetch origin fix/5h-run-plan-hardening
git checkout quant/w4a16
git merge --no-ff origin/fix/5h-run-plan-hardening
git push origin quant/w4a16
```

After step 3 and before step 4 is the right window to push this branch to
origin from this dev mirror if you prefer (no conflict at that point):
```bash
# On local dev mirror, after server-side recovery is done:
git checkout fix/5h-run-plan-hardening
git push origin fix/5h-run-plan-hardening
```

### What if the pipeline is still running?

The pipeline started 01:36 with ETA 06:30. If by the time you read this
it's >9 hours old and `ps aux | grep run_plan` shows it's still alive,
something is wedged — most likely a quant or eval step hung. Decide:
- **Let it finish + recover after** (preferred if anything is mid-eval)
- **Kill it, recover the partial results, restart**

```bash
# To check what it's doing now:
ps aux | grep -E "run_plan|local_eval|prepare_model|quantize"
tail -f scripts/logs/5h_20260523_013622_full/master.log
```

## What I deliberately did NOT do

- **Did not push this branch.** Pushing now would make recovery harder.
- **Did not modify `run_plan.sh`** (the orchestrator). Its `VAR=value
  run_exp ...` pattern looks suspicious but I tested it: bash DOES export
  prefixed vars to functions and their subprocesses. The Phase D pattern
  works as intended. The earlier fix commit `f884cb129` was for a
  DIFFERENT bug (`eval "env $WINNERS"` which would print env then exit).
- **Did not change the auto-commit cadence.** Auto-commit after each
  phase is the right granularity given the new pull-rebase recovery —
  rebase against newer origin should now be reliable.
- **Did not retroactively fix the running pipeline.** Cannot — the
  running pipeline is using the file contents it had when it started.
  Any change I make to `parse_results.py` or `commit_push.sh` here is
  picked up only on the NEXT pipeline run.

## Branch state

- Branch: `fix/5h-run-plan-hardening` (LOCAL ONLY)
- Base: `quant/w4a16` at commit `9e3f4a362` (current origin HEAD)
- Commits ahead of base: 1 (the fix commit, after I commit)
- Test suite: 167 / 167 passing

## Reference: how to verify the fixes work after merging

```bash
# Run the test suite (5 sec, no GPU)
python3 -m unittest discover tests

# Smoke-test pick_winners session filtering against a hand-crafted CSV
python3 scripts/experiment_5h/pick_winners.py \
    --csv /tmp/fake.csv --session NONEXISTENT --baseline-exp A_J0_baseline
# expect: empty output, stderr says "no rows for session NONEXISTENT"

# Smoke-test commit_push divergence handling (on a SCRATCH branch — don't
# do this on quant/w4a16)
git checkout -b scratch/test-commit-push
# ... make a fake commit that would diverge, run commit_push, see it
# either pull-rebase or fail loudly. NOT during a running pipeline.
```
