# `experiments/` — index

Reading-order navigation for the docs in this directory.

## "I just opened this repo. What do I do?"

| Step | Read | Purpose |
|---|---|---|
| 1 | `STATE_OF_PLAY.md` | 30-sec project context: what we tried, what worked, what's pending |
| 2 | `MORNING_PLAYBOOK.md` | Single-page action guide. **If v23 platform result is in hand, this is the whole playbook.** |

These two files cover ~95% of operational scenarios. The rest is reference.

## Reference docs (in order of operational relevance)

| Doc | When to read |
|---|---|
| `FULL_W4A16_MIXED_SKIP30_31_DOWN_FAILURE.md` | Negative result for full-g64 mixed BF16 skip of layers 30/31 `mlp.down_proj`; read before trying more full-W4A16 sensitivity runs. |
| `V24_PLAN.md` | Before/after submitting v24+. Decision tree by v23 outcome with concrete pack commands for each branch. |
| `V23_PLATFORM_LOG_CHECKLIST.md` | When v23 (or any future GPTQModel) platform result comes back. 8 grep blocks attribute the result to a hypothesis. |
| `AWQ_FEASIBILITY_ASSESSMENT.md` | Only if v23 platform acc < 10 AND H1/H2/H4 fixes don't move the needle. Then trigger the docs/awq_fallback_plan.md 4-step process. |
| `V23_VS_RTN_TARBALL_AUDIT.md` | When inspecting why one tarball worked but another didn't. File-level diff between RTN-pass (acc=42) and v22-fail (acc=0). |
| `PLATFORM_DEBUG_HANDOFF.md` | Originally for handing off to server-side Claude on a fresh instance. Still useful for context: server-side environment setup, hypothesis discipline warning. |

## Historical / less-current

| Doc | Status |
|---|---|
| `PLAN.md` | Night-work runbook from 2026-05-21. **Partially stale** — predates the H4 finding. Mental model says "v23 platform=0 is repetition collapse" which was later refined to "H1+H4 are the prime hypotheses, repetition is a SYMPTOM of bad quant not the root cause." Useful for the Exp E/F/G ladder (calibration tunes) but skip the v17-era hypothesis ranking. |

## Code artifacts (not docs)

| Path | Purpose |
|---|---|
| `op_fusion_verify.patch` | Drop-in `git apply` on top of `perf/op-fusion` branch. Adds runtime norm+add correctness check gated by `MINICPM_FUSION_VERIFY=1` env. Use when ready to GPU-validate op-fusion before merging into a v25+ variant. |

## Cross-references to outside experiments/

- `SUBMISSIONS.md` at repo root — canonical chronological submission log + hard-constraints table.
- `docs/calibration_gotchas.md` — qzeros bug history, truncation-side, repetition collapse, fp16 sed.
- `docs/awq_fallback_plan.md` — original AWQ fallback plan (pre-feasibility-check).
- `W4A16_README.md` at repo root — high-level competition + variant intro.
- `NIGHT_WORK_SUMMARY.md` at repo root — 2026-05-21 night-work changes (multi-adaptive calib + bug fixes).

## Reading flowchart

```
Just woke up, platform result in hand
  ↓
MORNING_PLAYBOOK.md → bash scripts/v24_auto.sh
                  ↓
                Done? → submit produced tarball
                Confused? → STATE_OF_PLAY.md → variant-dir status
                Variant fails preflight? → V23_PLATFORM_LOG_CHECKLIST.md
                v24_auto recommends MANUAL_FIX? → V24_PLAN.md branch 2/4

GPU available, want to test op-fusion before merging?
  ↓
op_fusion_verify.patch → apply + run with MINICPM_FUSION_VERIFY=1

v23 platform acc < 10 across multiple variants?
  ↓
AWQ_FEASIBILITY_ASSESSMENT.md → 4-step plan (10-min smoke test first)

Wondering what file does what?
  ↓
STATE_OF_PLAY.md → "Tooling inventory" + "Documents inventory" tables
```
