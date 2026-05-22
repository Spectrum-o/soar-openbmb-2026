# Unattended 5h Experiment Plan — Server-side Execution Guide

> Written 2026-05-22 evening by local Claude (no GPU). For server-side Claude
> on AutoDL to execute while the user sleeps.
>
> Goal: push W4A16 acc_ori from current 46.89 toward platform's correctness
> gate (likely ~80%). Plan covers 10+ experiments in ~5 GPU-hours, fully
> autonomous (auto commit + push after each step).

---

## TL;DR — what server-side Claude should do

```bash
# 1. Pull latest
cd /root/autodl-tmp/zyn/sglang   # or wherever the repo lives
git checkout quant/w4a16
git pull origin quant/w4a16

# 2. Cherry-pick the llm-compressor test script from parallel branch
git fetch origin parallel/non-gptqmodel-paths
git checkout origin/parallel/non-gptqmodel-paths -- scripts/test_llmcompressor_artifact.sh
chmod +x scripts/test_llmcompressor_artifact.sh

# 3. Activate env
source sglang_minicpm_sala_env/bin/activate

# 4. Preflight (5 min, no GPU compute)
bash scripts/experiment_5h/preflight.sh
# If any FAIL → fix before continuing. WARN about missing 5/14 artifact is OK.

# 5. Smoke test ALL experiments (~25 min, GPU)
#    Each experiment runs for max 120s — just verifies it boots.
bash scripts/experiment_5h/run_plan.sh --smoke 2>&1 | tee /tmp/5h_smoke.log
# Look for any FAILED in master log. Fix or skip-flag the broken ones.

# 6. Full 5h run (auto-commits + pushes after each experiment)
bash scripts/experiment_5h/run_plan.sh 2>&1 | tee /tmp/5h_full.log &
# User will sleep. Wake up to results in scripts/eval_results.csv.
```

---

## What the plan does (10 experiments)

| Phase | Exp | What | Knob | Re-quant? | Eval samples |
|---|---|---|---|---|---|
| A | A_J0_baseline | Reproduce 1849 locally | (none) | Yes (256 calib) | 150 |
| B | B_5_14_smoke | 5/14 llm-compressor artifact | (different tool) | No (use existing artifact) | 30 |
| B | B_5_14_full | Same, full eval if smoke ≥30 | — | No | 150 |
| C | C_rep_pen_sweep | repetition_penalty 1.0/1.05/1.10 | sampling-time | No | 30 each |
| D1 | D1_sym_false | Asymmetric quant | `GPTQ_SYM=False` | Yes | 60 |
| D2 | D2_damp_01 | Higher GPTQ damp | `GPTQ_DAMPENING_FRAC=0.1` | Yes | 60 |
| D3 | D3_calib_512 | More calib samples | `NUM_CALIB=512` | Yes | 60 |
| D4 | D4_calib_len_16k | Longer per-prompt | `MAX_CALIB_LEN=16384` | Yes | 60 |
| E | E_winners_combo | Stack Phase D winners | (auto-detected) | Yes | 150 |
| F1 | F1_multi_adaptive | v25 multi-adaptive | (existing variant) | Yes | 60 |
| F2 | F2_g64 | group_size=64 | (existing variant) | Yes | 60 |

Estimated total: **~4.5-5 hours of GPU time**.

---

## What's been added to the repo

```
scripts/experiment_5h/
├── run_plan.sh             # Master orchestration (--smoke or full)
├── preflight.sh            # Verify env, models, scripts before running
├── commit_push.sh          # Auto-commit + push wrapper (called after each exp)
├── parse_results.py        # Parse one experiment's log → CSV row
├── pick_winners.py         # Read CSV, pick Phase D winners for Phase E
└── rep_penalty_sweep.sh    # Phase C: 3-value sweep without re-quant

# Modified (env var driven knobs added):
submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py
  → make_quant_config() now reads:
      GPTQ_SYM             (default "True")
      GPTQ_DAMPENING_FRAC  (default unset)
  → write_sglang_compatible_quant_config() reads:
      GPTQ_SYM             (must match above)
```

Other env vars already supported by `prepare_model.sh` (pre-existing):
- `NUM_CALIB` (defaults to 256)
- `MAX_CALIB_LEN` (defaults to 8192)
- `CALIB_WINDOW_MODE` (tail / multi-adaptive)
- `DISABLE_CHAT_TEMPLATE` (0/1)

---

## CRITICAL: what knobs are SAFE vs RISKY

### Safe (validated, just config flips)

- `GPTQ_SYM=False` — asymmetric quant; production standard per RedHat recipes
- `GPTQ_DAMPENING_FRAC=0.1` — production uses 0.1, our default 0.01
- `NUM_CALIB=512` / `NUM_CALIB=1024` — more samples
- `MAX_CALIB_LEN=16384` — longer per-prompt window
- `CALIB_WINDOW_MODE=multi-adaptive` — multi-window mode (variant v25 ready)
- `repetition_penalty` sweep — sampling-time, no re-quant

### Risky (NOT included in 5h plan, requires GPU validation first)

- `GPTQ_SKIP_FIRST_LAYERS=N` / `GPTQ_SKIP_LAST_LAYERS=N` —
  **PROVISIONALLY NOT IMPLEMENTED** in this 5h plan because per-layer skip
  requires QUANT-TIME and LOAD-TIME alignment (v14 KeyError pitfall).
  Local Claude removed the half-baked write-only implementation to avoid
  giving server-side Claude a broken default. See "Future work" below.
- Switching to llm-compressor toolchain — Phase 4+ in the roadmap;
  significantly more work than env-var flip.
- AWQModifier — needs llm-compressor pipeline first.

---

## Smoke mode behavior (what `--smoke` actually does)

`run_plan.sh --smoke` runs the entire pipeline but:
- Sets `NUM_CALIB_EXP=8` (8 calibration samples instead of 256-1024)
- Sets `SAMPLES_FULL=3` (3 eval samples instead of 150)
- Uses `timeout 120` on every experiment — kills if still running after 2 min

Goal: **verify each experiment can BOOT through quant + serve + eval without
crashing**. Doesn't care about acc values. ~25 min total for all 10
experiments.

After smoke passes, run without `--smoke` for the real 5h plan.

---

## How auto commit-push works

After EACH experiment:
1. `parse_results.py` reads the experiment's log file, extracts acc/per-task, appends a row to `scripts/eval_results.csv`.
2. `commit_push.sh` stages `scripts/eval_results.csv` + `scripts/logs/5h_<session>/` and commits with message `"5h exp <name> (<session>)"`.
3. Pushes to `origin/quant/w4a16`.

If push fails (network blip), the commit is local and next iteration's push will catch up (multiple commits go in one push).

If commit fails entirely (e.g. nothing changed), it's a no-op.

**Failure handling**: `set -uo pipefail` is on (not `-e`). One bad experiment doesn't block the rest. Each experiment's log captures its own traceback.

---

## How Phase E picks winners

After Phase D (4 single-knob ablations), `pick_winners.py` reads
`scripts/eval_results.csv` and:

1. Finds the most recent `A_J0_baseline` row, reads its `acc_ori` (or `acc`).
2. For each Phase D experiment, checks if its `acc_ori` exceeds baseline by ≥ 2.0 pp.
3. Emits the env-var assignments for winners as a space-separated string.

Example output for stdin → eval:
```
GPTQ_SYM=False GPTQ_DAMPENING_FRAC=0.1
```

If no winners detected (all knobs near baseline), Phase E falls back to
"stack all 3 safe knobs" (`GPTQ_SYM=False GPTQ_DAMPENING_FRAC=0.1 NUM_CALIB=1024`)
so we still have an Phase E result.

---

## Resume support

`run_plan.sh --from <phase>` skips phases alphabetically before that.

```bash
# E.g. if Phase A succeeded but D2 crashed mid-way, resume from D:
bash scripts/experiment_5h/run_plan.sh --from D
```

Note: this doesn't restart a partial Phase D; it starts fresh from D1.

---

## What might break (known unknowns)

1. **Phase B (5/14 artifact) requires `test_llmcompressor_artifact.sh`** to be cherry-picked from `parallel/non-gptqmodel-paths` (see step 2 above). Without it, Phase B is skipped.
2. **5/14 artifact at `/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/`** may or may not still exist. Preflight warns if missing.
3. **Phase C (rep_penalty)** depends on which mechanism SGLang/SOAR supports. The sweep script probes 3 cases:
   - eval_model.py mentions `repetition_penalty` → forward via env
   - `sglang.launch_server --repetition-penalty` exists → server arg
   - Neither → SKIP Phase C with a log message (no failure, just no data)
4. **Phase D / E quant time**: each re-quant is ~15-30 min. If GPU has memory issues with NUM_CALIB=512 or MAX_CALIB_LEN=16384, that experiment will OOM. The log captures it; the plan continues.
5. **Auto commit-push may push WIP/broken intermediate state** to origin/quant/w4a16. That's intentional — partial results > no results. User can clean up after.

---

## Future work (NOT in this 5h plan)

These are real candidates per the research summary but require more setup:

- **Layer-skip mixed precision** (skip first 2 + last 2 MLP). Needs verified
  alignment between quant-time and load-time dynamic dict. Skipped to avoid
  v14-style KeyError. **If you have a 1-hour spike during the 5h plan, write
  a small test that quantizes with GPTQModel's dynamic skip + verifies
  safetensors doesn't contain .qweight for the skipped layers.**
- **Switch to llm-compressor toolchain** (AWQ, GPTQ-via-llm-compressor).
  See `experiments/QUANT_STRATEGY_RESEARCH_20260522.md` Phase 2-3.
- **SpinQuant R1+R2 offline rotation** — 1-day spike per research doc.

---

## What server-side Claude should report back

When user wakes up:

1. **Read `scripts/eval_results.csv`** — the table of all experiments.
2. **Identify the highest-acc_ori row** that beats J0 baseline.
3. **Compute the platform projection**: if local acc_ori = X, expected platform = X ± 2pp (based on 1849 evidence local↔platform delta).
4. **Reply with a one-page summary**:
   - 1-line winner of Phase D / E
   - Whether Phase B (5/14 llm-compressor) was viable
   - Whether Phase C (rep_penalty) showed any signal
   - Recommended next platform submission knob set
   - Any experiments that failed and need re-run

The user will read this when they wake up and decide what to submit to the platform tomorrow morning.

---

## Logs structure

```
scripts/
├── eval_results.csv                              # one row per experiment
└── logs/
    └── 5h_20260523_010000_full/                  # per-session subdir
        ├── master.log                            # plan orchestration log
        ├── A_J0_baseline.log                     # individual experiment logs
        ├── B_5_14_smoke.log
        ├── B_5_14_full.log
        ├── C_rep_pen_sweep.log
        ├── D1_sym_false.log
        ├── D2_damp_01.log
        ├── D3_calib_512.log
        ├── D4_calib_len_16k.log
        ├── E_winners_combo.log
        ├── F1_multi_adaptive.log
        └── F2_g64.log
```

All under git tracking (via `commit_push.sh`), so the user can `git log` and see exactly when each experiment finished, including the auto-commit message format `"5h exp <name> (<session>)"`.

---

## Quick reference — run from scratch

```bash
cd /root/autodl-tmp/zyn/sglang
git checkout quant/w4a16
git pull origin quant/w4a16
git fetch origin parallel/non-gptqmodel-paths
git checkout origin/parallel/non-gptqmodel-paths -- scripts/test_llmcompressor_artifact.sh
chmod +x scripts/test_llmcompressor_artifact.sh
source sglang_minicpm_sala_env/bin/activate
bash scripts/experiment_5h/preflight.sh && \
bash scripts/experiment_5h/run_plan.sh --smoke && \
bash scripts/experiment_5h/run_plan.sh
```

Five commands, ~5 hours wall-clock, fully autonomous.

---

**End of guide.**
