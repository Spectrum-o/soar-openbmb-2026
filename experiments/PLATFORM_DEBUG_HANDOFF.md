# Platform debug handoff — 2026-05-21

> **You are a fresh Claude session** running on the AutoDL server with
> port-forwarded VS Code. **Read this first**, then `SUBMISSIONS.md`,
> then `experiments/PLAN.md` (NB: PLAN.md's mental model is partly
> stale — see "What this supersedes" below).

---

## State of the world (2026-05-21 ~15:00 local)

Two SOAR platform submissions today, **both returned `acc_ori=0, final_score=0`**:

| Tarball | Description | Local acc (perf_public_set 150) | Platform acc | Notes |
|---|---|---|---|---|
| `soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v21.tar.gz` | MLP-only GPTQ, qzeros+calib fix baked in | **49.00%** | **0** | 13/150 catastrophic repetition loops on niah haystack |
| `soar_gptqmodel_v17_minconfig_full_attn_submission_20260520_v22.tar.gz` | Full-attn GPTQ (q/k/v/o + MLP), same qzeros+calib fix | not measured | **0** | bench timings literally identical to v17 (S1=626 S8=998 Smax=2290) |

**For context**: `soar_rtn_sym_w4a16_g128_marlin_submission_20260519_scalefix.tar.gz` (RTN, no GPTQModel) scored **~40 on the same platform**.

### What we know

- v21 (MLP-only) and v22 (full-attn) **both 0** → it's NOT the module set
- bench timings identical across v17/v21/v22 → SGLang serves, Marlin kernels run, generation produces tokens at the same speed → content is systematically broken, not "model failed to load"
- RTN works → `gptq_marlin` loader path on platform is fine
- v21 **local 49** vs **platform 0** with theoretically the same artifact → quantize-time pipeline differs between AutoDL and platform
- SOAR `eval_model.py` has **no hard `acc < 80 → 0` gate** (formula: `min(avg_score/80*100, 100)`). Local 49 should map to ~61 on platform. Platform giving 0 means the model is generating garbage on the private set, not failing a gate.
- `perf_private_set.jsonl` and `perf_public_set.jsonl` share length distribution and task types (per OpenBMB/SOAR-Toolkit README) — distribution drift is a weak explanation

### Best-fit root cause hypothesis (H1)

`fix_qzeros_for_marlin()` **silently no-op'd on the platform** because the platform's gptqmodel version wrote the quantized model in a layout the function didn't recognize. Specifically suspected paths:

1. **Single-file output**: glob `model-*.safetensors` missed `model.safetensors` (no shard suffix). 6.5GB W4A16 model fits in one shard if max_shard_size >= 10GB (HF transformers < 4.50 default).
2. **dtype mismatch**: function required `tensor.dtype == torch.int32`; gptqmodel 7.0.x may write `uint32` → skipped entirely.
3. **Non-uniform qzeros**: function required `unique == [0x77777777]` exactly; any other layout went to WARN-and-skip branch.

Local AutoDL quant logs (`scripts/logs/local_eval_quant_..._1779289794.log`) show the function patching 96 qzeros tensors across 3 shards on `1779289794`'s artifact (the one that locally evaled to 49). So fix works on AutoDL. Platform = unknown, no logs available.

### Secondary hypothesis (H2)

`prepare_env.sh` install gate was loose: `gptqmodel>=7.0,<8.0`, and the install was **skipped** if any 7.x was already on the platform's base env. Platform may have had pre-installed `gptqmodel` at a different .x point release than AutoDL's `7.0.0`.

### Tertiary hypothesis (H3) — python ABI mismatch

Confirmed from local logs (2026-05-21): **AutoDL runs Python 3.12.13**, while the platform per prepare_env.sh comments + the bundled `flash_attn-...cp310-cp310...whl` wheel is **Python 3.10**. gptqmodel 7.0.0 (and safetensors) install different binary wheels per Python ABI. If the 3.10 wheel has different default `max_shard_size`, different qzeros packing in a C++ extension, or different safetensors output schema, the on-disk artifact diverges before our fix function ever runs. The new `parse_quant_diagnostic.py` tool will surface a `python=` version mismatch in the diff report.

---

## What's already pre-processed (this turn)

All files committed locally on the `quant/w4a16` branch (or staged — run `git status` to confirm). **Sync with `git pull` on the server.**

| File | Change |
|---|---|
| `submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py` | Hardened `fix_qzeros_for_marlin()`: globs `*.safetensors`, handles uint32, per-element replacement, prints file inventory + sample values, HARD EXITS if no qzeros seen or post-write check fails. |
| `submission_gptq_v17_minconfig/quantize_gptqmodel_w4a16.py` | Same hardened function applied. |
| `submission_gptqmodel_calib_w4a16/prepare_env.sh` | Pinned `gptqmodel==7.0.0` (was `>=7.0,<8.0`). Added `python_pkg_exact_version` helper. Force-reinstall if installed version != pinned. Prints currently-installed version before forcing. |
| `submission_gptq_v17_minconfig/prepare_env.sh` | Same change (file is identical to v21's). |
| `submission_gptqmodel_calib_w4a16/prepare_model.sh` | Added DIAGNOSTIC blocks before AND after quantize: script dir listing, input dir listing, calib jsonl size/row count/first 200 chars, output dir listing, safetensors find, quantize_config.json dump. |
| `submission_gptq_v17_minconfig/prepare_model.sh` | Same diagnostic blocks. |

All four scripts pass `python3 -m py_compile` and `bash -n`.

### What the hardened fix function does differently

| Old behavior | New behavior |
|---|---|
| `glob("model-*.safetensors")` — misses single-file | `glob("*.safetensors")` — catches both layouts |
| `dtype != torch.int32` → silently skip | `uint32` accepted via `.view(torch.int32)`; other dtypes WARN with name |
| `unique == [BAD]` exact match | Per-element `mask = (tensor == BAD)`; replaces all matches, leaves rest alone |
| `total 0` → continue silently | `total 0 qzeros seen` → `raise RuntimeError(...)` so platform marks failure (does NOT consume slot per user rule) |
| no post-write check | Reload first shard, assert first qzeros tensor reads back as 0x88888888; raise if not |
| no diagnostic prints | Lists output_dir contents (filenames + sizes), prints sample unique values + hex of first 5 qzeros tensors |

---

## What this supersedes

**`experiments/PLAN.md` (night-work runbook) is partially obsolete**:

- PLAN.md's mental model was "v21 local 49 → platform 0 = repetition collapse on hidden eval". That's only **part** of the story — v22 (full-attn, different module set, same calib+qzeros recipe) also = 0, which the night-work mental model didn't account for.
- PLAN.md's Exp B (chat tpl OFF) and Exp C/D/E experiments are still useful as **calibration-recipe** ablations, but they don't address the platform-vs-local pipeline gap.
- The "drop fwe → real perf 36%" framing in PLAN.md still holds for predicting LOCAL→PLATFORM accuracy drops, just not for explaining the absolute zero.

**Keep using PLAN.md for**:
- Exp C/D/E (calibration ladder) — these are good experiments to run LOCALLY after we confirm the platform pipeline works at all
- The "tools/" inventory at the bottom — all tools are current and useful

---

## What to do next (priority order)

### Step 1: Sync + sanity check (5 min, no GPU)

```bash
cd /root/autodl-tmp/zyn/soar/sglang
git pull

# Verify all 6 changed files are present
git log --oneline -5
git diff HEAD~1 -- submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py | head -100

# Inspect the EXISTING v21 local artifact to confirm hardened fix would
# still recognize it (sanity check, no quantization needed)
python3 tools/inspect_quant_artifact.py \
    --artifact /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized
```

Expected: `inspect_quant_artifact.py` reports qzeros all = 0x88888888 across 3 shards.

### Step 2: Re-quant LOCALLY with hardened script (20-25 min, 1 GPU)

```bash
# This re-quantizes from scratch using the hardened script. Should
# produce IDENTICAL output to the existing local artifact, but with the
# new diagnostic prints in the log so we know exactly what gptqmodel
# wrote (and confirm the assertion path doesn't fire spuriously).
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --force-requant \
    --eval-data /root/autodl-tmp/zyn/soar/sglang/submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \
    --num-samples 150 \
    2>&1 | tee /root/autodl-fs/zyn/logs/v23_localcheck_$(date +%Y%m%d_%H%M%S).log
```

**Read the log for these lines**:
- `[prepare_model] DIAGNOSTIC: ...` (new sh-level)
- `[qzeros-fix] output_dir inventory ...`
- `[qzeros-fix] sample: ...` (this is the key new info)
- `[qzeros-fix] POST-CHECK ...`
- `[qzeros-fix] OK` at the end

**Expected**: total qzeros patched = 96 (matches the old log). POST-CHECK shows `first_val=-2004318072 (0x88888888)`. Local acc ≈ 49.

**Failure mode A**: if the hardened script HARD EXITS at any RuntimeError, the assertion logic itself is wrong — debug here before packing any tarball.

**Failure mode B**: if local acc drops vs the 49 baseline, something in the hardened replacement logic regressed — diff the actual qzeros byte-by-byte vs the old artifact.

### Step 3: Pack v23 + v24 tarballs

```bash
# v23 = MLP-only with hardened fix
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16 \
    --suffix _v23 \
    --output-dir .

# v24 = full-attn with hardened fix
python3 tools/pack_submission.py \
    --variant submission_gptq_v17_minconfig \
    --suffix _v24 \
    --output-dir .

# Pre-flight check
python3 tools/quant_config_validator.py \
    --tarball ./soar_gptqmodel_calib_w4a16_mlp_only_submission_*_v23.tar.gz
python3 tools/quant_config_validator.py \
    --tarball ./soar_gptqmodel_v17_minconfig_full_attn_submission_*_v24.tar.gz
```

### Step 4: Submit v23 ONLY (one slot)

**Reasoning**: v23 is the cheaper test of the hypothesis. If H1 is right (qzeros silent no-op was the bug), v23 should clear ≥40 on platform with MLP-only's smaller error. If v23 still = 0, H1 is wrong and we need to look elsewhere; no point burning a second slot on v24 yet.

**The key thing to read in the platform log** (this is why the diagnostic was added):
- `[prepare_env] gptqmodel ... already installed; skipping` vs `forcing gptqmodel==7.0.0 (currently: 7.X.Y)` — tells us if pinning bit
- `[versions] python=...` — confirm 3.10 vs AutoDL's 3.12 (H3)
- `[prepare_model] DIAGNOSTIC: output dir contents after quantize` — single file vs 3 shards
- `[qzeros-fix] sample: model.safetensors::... unique[:6]=[...] hex=[...]` — what the platform's gptqmodel actually wrote
- Any `[qzeros-fix] FATAL:` traceback — exact failure mode

**Easiest way to consume all of this**: pipe both logs through the new parser:

```bash
python3 tools/parse_quant_diagnostic.py \
    --input /path/to/platform_run.log \
    --compare /path/to/local_run.log \
    --output-md /tmp/diff_platform_vs_local.md
cat /tmp/diff_platform_vs_local.md
```

The diff report ends with a "Verdict heuristic" section that names the most likely divergence (version drift / file-count drift / FATAL on one side / qzeros-patched count mismatch).

### Step 5 — branches based on v23 result

| Platform v23 result | What it means | Next action |
|---|---|---|
| `acc_ori ≥ 40` | H1 confirmed; qzeros was the bug | Submit v24 (full-attn) as parallel track; both should land |
| `acc_ori == 0`, `[qzeros-fix] FATAL` raised | Hardened fix exited cleanly — log will name the new layout | Patch fix function for new layout, repack v25 |
| `acc_ori == 0`, `[qzeros-fix] OK` printed | qzeros fix works on platform too; the gap is elsewhere | Fall back to PLAN.md Exp C/D (sampling cap + repetition_penalty) or Exp E (multi-adaptive calib) |
| Platform crashed before quantize finished | environment issue — read DIAGNOSTIC blocks | If pin install failed, fall back to source build or bundle wheel |

---

## What to NOT do

- **Don't change the calibration recipe** (chat template, truncation, window mode) before confirming the qzeros pipeline works. The PLAN.md ablations are red herrings if every artifact has broken qzeros.
- **Don't add more "fixes" before reading platform logs.** The whole point of this turn's work is to make the next platform run produce diagnostic output. If you fix more things blindly, you can't tell which fix mattered.
- **Don't burn slots on multi-experiment submissions.** Submit one tarball, read the log, then submit the next. Slots are scarce.
- **Don't `git push --force` or otherwise rewrite history.** Watchdog snapshots from night-work are valuable.
- **Don't change SGLang launch args.** Identical to RTN's launch args; not the suspect surface.

---

## Files / tools the server-side Claude needs to know about

| Path | Purpose |
|---|---|
| `SUBMISSIONS.md` | Canonical log of all 22+ submissions, hard constraints, open questions |
| `experiments/PLAN.md` | Night-work runbook (calib-recipe experiments; partially obsolete per above) |
| `docs/calibration_gotchas.md` | qzeros bug / truncation-side / repetition collapse knowledge |
| `docs/awq_fallback_plan.md` | If GPTQ dies completely, 3-step AWQ alt |
| `tools/inspect_quant_artifact.py` | qzeros + scales sanity check (CPU, no model load) |
| `tools/repetition_analyzer.py` | Per-prediction loop detection (built last night) |
| `tools/parse_quant_diagnostic.py` | **NEW (2026-05-21)**: parse the new DIAGNOSTIC + qzeros-fix SUMMARY blocks from a quant log. Use `--compare` to diff platform log vs local log; the heuristic verdict at the end of the diff report points to the most likely divergence. Backward-compatible with legacy `[qzeros-fix]` format. Has 22 unit tests in `tests/test_parse_quant_diagnostic.py`. |
| `tools/quant_config_validator.py` | Tarball pre-flight (qzeros / sed / SGLANG_SERVER_ARGS) |
| `tools/pack_submission.py` | Auto-pack v23/v24/etc |
| `scripts/logs/local_eval_quant_*.log` | Local quantization runs; grep `qzeros-fix` for fix history |
| `outputs/20260520_234226/predictions.jsonl` | v21 local eval (150 samples, the one that scored 49) |
| `/root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized/` | The actual v21 quant artifact on AutoDL |
| `/root/autodl-fs/zyn/logs/` | Watchdog snapshots |

User memory at `/home/zyn/.claude/projects/-home-zyn-program-2026-Spring-mlsys-contests-Sora-2026-sglang/memory/` is auto-loaded — has user background + competition context + hard constraints from prior failures.

---

## TL;DR

1. **v21+v22 both = 0 on platform** despite v21 local = 49. RTN = 40. So it's neither module-set nor "all 4-bit quant is broken".
2. **Best fit**: `fix_qzeros_for_marlin()` silently no-op'd on the platform because the platform's gptqmodel wrote a layout the function didn't expect (likely single-file `model.safetensors` + glob mismatch, or different dtype).
3. **Pre-processed**: hardened the fix to (a) catch any safetensors layout, (b) HARD EXIT on suspicious states, (c) print diagnostic info; pinned `gptqmodel==7.0.0`; added shell-level diagnostic prints before/after quantize.
4. **Next step**: re-quant locally to confirm no regression → pack v23 → submit one slot → read the new diagnostic output → that tells us exactly what to fix next.
