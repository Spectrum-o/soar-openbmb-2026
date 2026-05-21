# v24 plan — branches by v23 platform outcome

> Pre-prepared so that whenever v23 returns from the SOAR platform
> (~5h after submission), the next iteration is a paste-and-run away
> rather than a fresh planning session.

## Decision tree

```
v23 platform result
│
├── acc >= 30 (H1 + H4 fix worked, room to push for ceiling)
│   ├── v24-perf: v23 + chunked-prefill 65K (+83% throughput per
│   │             config/chunked-prefill-tuned measurement)
│   │             → variant dir: submission_gptqmodel_calib_w4a16_v24/
│   │             → see that dir's README_SUBMISSION.md for pack steps
│   │
│   ├── v25-perf-fusion: v24-perf + op-fusion (~10-15% additional
│   │                     decode speedup; requires prior GPU validation)
│   │                     → not built yet; build by forking v24 dir
│   │                       with sglang/python/sglang/srt/models/minicpm.py
│   │                       replaced by perf/op-fusion's version
│   │                     → REQUIRES: MINICPM_FUSION_VERIFY=1 run passes
│   │                       (norm_diff < 1e-3 for ~50 layers)
│   │
│   └── parallel calibration push for accuracy (R5 ceiling):
│       Exp E: multi-adaptive calib + 300 samples (PLAN.md)
│       Exp F: --group-size 64 / --desc-act (PLAN.md)
│
├── acc == 0, log shows [qzeros-fix] FATAL
│   └── New gptqmodel layout caught by hardened fix. Read FATAL
│       message + sample hex values from log. Extend
│       fix_qzeros_for_marlin in submission_gptqmodel_calib_w4a16/
│       quantize_gptqmodel_w4a16.py to handle the new pattern. Repack
│       v23 itself (not a v24 fork) since this is a v23 fix.
│
├── acc == 0, log shows [qzeros-fix] OK + bench timing ~ v22
│   └── H1 + H4 were not the platform=0 cause. Try ONE single-variable
│       change per submission via a NEW fork dir, NOT by mutating v23:
│       a. submission_gptqmodel_calib_w4a16_v24_no_dtype_key/
│          fork of v23 with `dtype: "float16"` removed from
│          quantize_gptqmodel_w4a16.py's
│          write_sglang_compatible_quant_config (keep torch_dtype).
│          RTN doesn't set the new HF 5.x key.
│       b. submission_gptqmodel_calib_w4a16_v24_pin_transformers/
│          fork of v23 that bundles a known-good transformers wheel
│          and force-reinstalls in prepare_env.sh. Forces both AutoDL
│          and platform onto the same version.
│       c. submission_gptqmodel_calib_w4a16_v24_bits8/
│          fork of v23 with --bits 8 in prepare_model.sh. If W8A16
│          works at 8-bit and fails at 4-bit, the platform's 4-bit
│          kernel is the issue.
│
└── acc == 0, sglang crashed before bench
    └── Read tracesback from server log. Usually one of:
        - dtype mismatch (sed-patch missed a file)
        - KeyError on a quantized module name (dynamic skip rules
          don't match what gptqmodel produced)
        - flash_attn missing (wheel install failed)
        These are all addressable from V23_PLATFORM_LOG_CHECKLIST.md
        Block 6/7. Patch v23's quantize_gptqmodel_w4a16.py or
        prepare_env.sh in place; this is a bug fix, not a fork.
```

## Why "fork dir" not "mutate v23"

v23 submission tarball was packed from a specific source dir state. If
we mutate that dir, we lose the ability to re-pack v23 (or any other
variant that inherited from v23's state) deterministically. Every
follow-up variant gets its OWN dir, with symlinks back to v23 for any
file it doesn't change. This keeps the diff between variants visible
in `git diff` between dirs and in `ls` output:

```
submission_gptqmodel_calib_w4a16/         ← v23 source, locked
submission_gptqmodel_calib_w4a16_v24/     ← v24-perf, 2 tracked files (README + prepare_env.sh)
submission_gptqmodel_calib_w4a16_v24_no_dtype_key/    ← (if built) 3 tracked files
...
```

Future bug fixes to the quant script flow to all variants via the
symlinked `quantize_gptqmodel_w4a16.py`. Per-variant deltas stay
minimal and visible.

## Concrete commands by branch

### Branch 1: v23 acc >= 30 → v24-perf

The v24 variant dir already exists at
`submission_gptqmodel_calib_w4a16_v24/` (preflight-clean as of this
commit). Just pack:

```bash
cd /root/soar/sglang
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v24 --check-only
# expect: [check] /path/...v24: OK (no problems)

python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v24 \
    --suffix _v24 --output-dir .

# Optional sanity run (re-quant + eval ~65 min on GPU; the artifact
# will be identical to v23's since the only diff is prepare_env.sh's
# launch flags, NOT the quant pipeline):
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16_v24 \
    --skip-quant \
    --eval-data submission_gptqmodel_calib_w4a16_v24/perf_public_set.jsonl \
    --num-samples 150

# Submit the v24 tarball. Expected: acc_ori ≈ v23's, benchmark_duration
# drops 50%+ across S1/S8/Smax.
```

### Branch 2: acc == 0 + [qzeros-fix] FATAL

```bash
# Read the FATAL line and the sample hex values
grep -A2 "qzeros-fix.*FATAL\|qzeros-fix.*sample" /path/to/platform_v23.log

# Patch submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py's
# fix_qzeros_for_marlin function to handle the new pattern. The hardened
# function already prints sample hex values; use those to extend the
# replacement logic.

# Repack v23 itself (not a fork — this is a v23 bug fix):
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16 \
    --suffix _v23b --output-dir .
```

### Branch 3: acc == 0 + qzeros OK → build a single-variable fork

For each hypothesis, build a new dir mirroring the v24 pattern:

```bash
V23=submission_gptqmodel_calib_w4a16
NEW_VARIANT=submission_gptqmodel_calib_w4a16_v24_no_dtype_key   # or other

mkdir -p "$NEW_VARIANT"
# Symlink the shared assets
for f in sglang flash_attn-*.whl perf_public_set.jsonl \
         prepare_env.sh prepare_model.sh; do
    ln -sf "../$V23/$f" "$NEW_VARIANT/$f"
done

# Take a writable copy of just the file you want to modify
rm "$NEW_VARIANT/quantize_gptqmodel_w4a16.py"
cp "$V23/quantize_gptqmodel_w4a16.py" "$NEW_VARIANT/"

# Make the surgical change. For no_dtype_key:
#   In write_sglang_compatible_quant_config, remove the line:
#     config["dtype"] = "float16"
#   Keep: config["torch_dtype"] = "float16"

# Write a v24-specific README explaining the precondition + diff

# Preflight + pack
python3 tools/pack_submission.py --variant "$NEW_VARIANT" --check-only
python3 tools/pack_submission.py --variant "$NEW_VARIANT" \
    --suffix _v24_no_dtype_key --output-dir .
```

### Branch 4: acc == 0 + sglang crashed early

This is the easiest to debug. The crash traceback names the bug.

- `RuntimeError: query and key must have the same dtype` → sed-patch
  didn't apply somewhere. Check V23_PLATFORM_LOG_CHECKLIST.md Block 6.
- `KeyError: model.layers.X.self_attn.Y.weight` → quantize_config.json's
  `dynamic` skip patterns don't match what gptqmodel emitted. Update
  the regex in `make_quant_config` to match the platform's gptqmodel
  module names.
- `ImportError: flash_attn` → wheel install failed; check the wheel
  bundled in submission dir is the right cp310 version.

Patch v23 in place; repack as v23b.

## Op-fusion: deferred to v25

`perf/op-fusion` is CPU-equivalence-verified but not GPU-validated. To
bring it in:

1. Run GPU validation locally (one prompt with `MINICPM_FUSION_VERIFY=1`);
   confirm per-layer norm_diff < 1e-3.
2. Once validated, build a v25 dir as a fork of v24 with the bundled
   `sglang/python/sglang/srt/models/minicpm.py` REPLACED (not symlinked)
   by `origin/perf/op-fusion`'s version. The other v24 symlinks (sglang/
   subtree minus minicpm.py, flash_attn.whl, prepare_env.sh) stay as-is.
3. The replacement is structural — v25's sglang/ can't be a single
   symlink to v23's; you need to copy v23's sglang/ tree then overlay
   the op-fusion minicpm.py. Document the procedure in v25's README.

## What pre-prep saves vs what it doesn't

Pre-prepped tonight:
- ✅ `submission_gptqmodel_calib_w4a16_v24/` — Branch 1 (v24-perf) ready
- ✅ This decision tree
- ✅ `tools/pack_submission.py` whitelist for v24+ chunked-prefill 65K
- ✅ 2 new tests covering the whitelist behavior

Not pre-prepped (intentional):
- ❌ Branch 2 / 3 / 4 fork dirs — need v23 platform log first to choose
- ❌ Op-fusion bundle (v25) — needs GPU validation first

The pre-prep buys ~10 min when v23 comes back. Use that time to read
the platform log and decide the branch.
