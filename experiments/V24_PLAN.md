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
│   │             → scripts/prep_v24_perf.sh
│   │
│   ├── v25-perf-fusion: v24-perf + op-fusion (~10-15% additional
│   │                     decode speedup; requires prior GPU validation)
│   │                     → scripts/prep_v24_perf.sh --with-fusion
│   │                     → REQUIRES: MINICPM_FUSION_VERIFY=1 run
│   │                       passing first (norm_diff < 1e-3)
│   │
│   └── parallel calibration push for accuracy (R5 ceiling):
│       Exp E: multi-adaptive calib + 300 samples (PLAN.md)
│       Exp F: --group-size 64 / --desc-act (PLAN.md)
│
├── acc == 0, log shows [qzeros-fix] FATAL
│   └── New gptqmodel layout caught by hardened fix. Read FATAL
│       message + sample hex values from log. Extend
│       fix_qzeros_for_marlin in submission_*/quantize_gptqmodel_w4a16.py
│       to handle the new pattern. Repack as v24-qzeros-extended.
│
├── acc == 0, log shows [qzeros-fix] OK + bench timing ~ v22
│   └── H1 + H4 were not the platform=0 cause. Try ONE single-variable
│       change per submission:
│       a. v24-no-dtype-key: remove `dtype: "float16"` from
│          write_sglang_compatible_quant_config (just keep
│          `torch_dtype`). RTN doesn't set the new HF 5.x key.
│       b. v24-pin-transformers: bundle a known-good transformers
│          wheel + pin install. Forces both AutoDL and platform onto
│          the same version.
│       c. v24-bits8: --bits 8 (W8A16). If it works at 8-bit and
│          fails at 4-bit, the platform's 4-bit kernel is the issue.
│
└── acc == 0, sglang crashed before bench
    └── Read tracesback from server log. Usually one of:
        - dtype mismatch (sed-patch missed a file)
        - KeyError on a quantized module name (dynamic skip rules
          don't match what gptqmodel produced)
        - flash_attn missing (wheel install failed)
        These are all addressable from V23_PLATFORM_LOG_CHECKLIST.md
        Block 6/7.
```

## Concrete commands per branch

### Branch 1: acc >= 30 → v24-perf

```bash
# Apply chunked-prefill 65K back to submission tarballs
bash scripts/prep_v24_perf.sh

# Verify the change took
git diff submission_gptqmodel_calib_w4a16/prepare_env.sh | grep chunked-prefill
# expected: -8192 / +65536 + max-prefill-tokens + mem-fraction-static

# Re-quant + eval locally to confirm no regression (~65 min)
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --force-requant \
    --eval-data submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \
    --num-samples 150

# Pack v24
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16 \
    --suffix _v24_perf \
    --output-dir .

# Submit
```

Expected platform: corrected_acc same as v23, but `benchmark_duration` drops 50%+ in all of S1/S8/Smax. Final_score should rise materially.

### Branch 2: acc == 0 + [qzeros-fix] FATAL

```bash
# Read the FATAL line and the sample hex values
grep -A2 "qzeros-fix.*FATAL\|qzeros-fix.*sample" /path/to/platform_v23.log

# Example: if FATAL says "found 1 .bin files" → gptqmodel switched to
# torch format. Fix: extend the function to handle .bin in addition
# to .safetensors. Patch submission_*/quantize_gptqmodel_w4a16.py.

# Example: if sample shows unique=[X] hex=['0xYY...'] where YY != 77
# and != 88 → new qzeros encoding. Need to figure out the mapping.
# E.g., gptqmodel might switch to asymmetric quant with qzeros=8 OR 9
# OR a different packed representation. Inspect the artifact byte-level
# locally via tools/inspect_quant_artifact.py to understand.
```

Iterate quickly: small source change → pack → submit.

### Branch 3: acc == 0 + qzeros OK

H1 + H4 weren't sufficient. **Most likely candidates in priority order**:

#### v24-no-dtype-key (try first; cheapest)

```bash
# Edit submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py
# In write_sglang_compatible_quant_config function, remove this line:
#   config["dtype"] = "float16"
# Keep:
#   config["torch_dtype"] = "float16"
```

Rationale: RTN-scalefix's quantize_gptq_rtn_sym.py only writes `torch_dtype`. Our script writes BOTH `torch_dtype` and `dtype`. The new HF 5.x `dtype` key may be parsed wrong by older transformers, overriding the right value.

```bash
# Then pack as v24-no-dtype-key
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16 \
    --suffix _v24_no_dtype_key \
    --output-dir .
```

#### v24-pin-transformers

If we suspect platform transformers version is the issue, bundle a wheel:

```bash
# Download a known-good transformers wheel into the variant dir
cd submission_gptqmodel_calib_w4a16/
pip download transformers==4.46.0 --dest . --no-deps
# Add to prepare_env.sh: uv pip install --force-reinstall --no-deps ./transformers-4.46.0-*.whl
# Pack
```

Note: this is heavy (transformers ~30MB) and might conflict with platform's pre-installed transformers. Use ONLY if other branches exhausted.

#### v24-bits8

```bash
# In submission_gptqmodel_calib_w4a16/prepare_model.sh, set:
# QUANT_ARGS+=(--bits 8)
# Then pack as v24_bits8
```

W8A16 should pass correctness easily but at perf cost. Useful as a sanity-check submission.

### Branch 4: acc == 0 + sglang crashed early

This is the easiest to debug. The crash traceback names the bug.

- `RuntimeError: query and key must have the same dtype` → sed-patch didn't apply somewhere. Check Block 6 of V23_PLATFORM_LOG_CHECKLIST.md.
- `KeyError: model.layers.X.self_attn.Y.weight` → quantize_config.json's `dynamic` skip patterns don't match what gptqmodel emitted. Need to update the regex in `make_quant_config` to match the platform's gptqmodel module names.
- `ImportError: flash_attn` → wheel install failed; bundle the wheel or improve fallback chain.

## Op-fusion: a v25 deferral

`perf/op-fusion` is pure-torch-CPU equivalence-verified but not GPU-validated. To bring it in:

1. Run GPU validation locally (one prompt with `MINICPM_FUSION_VERIFY=1`); confirm per-layer norm_diff < 1e-3.
2. Once validated, cherry-pick:
   ```bash
   git cherry-pick 79b0c20f4    # the actual fusion change
   # (skip the test commit 87c22ec88 — already on quant/w4a16 via prior infra)
   ```
3. Re-pack v25 with the cherry-picked sglang/python/sglang/srt/models/minicpm.py
4. Verify pack_submission --check-only still passes (op-fusion doesn't change any canary)

Document this in V25_PLAN.md when the time comes.

## What pre-prep saves vs what it doesn't

Pre-prepped tonight:
- ✅ `scripts/prep_v24_perf.sh` — chunked-prefill 65K restoration ready
- ✅ This decision tree

Not pre-prepped (intentional):
- ❌ Actual v24-perf tarball — needs `prep_v24_perf.sh` invocation + pack (5 min)
- ❌ Branch 2/3/4 source variants — need the v23 platform log first to choose
- ❌ Op-fusion bundle — needs GPU validation first

The pre-prep buys ~10 min when v23 comes back. Use that time to read the platform log and decide the branch.
