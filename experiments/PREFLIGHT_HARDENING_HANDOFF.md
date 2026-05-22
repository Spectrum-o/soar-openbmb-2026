# Preflight hardening handoff (branch `quant/preflight-hardening`)

> Written 2026-05-23 by parallel-Claude while other-Claude was working
> Phase 2-5 (AWQ variant + per-task parser + mixed-precision + docs).
> This file documents what's on THIS branch only.

## What's on this branch

3 additive changes (no in-place mutation of running 5h experiment files):

| File | Purpose |
|---|---|
| `tools/verify_artifact_dequant.py` | Multi-module dequant verifier — closes the gap between "qzeros bytes look right" (`inspect_quant_artifact.py`) and "5h platform slot" |
| `tools/lint_latent_assertions.py` | Static lint for the v23 self-bug class — hardcoded shard index near a critical assertion |
| `submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py` | Added `GPTQ_DESC_ACT` + `GPTQ_STATIC_GROUPS` env knobs (default False; behavior unchanged when unset) |

Plus 3 new test files (30 new test cases; full suite goes from 112 → 142).

## Why these, not AWQ / mixed-precision / docs

Other-Claude owns:
- Phase 2: AWQ via llm-compressor (`submission_awq_llmcompressor/`)
- Phase 3: per-task accuracy parser
- Phase 4: mixed-precision (lightning-skip) variant — stretch
- Phase 5: documentation

I picked **orthogonal** work that doesn't collide:
1. **Artifact-level verifier** — every variant (GPTQ, AWQ, mixed-precision)
   benefits from a "did this artifact load mathematically intact" check.
2. **v23-shape lint** — applies to every quantize script in every variant,
   future-proofs against re-introduction.
3. **`desc_act` knob** — the historical "Marlin doesn't support
   desc_act=True" claim was WRONG (verified by reading
   `python/sglang/srt/layers/quantization/gptq.py`). With this knob in place,
   trying activation-order quant is one `GPTQ_DESC_ACT=True bash prepare_model.sh`
   away.

## How to USE the new tools

### Before submitting any quant artifact

```bash
# 1. Mathematical sanity (NEW — closes the v23 gap)
python3 tools/verify_artifact_dequant.py \
    --artifact /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized \
    --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
    --fast
# expect: [verdict] OK

# 2. Static lint (NEW — catches v23-shape bugs in source)
python3 tools/lint_latent_assertions.py
# expect: [lint] OK

# 3. Existing preflight (unchanged)
python3 tools/pack_submission.py --variant submission_gptqmodel_calib_w4a16 --check-only
```

If `verify_artifact_dequant` says `[verdict] OK` and the existing `inspect_quant_artifact`
also passes, the artifact is **mathematically intact**. If it still produces
acc=0 on the platform after that, the cause is **not in the quant math** — it's
calibration, tokenizer drift (already fixed in H4), or model-loading semantics.

### To try desc_act (next 5h slot candidate)

```bash
# Re-quant with act-order enabled. Pairs well with static_groups.
GPTQ_DESC_ACT=True GPTQ_STATIC_GROUPS=True \
  bash submission_gptqmodel_calib_w4a16/prepare_model.sh \
    --input /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
    --output /root/autodl-fs/zyn/models/submission-desc_act-quantized

# Verify before committing to a platform slot
python3 tools/verify_artifact_dequant.py \
    --artifact /root/autodl-fs/zyn/models/submission-desc_act-quantized \
    --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
    --fast

# If verifier OK → bench locally → fork a new variant dir → pack
```

**Expected effect on acc**: activation-order quant prioritizes high-activation
channels (quantize them first when error budget is fresh). The original GPTQ
paper used this; AutoGPTQ / llm-compressor's recommended recipe includes it.
On long-context models the gain is reportedly +2–10pp, but it's
**hypothesis-not-evidence** until we burn a slot to test.

**Why I didn't fork a `_v25_desc_act` variant dir**: that would require
running `prepare_model.sh` on a GPU to produce the actual artifact, then
locally eval'ing 150 samples — which conflicts with the in-flight 5h
experiment plan. The knob is plumbed and the verifier is ready;
operator-Claude (you, in the morning) can flip it when there's a free GPU.

## What I deliberately did NOT do

- **Did not change defaults.** `desc_act` defaults to False everywhere.
  In-flight A/B/D1/D2/D3/D4/E/F1/F2 experiments produce byte-identical
  artifacts to before this branch.
- **Did not edit `tools/hard_constraints_lint.py`.** Other-Claude was
  actively modifying it (uncommitted regex tweaks for AWQ mode detection).
- **Did not touch any `submission_*/` symlinked files.** Variant fork
  dirs that symlink back to v23 base still pick up the desc_act knob
  for free (since `quantize_gptqmodel_w4a16.py` IS shared via symlink).
- **Did not run any platform-touching commands.** All work CPU-only
  on the dev mirror.

## v23 self-bug retrospective (the failure this branch defends against)

The bug:
```python
# v23's broken POST-CHECK
shards = sorted(glob("*.safetensors"))
verified = False
for key in shards[0]:           # <-- only looked at shard 0!
    if is_qzero(key) and ...:
        verified = True
if not verified and total_patched > 0:
    raise FATAL                 # fires even though patch SUCCEEDED
```

Why it failed: for MLP-only artifacts, `shards[0]` contains
`embed_tokens` + `lm_head` — no quantized weights. The qzeros patch
correctly touched shards 2-3 (96/96 tensors), but POST-CHECK looked
at the wrong shard and asserted False.

The cost: one 5h platform slot consumed testing my own assertion code
rather than the H1/H4 hypotheses we cared about.

The defense (this branch):
1. `lint_latent_assertions.py` detects `shards[0]`-style indexing within
   30 lines of a FATAL/assert/sys.exit, catching the same shape in any
   future code.
2. `verify_artifact_dequant.py` operates on the LIVE artifact and
   discovers the actual `.qweight` modules dynamically — there's no
   way for it to "miss" a shard the way the v23 POST-CHECK did.
3. The unit tests in `tests/test_lint_latent_assertions.py` use
   synthetic versions of both the v23 bug shape AND the post-fix shape,
   so the regression is captured even if the original commits get
   squashed away.

## Test invariants

After this branch:
- `python3 -m unittest discover tests` → 142 passing (was 112; +30 new)
- All 4 variant `--check-only` preflights still pass:
  - submission_gptqmodel_calib_w4a16
  - submission_gptqmodel_calib_w4a16_v24_pin_transformers
  - submission_gptqmodel_calib_w4a16_v24_no_dtype_key
  - submission_gptq_v17_minconfig
- `tools/lint_latent_assertions.py` runs clean against all 9
  `submission_*/quantize_*.py` scripts (no false positives on
  the current v23+H4-fixed code)

## Merging back to `quant/w4a16`

This branch is **additive-only**. To merge:
```bash
git checkout quant/w4a16
git pull origin quant/w4a16    # in case other-Claude pushed
git merge --no-ff quant/preflight-hardening
python3 -m unittest discover tests   # sanity
```

There should be no conflicts since:
- New files in `tools/` and `tests/` don't collide
- The quantize_gptqmodel_w4a16.py edits add new env vars (additive)
- I did NOT touch `tools/hard_constraints_lint.py` (other-Claude's territory)

If `tools/hard_constraints_lint.py` has uncommitted edits from
other-Claude, those will appear as modifications on `quant/w4a16` after
they commit. Merging this branch won't touch that file.
