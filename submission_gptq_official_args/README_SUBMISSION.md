# SOAR GPTQ via llm-compressor + Official Serving Args (P3 ablation)

> Built 2026-05-23 14:40 as **Experiment 3** for tonight's W4A16 migration validation.
> Mirrors AWQ official-args variant exactly, except quantization algorithm:
> GPTQ (matches W4A16_README.md's explicit recommendation) instead of AWQ.

## Purpose

`W4A16_README.md` says verbatim: "**llm-compressor + GPTQ**". Our AWQ variant
deviates from this — we picked AWQ for SALA's `scale_emb=12` outlier channels
based on a research memo. P3 is the **clean A/B vs P2 (AWQ)**:

- P2: AWQ via llm-compressor + compressed-tensors + 65K serving args
- P3: GPTQ via llm-compressor + compressed-tensors + 65K serving args
- Same calib data (perf_public_set, 256 samples, 8K max len, MLP-only)
- Same H4 / auto_map fixes
- Same SGLANG_SERVER_ARGS

Whichever wins acc-wise is the algorithm we ship in tomorrow's submission.

## Recipe (mirrors official `quantize_to_w4a16.py`)

```python
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",       # symmetric, official default
    ignore=["lm_head", ...],
    dampening_frac=0.01,  # official default
)
```

Compared to AWQ variant's recipe:
```python
recipe = [
    AWQModifier(ignore=...),
    QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=...),
]
```

## Diff vs `submission_awq_official_args`

**Files**:
- `quantize_llmcompressor_gptq.py` (new, mirrors AWQ script with GPTQModifier)
- `prepare_model.sh` (new, calls gptq script with `--dampening-frac 0.01`)
- `prepare_env.sh` (same as AWQ except smoke test imports `GPTQModifier` not `AWQModifier`)

**Env vars** (set when running locally):
- `MLP_ONLY=1` (default) — same safety as AWQ
- `GPTQ_SCHEME=W4A16` (default, symmetric — matches official). To compare with
  AWQ_ASYM directly, set `GPTQ_SCHEME=W4A16_ASYM`.
- `DAMPENING_FRAC=0.01` (default). Raise to 0.1 if numerical issues.

## When to submit this variant

If P3 local acc_ori > P2 local acc_ori by ≥ 2pp, GPTQ is the better algorithm
for SALA → tomorrow's submission should be this variant. Otherwise stick with
AWQ.

## Why dampening_frac=0.01 not 0.1

Official `quantize_to_w4a16.py` uses 0.01 (its argparse default). The 5h
experiment plan tested D2 (dampening_frac=0.1) on the OLD gptqmodel path
and found +6.89pp. But:
1. That was gptqmodel (different implementation), not llm-compressor
2. That was gptq_marlin path, not compressed-tensors
3. 0.01 is the official W4A16_README recipe
4. We can ablate 0.1 in a future experiment if 0.01 underperforms

Start with the official recipe; deviate only with evidence.
