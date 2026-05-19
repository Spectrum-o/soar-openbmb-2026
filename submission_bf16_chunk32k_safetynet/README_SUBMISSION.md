# SOAR BF16 Safety-Net Submission

**Use this when you can't afford a 0-score result and have an untested experimental submission you don't want to gamble.**

## What it does

- NO model modification (just symlinks the original BF16 weights)
- NO quantization
- NO fp16 patch
- Only scheduling-level optimizations applied via SGLang args:
  - `--chunked-prefill-size 32768` (vs baseline 8192)
  - `--max-prefill-tokens 32768` (must match)
  - `--enable-mixed-chunk` (SARATHI-style piggyback)

## Why this is a safe bet

- **Correctness is mathematically identical to baseline.** No weights changed.
- **Local microbench on 2026-05-14 predicted +59% throughput, -60% TTFT** vs baseline (`chunked_prefill_size` 8K → 32K on SALA).
- All scheduling flags are SGLang built-ins; well-tested independently.
- No new dependencies (no gptqmodel install / kernel compile).
- Quick prepare phase (~5 min total: install sglang + symlink model).

## Expected score

- Baseline (no quant, no extra flags): final_score = 19.13
- This (BF16 + chunk32k + mixed_chunk): expected **20-26 final_score**

Modest improvement, but **guaranteed to not be 0**.

## When to use this

- Last submission of the day; can't afford waste
- Experimental quantization submissions failed correctness gate
- Want to lock in a baseline-beating result while iterating on the riskier path

## Tarball

`soar_bf16_chunk32k_safetynet_submission_20260519.tar.gz` (~3 MB)
