# SOAR W4A16 + Marlin — v3 (chunk32k + mixed-chunk)

**Variant of:** `submission_rtn_sym_v2_chunk32k`.
**Only diff from v2:** added `--enable-mixed-chunk`.

## Why this variant

Mixed chunked prefill (SARATHI-style): within a single forward batch, the scheduler can mix prefill-chunk tokens with running decode tokens. Prefill chunk saturates GPU compute, decode requests piggyback at ~10× less cost than a decode-only batch. Net effect: less GPU idle time during the long prefill phase.

This should mostly help **S8 / Smax** (multi-batch) where there are running decodes to piggyback. **S1** with one user has nothing to piggyback, so impact there is near zero.

## Theoretical risk

Some incompatibilities are documented but I'm not aware of any that hit this config:
- ngram speculative decoding disables mixed_chunk (we don't use spec)
- DP attention has its own constraints (we don't use DP, tp=1)

Should be a safe addition on top of v2.

## Full SGLANG_SERVER_ARGS

```
--disable-radix-cache
--attention-backend minicpm_flashinfer
--chunked-prefill-size 32768
--max-prefill-tokens 32768
--enable-mixed-chunk               # <- new in v3
--skip-server-warmup
--dense-as-sparse
--quantization gptq_marlin
--dtype float16
```

## What to look for in the eval log

The first SGLang init prints active features. If `mixed_chunk` is enabled correctly there should be a line mentioning it. If silently ignored, we'd see same numbers as v2.

## Fallback

If v3 produces same numbers as v2 (mixed_chunk silently ignored), no harm done — just use v2. If it crashes or hurts S1, fall back to v2.
