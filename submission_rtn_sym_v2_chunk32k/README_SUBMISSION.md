# SOAR W4A16 + Marlin — v2 (chunked-prefill 32K)

**Variant of:** `submission_rtn_sym_w4a16` (the v1 working baseline submitted 2026-05-19).
**Only diff from v1:** `--chunked-prefill-size` bumped from 8192 to 32768, with the required `--max-prefill-tokens 32768` companion.

## Why this variant

A local microbench on 2026-05-14 (RTX PRO 6000 Blackwell, MiniCPM-SALA, 64 prompts × 4096-in / 512-out) measured:

| Metric | chunk=8192 | chunk=32768 | Delta |
|---|---|---|---|
| Output throughput | 267.67 tok/s | 425.58 tok/s | **+59%** |
| Mean TTFT | 34049 ms | 13545 ms | **-60%** |
| Bench duration | 59.85 s | 37.64 s | **-37%** |

So this is the most "guaranteed" upgrade — pure scheduling change, no kernel touched.

## Gotcha that this variant handles

sglang has an undocumented `--max-prefill-tokens` default of **16384**. Setting `--chunked-prefill-size 32768` alone is silently capped at 16K by this limit. The two flags must be raised together.

## Full SGLANG_SERVER_ARGS

```
--disable-radix-cache
--attention-backend minicpm_flashinfer
--chunked-prefill-size 32768       # <- v1 was 8192
--max-prefill-tokens 32768         # <- new
--skip-server-warmup
--dense-as-sparse
--quantization gptq_marlin
--dtype float16
```

(CUDA graph stays on by default. Do NOT add `--disable-cuda-graph`.)

## What's NOT changed from v1

- Same `quantize_gptq_rtn_sym.py` (symmetric RTN, sym=True for Marlin).
- Same `prepare_model.sh`.
- Same SGLang source tree.

## Expected result vs v1

If the local microbench generalizes to the SOAR eval workload, v2 should beat v1 by **30-60%** in benchmark duration on both S8 and Smax. Correctness should be identical (no quantization change).

## Fallback

If v2 OOMs at high concurrency (32K chunks × multiple batches eat KV pool fast), drop back to v1 or try `chunked-prefill-size=16384` + `max-prefill-tokens=16384`.
