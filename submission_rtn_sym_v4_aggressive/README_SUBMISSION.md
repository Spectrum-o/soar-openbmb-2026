# SOAR W4A16 + Marlin — v4 (aggressive: chunk32k + mixed-chunk + mem-fraction + piecewise CUDA graph)

**Variant of:** `submission_rtn_sym_v3_mixedchunk`.
**Diffs from v3:** added `--mem-fraction-static 0.92` and `--enable-piecewise-cuda-graph`.

## Why this variant

Two layered optimizations on top of v3:

### 1. `--mem-fraction-static 0.92`
- Default is roughly 0.88. With a 96GB card, there is headroom to push KV pool larger.
- Bigger KV pool → higher concurrency ceiling, especially for **Smax**.
- **Watch for**: CUDA graph buffer / activation OOM if pushed too high.

### 2. `--enable-piecewise-cuda-graph`
- SALA's hybrid attention (25% sparse + 75% linear) likely can't capture as one monolithic CUDA graph.
- Piecewise CUDA graph lets each segment of the forward pass be captured independently.
- Should reduce decode kernel-launch overhead for the parts that DO graph-capture cleanly.
- **Watch for**: if some segment fails to capture, may silently fall back to eager — verify in logs.

## Risk profile

This is the **highest-risk** variant in the v1/v2/v3/v4 sequence. Two new flags, and one of them (piecewise-cuda-graph) hasn't been verified on the SALA hybrid path. If it crashes at startup, immediately fall back to v3.

## Full SGLANG_SERVER_ARGS

```
--disable-radix-cache
--attention-backend minicpm_flashinfer
--chunked-prefill-size 32768
--max-prefill-tokens 32768
--enable-mixed-chunk
--mem-fraction-static 0.92             # <- new
--enable-piecewise-cuda-graph          # <- new
--skip-server-warmup
--dense-as-sparse
--quantization gptq_marlin
--dtype float16
```

## Expected behavior

Best case: 5-15% additional speedup over v3 on Smax (from concurrency headroom) plus some decode latency reduction (from piecewise graph).

Realistic case: similar to v3 (one or both flags silently no-op).

Worst case: startup crash. Fall back to v3.

## What NOT to add

Tempting flags that are NOT included here (known incompatible or unverified):
- `--kv-cache-dtype fp8_e4m3` — incompatible with MiniCPM sparse backend (verified by past failure).
- `--enable-two-batch-overlap` — additional CPU/GPU interleave; not added because it's another unverified flag and v4 already stacks two new things.
- `--speculative-algorithm ngram` — disables mixed_chunk (mutually exclusive).
