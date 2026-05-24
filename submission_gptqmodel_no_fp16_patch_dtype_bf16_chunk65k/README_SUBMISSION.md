# v5j_dtype_bf16_chunk65k — push chunked-prefill to the max

> Extension of v5j_dtype_bf16_chunk32k. Adds chunked-prefill 65K (the
> 1849-W4A16-verified maximum) for the largest possible throughput on
> long-prompt workloads.

## Diff vs v5j_dtype_bf16_chunk32k

```diff
-    --chunked-prefill-size 32768 --max-prefill-tokens 32768
+    --chunked-prefill-size 65536 --max-prefill-tokens 65536
```

## Mem budget (84GB platform)

5 (model) + 0.80×84 KV (67) + 5.6 (65K buffer) = **77.6 GB ≤ 84** (6.4 GB headroom)

Identical mem budget shape as 1849 (which ran successfully on platform).

## Submission gate

Submit ONLY if chunk32k confirmed acc stays at ~82 AND chunk32k S1 < 717.
Otherwise this might:
- Not provide additional throughput gain (if 32K was already saturating)
- Hit OOM edge cases (only 6.4 GB headroom vs 9 GB for 32K)

## Expected platform

| Metric | chunk32k pred | **chunk65k pred** |
|---|---|---|
| acc_ori | ~82 | ~82 |
| S1 | ~500-550 | ~450-500 |
| final_score | ~28-32 | **~30-35** |
