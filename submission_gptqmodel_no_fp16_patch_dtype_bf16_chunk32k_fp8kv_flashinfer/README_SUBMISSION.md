# fp8kv cudagraph current — server-verified FP8 KV route

> **2026-05-26 update**: this variant now mirrors the live server run from
> `origin/quant/fp8-kv-cache` commit `4af118156`. The current runnable stack
> uses `fp8_e4m3`, `minicpm_flashinfer`, bfloat16 model dtype, chunk32k prefill,
> and explicit CUDA graph batch sizes.

## What the SOAR official actually says

`soar.openbmb.cn/toolkit` "路径一：量化加速"
> "可选路径：GPTQ W4A16 + Marlin Kernel + FP8 KV Cache"
>
> "开启 KV Cache FP8： `--kv-cache-dtype fp8_e5m2`，长上下文场景收益显著。
>  注意 Lightning Attention 层使用独立线性注意力状态，优化路径不同。"

So:
- **FP8 KV IS officially recommended** for SALA, not the previously-thought "incompatible"
- The live server route currently uses **`fp8_e4m3`**. Earlier local notes that
  treated `fp8_e5m2` as mandatory are superseded by the 2026-05-26 fp8kv
  cudagraph run.
- **Lightning attention uses linear state** (not KV cache), so FP8 only affects dense attention layers — no special handling needed

## Why prior attempts failed

`gptq_marlin`'s `GPTQMarlinConfig.get_quant_method` only routes `LinearBase`
and `FusedMoE`; it returns None (via `get_linear_quant_method` fallback)
for `RadixAttention`. So `layer.k_scale` stays None, `BaseKVCacheMethod`
is never instantiated, and FlashAttention rejects fp8 query inputs without
scales.

The 2026-05-15 fp8kv failures and the 2026-05-14 investigation doc both
identified this. The investigation's "Path D" is the cleanest fix:
extend gptq_marlin's `get_quant_method` to also handle RadixAttention.

## Diff vs chunk32k_safe

1. `apply_gptq_marlin_kv_method_patch.py` (NEW): 3-line patch on
   `python/sglang/srt/layers/quantization/gptq.py`. Pattern-matched +
   idempotent. Adds `RadixAttention → BaseKVCacheMethod` routing,
   mirroring `fp8.py:185-186`.

2. `prepare_env.sh`:
   - Added GPTQMARLIN_KV_PATCH block (runs patch after install)
   - Use `--attention-backend minicpm_flashinfer`. The `minicpm_flashattn` / FA3
     path still fails on Blackwell with `no kernel image`.
   - Keep `--kv-cache-dtype fp8_e4m3` and `--cuda-graph-bs 1 2 4 8 12 16 24 32`
     in SGLANG_SERVER_ARGS. KV pool storage is FP8; the MiniCPM Path Y patch
     keeps FlashInfer query planning at bf16/model dtype.

## Why compressed_k dtype mismatch is NOT a concern

In `MiniCPMSparseBackend.forward_extend`:
```python
# Line 953: branch on sequence length
if max(forward_batch.seq_lens_cpu) >= self.dense_len:
    # sparse top-k path (NO compressed_k)
    ...
else:
    # compressed_k path (would have dtype issue)
    allocate_and_compress_keys(..., dtype=k.dtype, ...)
```

`self.dense_len` is set at line 230:
```python
self.dense_len = 0 if self.dense_as_sparse else hf_config.sparse_dense_len
```

We use `--dense-as-sparse`, so `dense_len = 0` → all sequences hit the
sparse top-k path → `allocate_and_compress_keys` is never called →
no compressed_k dtype mismatch.

## Mem budget (84GB platform)

```
W4A16 model:     5 GB
KV pool static:  0.70 × 84 × 0.5 (fp8 is half of bf16) = 29.4 GB
chunked-prefill: 3 GB (32K)
                 -----
                 37.4 GB ≤ 84 GB ✓ (46.6 GB headroom 🚀)
```

The 46.6 GB headroom means:
- No activation OOM risk (cure for chunk65k crash class)
- Could support 2× concurrent requests for Smax
- Room to stack chunked-prefill 65K later

## Expected platform (per official guidance: "长上下文场景收益显著")

| Metric | v5j_dtype_bf16 (8K bf16 KV) | predicted fp8kv |
|---|---|---|
| acc_ori | 82.18 | ~80-82 (some quant noise from uncalibrated KV scales) |
| S1 | 717 | **~400-450** (smaller KV cache → faster decode reads) |
| S8 | 1067 | **~700-800** |
| Smax | 2343 | **~1500-1800** (concurrency boost from freed mem) |
| final_score | 23.18 | **~28-38** |

## REQUIRED smoke before submission

```bash
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv \
    --num-samples 30 --force-requant 2>&1 | tee /tmp/fp8kv_v2_smoke.log

# Check three signals:
# 1. Server starts (no FlashAttention rejection, no sparse-backend crash)
grep -iE "Capture cuda graph|AssertionError|RuntimeError" /tmp/fp8kv_v2_smoke.log | head -5

# 2. acc on 30 samples — should be within 2pp of v5j_dtype_bf16's 82.18
grep -iE "acc_ori|acc =" /tmp/fp8kv_v2_smoke.log | tail -5

# 3. Patch actually applied (line in startup log):
grep -iE "applying GPTQMarlin KV cache method patch" /tmp/fp8kv_v2_smoke.log | head -3
```

## Decision after smoke

| Smoke result | Action |
|---|---|
| acc_ori ≥ 80 | **Submit** — official-aligned config, expected +5-15 final_score |
| acc_ori 75-79 | Don't submit. Try Path C (calibrated scales via llm-compressor). |
| acc_ori < 75 | Don't submit. uncalibrated scale=1.0 too crude for SALA. |
| RuntimeError: dtype | Investigate — perhaps fa_impl_ver=3 still hits a flashinfer FA2 path somewhere |
| FlashAttention only support fp16/bf16 | Path D didn't activate; check patch tool ran |

## Why this is THE recommended next experiment after chunk32k_safe

Per official guidance:
- W4A16 + Marlin + FP8 KV is the **canonical stack** SOAR recommends
- "长上下文场景收益显著" — the eval set is long-context (NIAH, QA, CWE)
- If it works, this beats every other variant we've prepared

After smoke clears, this should be the next platform submission.

## Latest Server Evidence

Latest pulled fp8kv branch:

- `origin/quant/fp8-kv-cache @ 4af118156`
- server run snapshot:
  `zyn_eval_runs/fp8kv_cg_e4m3_20260526_180812_snapshot_0060`
- snapshot result: 60/60 completed, `ori_accuracy=81.83`,
  `overall_accuracy=100`, `tps=147.12`

Runtime settings mirrored here:

```bash
--attention-backend minicpm_flashinfer \
--quantization gptq_marlin \
--kv-cache-dtype fp8_e4m3 \
--dtype bfloat16 \
--chunked-prefill-size 32768 \
--max-prefill-tokens 32768 \
--mem-fraction-static 0.70 \
--max-running-requests 32 \
--cuda-graph-bs 1 2 4 8 12 16 24 32
```
