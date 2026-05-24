# v5j_dtype_bf16_chunk32k_fp8kv — FP8 KV cache via Path B patch

> **HIGH-RISK, HIGH-REWARD experiment.** Requires GPU smoke before any
> platform submission. Estimated +5-15 final_score IF it works, but
> non-trivial chance acc drops below 80% gate.

## What this variant tests

Implements **Path B** from `notes/fp8-kv-cache-investigation` (2026-05-14):
patch `minicpm_backend.py` so when `layer.k_scale is None` (which happens
under `--quantization gptq_marlin` since gptq_marlin's quant_config only
handles LinearBase, not RadixAttention), the FA3 kernel still gets a
default `k_descale = 1.0` instead of choking.

This unlocks `--kv-cache-dtype fp8_e4m3` on the W4A16+bf16 path verified
by v5j_dtype_bf16 (acc_ori 82.18).

## Diff vs chunk32k_safe

1. `apply_fp8_kv_fallback_patch.py` (NEW): patches both `forward_extend`
   and `forward_decode` sites in minicpm_backend.py.

2. `prepare_env.sh`:
   - Added FP8 KV PATCH block (runs patch tool after install)
   - `--attention-backend minicpm_flashinfer` → `minicpm_flashattn` (FA3 supports fp8)
   - Added `--kv-cache-dtype fp8_e4m3`

## Mem budget (84GB platform)

```
W4A16 model:    5 GB
KV pool static: 0.70 × 84 × 0.5 (fp8 is half of bf16) = 29.4 GB
chunked-prefill buffer (32K): 3 GB
                              -----
                              37.4 GB ≤ 84 GB ✓ (46.6 GB headroom 🚀)
```

The 46.6 GB headroom means: no activation OOM risk, room for more
concurrent requests, possibly room to push chunked-prefill higher.

## Risks (read carefully)

1. **Default scale=1.0 is uncalibrated.** SALA's scale_emb=12 makes
   activations larger than typical. KV magnitudes may overflow fp8_e4m3
   range or lose too much precision. Expected acc drop 1-3pp; could be 5+.

2. **minicpm_flashattn may not fully support SALA sparse attention.**
   We've been using minicpm_flashinfer exclusively. Need verification
   that sparse_get_topk + InfLLMv2 codepaths work with flashattn backend.

3. **Patch may break import.** Python file modification at install time.
   Patch is idempotent + exits 1 on failure, so caught at startup.

## REQUIRED GPU smoke before submission

```bash
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv \
    --num-samples 30 --force-requant 2>&1 | tee /tmp/fp8kv_smoke.log

# Check:
# 1. Server start (no crash):
grep -iE "Capture cuda graph|AssertionError|RuntimeError" /tmp/fp8kv_smoke.log

# 2. acc on 30 samples:
grep -iE "acc_ori|acc =" /tmp/fp8kv_smoke.log | tail -5

# 3. FP8 KV actually engaged:
grep -i "fp8_e4m3\|kv_cache_dtype" /tmp/fp8kv_smoke.log | head -5
```

## Decision after smoke

| Smoke result | Action |
|---|---|
| acc_ori ≥ 80 | **Submit** — final_score predicted 28-38 |
| acc_ori 75-79 | Don't submit. Try Path C (calibrated scales). |
| acc_ori < 75 | Don't submit. FP8 KV needs calibration on SALA. |
| Server crash | Don't submit. Investigate flashattn compatibility. |

## Expected platform (if smoke clears)

| Metric | v5j_dtype_bf16 | **fp8kv (32K)** predicted |
|---|---|---|
| acc_ori | 82.18 | ~80-82 |
| S1 | 717 | **~400-450** |
| S8 | 1067 | **~700-800** |
| Smax | 2343 | **~1700-1900** (2× concurrency possible) |
| final_score | 23.18 | **~28-38** |
