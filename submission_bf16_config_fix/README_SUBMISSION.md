# SOAR BF16 Minimal-Diff Submission (v5c)

> 2026-05-23 BF16 path bisection. v3 (no bundle) and v5 (full bundle) both
> crashed in the same way (transformers model_type list dump). v5c reverts
> to 1849's exact configuration and changes ONE thing: removes
> `--quantization gptq_marlin` and replaces prepare_model with symlink.
>
> Maximum-conservative attempt. If v5c crashes, BF16-via-prepare_env path is
> fundamentally broken on this platform's stack.

## Diff from 1849 (the only verified working configuration)

| Component | 1849 | **v5c** |
|---|---|---|
| transformers pin 4.57.1 | ✅ | ✅ identical |
| Bundled flash_attn wheel | ✅ | ✅ identical (symlink to 1849's) |
| Bundled SGLang | ✅ | ✅ identical (symlink to 1849's) |
| gptqmodel install | ✅ | ✅ identical (kept for env consistency) |
| accelerate + ninja | ✅ | ✅ identical |
| Import smoke test | ✅ | ✅ identical |
| fp16 sed-patch on minicpm_backend | ✅ | ✅ identical |
| GPTQMODEL_MARLIN_USE_FP32=1 | ✅ | ✅ identical |
| prepare_model = quantize | ✅ | **❌ symlink only (1-line diff)** |
| `--quantization gptq_marlin` | ✅ | **❌ REMOVED (the only server-args change)** |
| `--dtype float16` | ✅ | ✅ identical |
| `--chunked-prefill-size 8192` | ✅ | ✅ identical (NOT 32K) |
| `--max-prefill-tokens` | (default 16384) | (default 16384) identical |
| `--enable-mixed-chunk` | ❌ | ❌ (NOT added) |

**Two changes only**:
1. `prepare_model.sh` symlinks instead of quantizing
2. `--quantization gptq_marlin` removed from server args

Everything else IS 1849.

## Expected outcome

- **acc preserved == baseline 19.13** (no quantization, no flag changes that affect throughput)
- **final_score ~19-22** (essentially matching baseline)
- **Does NOT hit rank 20** (~30 needed) — this is a diagnostic submission to verify BF16 path is reachable

## What this tests

If v5c **WORKS**:
- BF16 path is viable
- The throughput-boosting flags (chunked-prefill 32K, mixed-chunk, max-prefill-tokens 32K) caused v3/v5 crashes
- Next step: v5d = v5c + chunked-prefill 32K isolated (single variable)

If v5c **CRASHES**:
- BF16-mode prepare_env is broken on platform regardless of args
- The fp16 sed-patch may not be compatible with `--dtype float16` when there's no quantization
- Or, removing `--quantization gptq_marlin` triggers a different SGLang loader path that doesn't know minicpm_sala
- Strategic implication: **abandon BF16 submissions today**, focus 100% on 5h local experiments

## Why the fp16 sed-patch is kept (matches 1849)

1849 has sed-patch + --dtype float16. The combination loads weights as fp16
internally. For BF16-mode without quantization, --dtype float16 still casts
to fp16, so the sed-patch keeps the backend consistent.

If v5c crashes specifically due to this combination, v5d would drop sed-patch
+ change --dtype to bfloat16 to test "true BF16 mode".

## Tarball

```
soar_bf16_minimal_diff_<timestamp>.tar.gz
size ~240 MB (bundles flash_attn + sglang)
```
