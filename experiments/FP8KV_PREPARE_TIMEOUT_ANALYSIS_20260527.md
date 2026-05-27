# FP8KV prepare/startup analysis, 2026-05-27

## Goal

Fix the fp8kv submission startup path without changing the inference strategy
more than necessary.

The platform-proven baseline remains the MLP-only W4A16 chunk32k package:

- `submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe`
- platform result: `acc_ori=80.31`, `final_score=22.90`
- shared server args: `minicpm_flashinfer`, `gptq_marlin`,
  `dtype=bfloat16`, chunked prefill 32768, max prefill 32768,
  mem fraction 0.70

## What failed

The package

`/autodl-fs/data/zyn/submissions/soar_fp8kv_flashinfer_prepare_cache_20260527_102811.tar.gz`

failed on platform at service startup:

```text
RuntimeError: Ninja build failed
ninja: error: '/root/autodl-tmp/zyn/fp8kv_platform_prewarm_home/.cache/flashinfer/.../batch_prefill_paged_kernel_mask_0.cu',
needed by '...batch_prefill_paged_kernel_mask_0.cuda.o',
missing and no known rule to make it
```

Root cause: the bundled `flashinfer_cache_0.5.3_120f.tar.gz` was generated on
the local prewarm machine. Its FlashInfer `build.ninja` metadata contained
absolute paths to that machine's cache/source tree. FlashInfer still invokes
ninja during `build_and_load()`, so a copied cache is not safe just because it
contains an e4m3 `.so`.

The old audit only checked that an e4m3 cached op `.so` existed. That was too
weak and allowed a non-portable cache bundle into the tarball.

## Current fix

Do not bundle locally generated FlashInfer JIT cache.

The fp8kv package now does this instead:

1. Requires a bundled cp310 `flash_attn-*.whl` by default.
2. Keeps direct GitHub `flash-attn` download opt-in only with
   `ALLOW_FLASH_ATTN_DOWNLOAD=1`.
3. Preserves existing platform `~/.cache/flashinfer` by default.
4. Does not restore a local `flashinfer_cache_0.5.3_120f.tar.gz`.
5. If platform FlashInfer cache is empty, lets FlashInfer JIT in the platform
   environment.
6. Keeps FlashInfer cache deletion diagnostic-only via
   `FORCE_FLASHINFER_CACHE_REBUILD=1`.

CUDA graph remains enabled. The local prewarm log showed graph capture itself
was 18.80s after weight load, so disabling CUDA graph is not the fix for this
failure.

## Server args

The fp8kv server args remain:

```text
--attention-backend minicpm_flashinfer
--chunked-prefill-size 32768
--max-prefill-tokens 32768
--mem-fraction-static 0.70
--max-running-requests 32
--skip-server-warmup
--dense-as-sparse
--quantization gptq_marlin
--kv-cache-dtype fp8_e4m3
--dtype bfloat16
--cuda-graph-bs 1 2 4 8 12 16 24 32
```

The startup fix is scoped to prepare/cache packaging, not quantization or
generation behavior.

## Audits

Current local no-cache candidate:

`/autodl-fs/data/zyn/submissions/soar_fp8kv_flashinfer_no_jit_cache_20260527_113508.tar.gz`

- size: `168986242` bytes, shown by `ls -lh` as `162M`
- md5: `fe9335f77135bbacf4621ec5b66955d4`

Use these before uploading a new fp8kv package:

```bash
bash scripts/audit_fp8kv_submission_tarball.sh \
  /path/to/soar_fp8kv_flashinfer_no_jit_cache_<timestamp>.tar.gz

bash scripts/compare_fp8kv_prepare_to_baseline.sh \
  --baseline-variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe \
  --fp8kv-tarball /path/to/soar_fp8kv_flashinfer_no_jit_cache_<timestamp>.tar.gz
```

The audit now rejects:

- tiny runtime snapshots
- missing cp310 `flash_attn` wheel
- cp312 `flash_attn` wheel
- bundled `flashinfer_cache_0.5.3_120f.tar.gz`
- old default GitHub download fallback
- old unconditional FlashInfer cache deletion
- bundled FlashInfer cache restore logic
- FlashInfer/ninja build metadata in the tarball

The previously submitted `102811` package is now a negative control and should
fail the new audit because it contains `flashinfer_cache_0.5.3_120f.tar.gz`.

## Residual risk

If the platform has no suitable FlashInfer cache, first launch will JIT
FlashInfer kernels in the platform environment. That is preferable to copying
machine-local JIT metadata. The previous 21-minute platform run reached server
startup and failed on stale cache paths, so the earlier indefinite
`DOWNLOADING` suspicion is no longer the main diagnosis for that package.
