# FP8KV prepare timeout analysis, 2026-05-27

## Goal

Fix the fp8kv submission that stayed in the platform
`DOWNLOADING` / preparation stage, without changing the inference strategy more
than necessary.

The comparison baseline is the platform-successful MLP-only W4A16 package:

`soar_w4a16_bf16_chunk32k_safe_20260524_163047.tar.gz`

Platform result:

- wall time: about 2h22m
- `acc_ori=80.31`
- `final_score=22.90`
- server args: `minicpm_flashinfer`, `gptq_marlin`, `dtype=bfloat16`,
  chunked prefill 32768, mem fraction 0.70

## Prepare-stage diff against the successful package

Both packages already do the same heavyweight prepare work:

- Install the bundled editable `sglang/python` when present.
- Pin/install `gptqmodel==7.0.0`.
- Pin/install `transformers==4.57.1`, compatible `huggingface-hub` and
  `tokenizers`.
- Install `accelerate` and `ninja`.
- Use a bundled `flash_attn-2.8.3+cu128torch2.9` wheel when available, falling
  back to a direct wheel URL only if the bundled wheel is missing.

Those shared steps are not the new timeout suspect when the package is actually
packed the same way. A critical caveat: the git variant directory does not store
the large `flash_attn-*.whl`; it must be injected at tarball-build time. If a
platform package is built directly from the bare variant without that wheel,
`prepare_env.sh` can fall back to a GitHub download. That exact failure class
has already caused platform `PREPARING` / `DOWNLOADING` stalls, so the fp8kv
package is now offline-by-default.

The fp8kv FlashInfer package adds:

- `apply_gptq_marlin_kv_method_patch.py` on `gptq.py`.
- `--kv-cache-dtype fp8_e4m3`.
- `--max-running-requests 32`.
- `--cuda-graph-bs 1 2 4 8 12 16 24 32`.
- `ENABLE_SM120=1`.
- `FLASHINFER_CUDA_ARCH_LIST=12.0f`.
- Previously: unconditional `rm -rf ~/.cache/flashinfer`.

The last item is the most preparation/startup-relevant delta. The deletion is
fast by itself, but it forces cold FlashInfer JIT during server startup. The
platform UI can still show this as `DOWNLOADING` / preparation because that
stage covers contestant setup and service readiness.

## Change made

Updated:

`submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer/prepare_env.sh`

Behavior is now:

1. Require `flash_attn` to be either already importable or provided as a bundled
   `flash_attn-*.whl`. Direct GitHub download is disabled unless
   `ALLOW_FLASH_ATTN_DOWNLOAD=1` is set explicitly.
2. Preserve an existing `~/.cache/flashinfer/<version>/120f` cache.
3. If platform cache is empty and the package includes
   `flashinfer_cache_0.5.3_120f.tar.gz`, restore it.
4. Fall back to normal JIT only when no reusable cache exists.
5. Only clear FlashInfer cache when explicitly requested with
   `FORCE_FLASHINFER_CACHE_REBUILD=1`.

Added bundled cache:

`submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer/flashinfer_cache_0.5.3_120f.tar.gz`

It is generated from a clean local HOME with the platform-target server args:
`minicpm_flashinfer`, `gptq_marlin`, `kv_cache_dtype=fp8_e4m3`,
`dtype=bfloat16`, `max_running_requests=32`, and CUDA graph batches
`1 2 4 8 12 16 24 32`.

Local prewarm result:

- server ready after weight load plus CUDA graph capture
- weight load: about 4m40s in the current local environment
- CUDA graph capture: 18.80s
- request smoke: `GET /v1/models` returned 200, one short chat completion
  returned 200
- generated cache: `~/.cache/flashinfer/0.5.3/120f`, about 17MB uncompressed
- key cached op:
  `batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_...so`

The compressed bundle is small enough to include in the submission package.

## What can be packed locally

Already packed in the successful/fp8kv family:

- Patched `sglang/python` source through the variant's `sglang` link.
- GPTQ quantization scripts and calibration JSONL.

Packed now:

- FlashInfer JIT cache for `flashinfer==0.5.3`, arch key `120f`.

Packed at tarball-build time, not committed to git:

- `flash-attn` cp310 prebuilt wheel. It is larger than GitHub's normal 100MB
  single-file limit, so WSL should pass the local wheel path to
  `scripts/pack_fp8kv_prepare_cache.sh`. The script temporarily links it into
  the variant, packs the tarball with symlinks dereferenced, then removes the
  temporary link.

Possible but not selected as this first fix:

- Full Python wheelhouse for `gptqmodel`, `transformers`, `tokenizers`,
  `huggingface-hub`, `accelerate`, `ninja`.

Reason: these dependencies were already part of the successful package's
prepare path. The immediate non-negotiable offline artifact is the flash-attn
wheel, because the bare repository cannot commit it and a missing wheel routes
prepare back to GitHub unless guarded.

## Residual risk

The bundled `120f` cache currently contains the FlashInfer bf16/e4m3/bf16
batch-prefill module reached by the local smoke run. If the platform run asks
FlashInfer for another module not present in the bundle, FlashInfer will still
JIT that missing module. This is intentional: the cache restore is a
startup-risk reduction, not a hard dependency. If the cache is incompatible,
removing it or setting `RESTORE_FLASHINFER_JIT_CACHE=0` falls back to normal
JIT.

CUDA graph stays enabled. The local fp8kv logs showed graph capture itself was
seconds-scale once the server had the needed kernels, so disabling CUDA graph
is not the first fix for a prepare-stage timeout.

The cache hypothesis is not treated as proven. Local prewarm made the cache
small and startup finite. The robust fix for the last platform slot is therefore
two-part: remove default network download from prepare, and avoid forced cold
FlashInfer JIT when a reusable cache exists.
