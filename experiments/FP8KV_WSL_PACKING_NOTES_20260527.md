# FP8KV WSL packing notes, 2026-05-27

This repo intentionally does not commit the large `flash_attn` wheel. GitHub's
normal single-file limit is 100MB, while the cp310 SM120 wheel is about 152MB.

The fp8kv prepare-cache package should still include the wheel in the final
tarball to avoid platform-side GitHub download latency. Use the local WSL wheel
only at packing time:

```bash
cd /path/to/soar-openbmb-2026

bash scripts/pack_fp8kv_prepare_cache.sh \
  --wheel /path/to/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl \
  --out-dir ./dist
```

The script:

- temporarily links the local `flash_attn-*.whl` into
  `submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer/`
- runs `bash -n` and `tools/pack_submission.py --check-only`
- packs the submission tarball
- verifies the tarball contains the wheel and
  `flashinfer_cache_0.5.3_120f.tar.gz`
- verifies `prepare_env.sh` still uses `fp8_e4m3`,
  `--max-running-requests 32`, and CUDA graph batches
  `1 2 4 8 12 16 24 32`
- verifies the bundled FlashInfer cache contains an e4m3 cached op `.so`
- removes the temporary wheel link before exit

If the local wheel has a different filename, it must still start with
`flash_attn-`, end with `.whl`, and include `-cp310-cp310-` in the filename.
The platform Python is 3.10; do not pack a cp312 wheel.

Before uploading, audit the exact tarball:

```bash
bash scripts/audit_fp8kv_submission_tarball.sh \
  ./dist/soar_fp8kv_flashinfer_prepare_cache_<timestamp>.tar.gz
```

This catches the known bad cases: tiny runtime snapshot instead of a full SOAR
package, missing wheel, wrong Python ABI wheel, missing FlashInfer cache,
default network download fallback, default cache deletion, and fp8kv/cudagraph
argument drift.

Then compare the prepare path against the platform-proven `chunk32k_safe`
variant:

```bash
bash scripts/compare_fp8kv_prepare_to_baseline.sh \
  --baseline-variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe \
  --fp8kv-tarball ./dist/soar_fp8kv_flashinfer_prepare_cache_<timestamp>.tar.gz
```

Negative-control check:

```bash
scripts/audit_fp8kv_submission_tarball.sh \
  /root/autodl-tmp/zyn/sglang_check_branch/zyn_submission_packages/fp8kv_cudagraph_current/fp8kv_cudagraph_current.tar.gz
```

Expected result:

```text
FAIL: tarball is too small (190076 bytes); likely not a full SOAR package
```

That 188K archive is only a runtime snapshot/check artifact. Do not upload it
to the platform. The upload candidate should be the 166M full package with md5
`f5513a55a6c43b7ba8f11dc68871ed46`.

The final tarball is what should be uploaded to the SOAR platform.
