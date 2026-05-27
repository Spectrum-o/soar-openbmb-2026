# FP8KV residual data index, 2026-05-27

This file records the useful residual data from the fp8kv startup investigation
without committing large wheels, submission tarballs, model files, or raw logs.

## GitHub state

Primary repo:

`https://github.com/Spectrum-o/soar-openbmb-2026`

Working branch:

`fix/fp8kv-prepare-cache-preserve`

Important committed artifacts:

- `experiments/FP8KV_PREPARE_TIMEOUT_ANALYSIS_20260527.md`
- `experiments/FP8KV_WSL_PACKING_NOTES_20260527.md`
- `experiments/FP8KV_RESIDUAL_DATA_INDEX_20260527.md`
- `scripts/pack_fp8kv_prepare_cache.sh`
- `scripts/audit_fp8kv_submission_tarball.sh`
- `scripts/compare_fp8kv_prepare_to_baseline.sh`

## Failed package

Failed platform package:

`/autodl-fs/data/zyn/submissions/soar_fp8kv_flashinfer_prepare_cache_20260527_102811.tar.gz`

Metadata:

- size: `173614932` bytes, shown by `ls -lh` as `166M`
- md5: `f5513a55a6c43b7ba8f11dc68871ed46`

Failure:

- platform reached SGLang startup, then failed during CUDA graph capture
- FlashInfer/ninja tried to read generated `.cu` files from a local prewarm
  path embedded in `build.ninja`
- root cause was bundled local `flashinfer_cache_0.5.3_120f.tar.gz`

Do not upload this package again.

## Current package policy

Current local no-cache candidate:

`/autodl-fs/data/zyn/submissions/soar_fp8kv_flashinfer_no_jit_cache_20260527_113508.tar.gz`

- size: `168986242` bytes, shown by `ls -lh` as `162M`
- md5: `fe9335f77135bbacf4621ec5b66955d4`

The current fp8kv package policy is:

- include bundled cp310 `flash_attn-*.whl`
- do not include `flashinfer_cache_0.5.3_120f.tar.gz`
- do not clear platform `~/.cache/flashinfer` by default
- do not restore locally generated FlashInfer JIT cache
- let FlashInfer JIT in the platform environment when platform cache is empty

Shared server args kept from the platform-proven chunk32k baseline:

- `--chunked-prefill-size 32768`
- `--max-prefill-tokens 32768`
- `--mem-fraction-static 0.70`
- `--quantization gptq_marlin`
- `--dtype bfloat16`

fp8kv-only server args:

- `--kv-cache-dtype fp8_e4m3`
- `--cuda-graph-bs 1 2 4 8 12 16 24 32`
- `--max-running-requests 32`

## Verification commands

Run these before any future upload of the same package family:

```bash
cd /root/autodl-tmp/zyn/sglang_w4a16

bash scripts/audit_fp8kv_submission_tarball.sh \
  /path/to/soar_fp8kv_flashinfer_no_jit_cache_<timestamp>.tar.gz

bash scripts/compare_fp8kv_prepare_to_baseline.sh \
  --baseline-variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe \
  --fp8kv-tarball /path/to/soar_fp8kv_flashinfer_no_jit_cache_<timestamp>.tar.gz
```

The old `102811` package is a negative control for the new audit and should
fail because it contains the bundled FlashInfer cache.

## Local smoke and shard notes

Useful local logs and small eval outputs are still on the server, but are not
worth committing as raw files.

Prewarm log:

`/root/autodl-tmp/zyn/fp8kv_platform_prewarm_logs/server.log`

Key facts from that log:

- server args used `minicpm_flashinfer`, `gptq_marlin`, `kv_cache_dtype=fp8_e4m3`
- weight load took about 4m40s in the local server environment
- CUDA graph capture for batches `1 2 4 8 12 16 24 32` took 18.80s
- `GET /v1/models` returned 200
- one short chat completion returned 200

Small local eval summaries under `/root/autodl-tmp/zyn/eval_runs`:

- `fp8kv_cg_e4m3_20260526_180812`: 150 completed, `ori_accuracy=83.13`,
  `overall_accuracy=100`, duration 2743.25s. This run had obvious long-tail
  shards.
- `fp8kv_cg_e5m2_first60_20260527`: 30 completed, `ori_accuracy=73.89`,
  `overall_accuracy=92.36`, duration 162.53s.

These shards are useful as debugging context only; they are not
platform-equivalent proof.

## Server directories

Residual directories under `/root/autodl-tmp/zyn`:

- `sglang_w4a16`: current GitHub worktree for fp8kv package scripts and docs.
- `sglang_check_branch`: clean comparison worktree for the timed-out check
  branch.
- `sglang_runtime`: runtime worktree, about 11G, branch
  `quant/fp8-kv-cache`; do not delete automatically without checking branch
  state.
- `sglang_full_w4a16`: full W4A16 worktree, about 12G, detached at
  `github-submit/exp/full-w4a16`; do not delete automatically.
- `uv_cache`: about 17G local dependency cache. Rebuildable if disk space
  matters.
- `fp8kv_platform_prewarm_home`, `fp8kv_platform_prewarm_logs`, `eval_runs`,
  and `logs`: small residual run records.

## Do not upload or commit

- Do not upload the failed `102811` package again.
- Do not upload the 188K `fp8kv_cudagraph_current.tar.gz` runtime snapshot.
- Do not commit `flash_attn-*.whl`.
- Do not commit final submission tarballs.
- Do not commit raw long server logs or model artifacts.
