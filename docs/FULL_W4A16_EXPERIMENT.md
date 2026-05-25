# Full W4A16 Experiment Branch Notes

Branch: `exp/full-w4a16`

Purpose: evaluate whether MiniCPM-SALA can survive full GPTQModel W4A16 after
the current loader, tokenizer, qzeros, and calibration fixes. This branch is
not the stable FP8KV serving path.

## Submission Directory

`submission_gptqmodel_full_w4a16/`

The directory is self-contained for packaging through `tools/pack_submission.py`.
It reuses stable shared assets from `submission_gptqmodel_calib_w4a16` through
symlinks and carries its own full-scope quantizer and README.

## Required Checks Before Upload

```bash
python3 tools/pack_submission.py   --variant submission_gptqmodel_full_w4a16   --check-only

python3 submission_gptqmodel_full_w4a16/quantize_gptqmodel_w4a16.py   --input /root/autodl-fs/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_dequant-quantized   --output /tmp/full_w4a16_dry   --calib-jsonl submission_gptqmodel_full_w4a16/perf_public_set.jsonl   --num-calib 2   --max-calib-len 512   --dry-run
```

## Interface Notes

- Platform calls `prepare_env.sh`, then `prepare_model.sh --input ... --output ...`.
- `prepare_env.sh` must export `SGLANG_SERVER_ARGS`; platform serving depends on it.
- Local OpenAI smoke requests should use `/v1/chat/completions` and bypass local
  proxy with `curl --noproxy '*'` or `no_proxy=localhost,127.0.0.1`.
- Do not mix FP8KV into this branch until full W4A16 alone is measured.

## Push Notes

This environment rewrites `git@github.com:` to HTTPS globally, while HTTPS has
no username/token. Use this form for reliable push from AutoDL:

```bash
env GIT_CONFIG_GLOBAL=/dev/null   git push git@github.com:Spectrum-o/soar-openbmb-2026.git exp/full-w4a16
```

If push fails after a local commit, save a persistent patch before doing any
cleanup:

```bash
mkdir -p /root/autodl-fs/zyn/backups/full_w4a16_push

git format-patch origin/quant/w4a16..HEAD   --output-directory /root/autodl-fs/zyn/backups/full_w4a16_push
```
