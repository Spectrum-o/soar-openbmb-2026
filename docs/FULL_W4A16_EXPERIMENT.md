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

python3 submission_gptqmodel_full_w4a16/quantize_gptqmodel_w4a16.py   --input /root/autodl-fs/models/OpenBMB/MiniCPM-SALA   --output /tmp/full_w4a16_dry   --calib-jsonl submission_gptqmodel_full_w4a16/perf_public_set.jsonl   --num-calib 2   --max-calib-len 512   --dry-run
```

## Interface Notes

- Platform calls `prepare_env.sh`, then `prepare_model.sh --input ... --output ...`.
- `prepare_env.sh` must export `SGLANG_SERVER_ARGS`; platform serving depends on it.
- Local OpenAI smoke requests should use `/v1/chat/completions` and bypass local
  proxy with `curl --noproxy '*'` or `no_proxy=localhost,127.0.0.1`.
- Do not mix FP8KV into this branch until full W4A16 alone is measured.
- Current full accuracy pass uses `GROUP_SIZE=64`, all 150 public calibration
  rows, multi-adaptive windows capped by `MAX_CALIB_WINDOWS=4`, and
  answer-aware calibration. With chat template enabled, compact `gold` answers
  are rendered as an `assistant` turn rather than appended into the user prompt.
- Compare new full results against the known full-g128 platform baseline:
  `acc_ori=78.27`, `final_score=22.85`. The preferred target is `acc_ori>=80`,
  but `78.27-80` is a slot-cost decision, not an automatic zero-score failure.
- While FP8KV is still being debugged, prefer
  `bash scripts/run_full_w4a16_platform_acc_now.sh`; it starts only if GPU
  memory and the target port are already free and otherwise exits without
  waiting.
- After the full local eval finishes, run
  `python3 scripts/full_w4a16_decide_after_eval.py` to summarize the latest
  result, inspect per-task failures, and choose either pack-for-platform or the
  next single-variable accuracy knob.

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
