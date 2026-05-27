# MLP-only artifact and long-tail notes, 2026-05-27

## Current direction

The active direction is still:

- MLP-only W4A16
- `gptq_marlin`
- `dtype=bfloat16`
- KV cache `fp8_e4m3`
- `minicpm_flashinfer`
- CUDA graph enabled with batch sizes `1 2 4 8 12 16 24 32`

The current strongest local baseline remains:

`/autodl-fs/data/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized`

with full 150 public-set accuracy `83.13` in:

`/root/autodl-tmp/zyn/eval_runs/fp8kv_cg_e4m3_20260526_180812`

`v25_multi_adaptive` was only checked as a candidate MLP-only artifact because
its quantization log looked lower-loss. It is not a direction change.

## v25_multi_adaptive first-30 result

Candidate:

`/autodl-fs/data/zyn/models/submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive-quantized`

Serving config matched the baseline except for the model artifact:

```bash
MODEL_PATH=/autodl-fs/data/zyn/models/submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive-quantized \
KV_DTYPE=fp8_e4m3 \
PORT=31111 \
MAX_RUNNING_REQUESTS=32 \
CUDA_GRAPH_BS="1 2 4 8 12 16 24 32" \
bash run_sala.sh
```

The server loaded successfully and captured CUDA graph. The checkpoint load was
slow from `/autodl-fs`, but it was not a CUDA graph failure.

First-30 evaluation was stopped after 28 completed samples:

- completed: `28 / 30`
- missing long-tail indices: `[10, 21]`
- completed score sum: `22.4`
- completed accuracy: `80.0`
- best possible first-30 accuracy if both missing samples were perfect:
  `81.33`

This is below the current baseline first-30 accuracy `85.67`, so the candidate
is rejected and should not be expanded to 60 or 150 samples.

Same-index comparison:

```text
safe_e4m3: n=30 acc=85.67 out=103782 avg_out=3459.4 max_out=65546
  cwe: n=6 acc=95.0 wrong=[10, 15, 20] avg_out=11558.3
  fwe: n=6 acc=100.0 wrong=[] avg_out=746.7
  mcq: n=6 acc=66.67 wrong=[7, 27] avg_out=4441.8
  niah: n=6 acc=100.0 wrong=[] avg_out=445.5
  qa: n=6 acc=66.67 wrong=[19, 29] avg_out=104.7
multi_adaptive: n=28 acc=80.0 out=35694 avg_out=1274.8 max_out=7374
  cwe: n=5 acc=88.0 wrong=[0, 5, 15, 20, 25] avg_out=615.0
  fwe: n=5 acc=100.0 wrong=[] avg_out=744.8
  mcq: n=6 acc=66.67 wrong=[7, 27] avg_out=4307.5
  niah: n=6 acc=100.0 wrong=[] avg_out=411.3
  qa: n=6 acc=50.0 wrong=[14, 19, 29] avg_out=97.0
```

## v25_g64 load failure

Candidate:

`/autodl-fs/data/zyn/models/submission_gptqmodel_calib_w4a16_v25_g64-quantized`

Serving config matched the baseline except for the model artifact. The artifact
did not reach evaluation: SGLang failed while loading checkpoint shard 1 with a
Marlin parameter shape assertion:

```text
python/sglang/srt/layers/parameter.py:172
assert param_data.shape == loaded_weight.shape
```

The failure happened in `load_merged_column_weight()` for a fused column-linear
weight. The artifact uses `group_size=64`; the current MiniCPM-SALA Marlin load
path has only been verified with the runnable `group_size=128` artifacts. Treat
this candidate as incompatible with the current serving stack unless the loader
layout is explicitly fixed and smoke-tested.

Conclusion: do not submit or expand `v25_g64`.

## Long-tail interpretation

Long outputs mainly hurt runtime first. They do not automatically imply a wrong
answer.

Evidence from the full `safe_e4m3` 150-sample baseline:

- `11` samples reached about `65546` output tokens.
- Their combined score was `9.6 / 11`.
- The `fwe` max-output samples all scored `1.0`.
- Some `cwe` max-output samples still scored `0.8` or `0.9`.

Max-output samples in the full baseline:

```text
cwe: indices [10, 50, 90, 110, 130, 135], average score 0.767
fwe: indices [101, 96, 106, 141, 146], average score 1.0
```

However, long outputs are still a platform-risk issue because a few tail
requests can dominate total wall time. The platform fp8kv submission running
longer than the previous 2.5h runs is therefore more likely to be caused by
tail requests decoding to `max_tokens` than by the server failing to start,
assuming the platform has already reached the inferencing stage.

Known local long-tail example from the baseline:

```text
index=10 task=cwe score=0.9 output_tokens=65546
prediction tail repeatedly says: "I've re-checked the list."
```

This is not safe to fix with an arbitrary repetition stop rule unless that rule
is explicitly accepted, because it can change benchmark behavior and may be
considered an output heuristic rather than a quantization improvement.

## MR=8 diagnostic

One low-concurrency diagnostic was run on the current best artifact:

- model: `submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized`
- KV dtype: `fp8_e4m3`
- CUDA graph: enabled, bs `1 2 4 8`
- server `MAX_RUNNING_REQUESTS=8`
- client concurrency `8`
- run dir: `/root/autodl-tmp/zyn/eval_runs/safe_e4m3_mr8_first30_20260527`

The run was stopped at 29 completed samples because it is not a final serving
strategy. Same-index comparison against the normal first-30 baseline:

```text
safe_e4m3: n=30 acc=85.67 out=103782 avg_out=3459.4 max_out=65546
  cwe: n=6 acc=95.0 wrong=[10, 15, 20] avg_out=11558.3
  fwe: n=6 acc=100.0 wrong=[] avg_out=746.7
  mcq: n=6 acc=66.67 wrong=[7, 27] avg_out=4441.8
  niah: n=6 acc=100.0 wrong=[] avg_out=445.5
  qa: n=6 acc=66.67 wrong=[19, 29] avg_out=104.7
mr8: n=29 acc=84.83 out=38666 avg_out=1333.3 max_out=7192
  cwe: n=5 acc=92.0 wrong=[5, 15, 20] avg_out=716.6
  fwe: n=6 acc=100.0 wrong=[] avg_out=749.0
  mcq: n=6 acc=83.33 wrong=[27] avg_out=4499.7
  niah: n=6 acc=100.0 wrong=[] avg_out=490.2
  qa: n=6 acc=50.0 wrong=[14, 19, 29] avg_out=108.3
```

The useful signal is attribution only: some samples are sensitive to batching
pressure and output tail behavior. It is not a submission strategy because
lowering max-running requests would hurt benchmark duration and final score.
Keep CUDA graph and normal concurrency for candidate selection.

## Platform tail-risk note

The live fp8-KV submission was reported still running at about 4 hours on
2026-05-27, while earlier successful submissions usually finished within about
2.5 hours. The later platform status showed it was still in `DOWNLOADING` /
submit preparation, not `INFERENCING`.

That changes the primary interpretation: treat it first as a prepare/startup
problem rather than a correctness long-tail problem. The most actionable
suspect for the fp8kv package family is preserving FlashInfer's cache instead
of clearing `~/.cache/flashinfer`, because clearing it forces cold SM120 JIT on
the next service launch. See `PLATFORM_TEST_LOGIC_20260527.md` for the platform
stage breakdown.

This does not justify arbitrary stop rules or answer parsing. The next aligned
optimization is to improve the MLP-only artifact under the normal serving
configuration so fewer samples enter unstable long-output tails.

## Next action

Do not continue `v25_multi_adaptive`.

Reasonable next actions are:

1. Keep `safe_e4m3` as the current best runnable candidate.
2. Ask for or inspect the platform stage log. If it is in `INFERENCING`, expect
   long-tail requests; if it is still in `PREPARING`, inspect package download
   and setup time instead.
3. Only test another MLP-only artifact if it can be screened on first-30 and
   rejected early when its best possible score drops below `85.67`.
4. Next MLP-only GPTQ quality candidate should be quantized from the BF16 base
   `/root/autodl-fs/models/OpenBMB/MiniCPM-SALA`, not from the compressed
   `/autodl-fs/data/zyn/models/MiniCPM-SALA-W4A16` directory.
