# MLP-only W4A16 KV dtype comparison, 2026-05-27

Goal: improve the MLP-only W4A16 accuracy path without task-specific decoding
hacks.  CUDA graph stayed enabled in all server runs.

## Baseline

Model artifact:

`/autodl-fs/data/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized`

The strongest complete local baseline remains:

`/root/autodl-tmp/zyn/eval_runs/fp8kv_cg_e4m3_20260526_180812`

Configuration:

- quantization: `gptq_marlin`
- model dtype: `bfloat16`
- KV dtype: `fp8_e4m3`
- attention backend: `minicpm_flashinfer`
- CUDA graph: enabled, bs `1 2 4 8 12 16 24 32`

Full 150-sample result:

- overall acc: `83.13`
- task acc: `cwe=89.0`, `fwe=100.0`, `mcq=63.33`, `niah=100.0`, `qa=63.33`

## Same-index comparisons

All numbers below are on public-set indices `[0, 30)`, except the BF16 KV run
which was stopped at 24 completed samples because it was already lower accuracy
and the remaining requests entered a long-output tail.

| run | completed | acc | output tokens | avg output | main observation |
| --- | ---: | ---: | ---: | ---: | --- |
| `fp8_e4m3` | 30 | 85.67 | 103782 | 3459.4 | best observed KV setting |
| `fp8_e5m2` | 30 | 73.89 | 37338 | 1244.6 | strong negative, mainly MCQ/QA |
| `bf16/auto KV` | 24 | 77.36 | 30188 | 1257.8 | no accuracy lift; long-output tail on remaining samples |

Task breakdown:

`fp8_e4m3`:

- `cwe`: 95.0, wrong/partial indices `[20, 15, 10]`
- `fwe`: 100.0
- `mcq`: 66.67, wrong indices `[27, 7]`
- `niah`: 100.0
- `qa`: 66.67, wrong indices `[19, 29]`

`fp8_e5m2`:

- `cwe`: 91.67, wrong/partial indices `[0, 20, 10, 15]`
- `fwe`: 94.44, wrong/partial index `[6]`
- `mcq`: 33.33, wrong indices `[7, 27, 17, 22]`
- `niah`: 100.0
- `qa`: 50.0, wrong indices `[19, 29, 14]`

`bf16/auto KV` partial:

- `cwe`: 95.0 on 2 completed samples, wrong/partial index `[20]`
- `fwe`: 91.67 on 4 completed samples, wrong/partial index `[26]`
- `mcq`: 50.0, wrong indices `[7, 2, 27]`
- `niah`: 100.0
- `qa`: 66.67, wrong indices `[14, 29]`

## Conclusion

Changing KV dtype is not the next useful accuracy lever:

- `fp8_e5m2` is clearly worse than `fp8_e4m3` on the same first 30 samples.
- BF16 KV did not provide a clean accuracy upper bound.  It fixed some behavior
  but regressed other samples, and it hit a long-output tail before completing
  all 30 samples.
- The current best practical direction remains `fp8_e4m3` with CUDA graph
  enabled, then improve MLP-only W4A16 artifact/output stability.

Next reasonable steps:

1. Compare existing MLP-only artifacts under the same `fp8_e4m3` serving config.
2. Inspect MCQ/QA wrong samples for generic output-stability issues.
3. Avoid task-specific postprocessing, answer parsing changes, or arbitrary
   repetition-stop heuristics.

Reproduce the comparison table with:

```bash
python scripts/zyn_compare_eval_runs.py \
  --start-index 0 \
  --end-index 30 \
  --run e4m3=/root/autodl-tmp/zyn/eval_runs/fp8kv_cg_e4m3_20260526_180812/predictions.jsonl \
  --run e5m2=/root/autodl-tmp/zyn/eval_runs/fp8kv_cg_e5m2_first60_20260527/predictions.jsonl \
  --run bf16kv=/root/autodl-tmp/zyn/eval_runs/bf16kv_auto_first30_20260527/predictions.jsonl
```
