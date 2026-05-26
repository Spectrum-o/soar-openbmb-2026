# Full W4A16 mixed skip30/31 down failure — 2026-05-27

## Summary

The narrow mixed-sensitive experiment that kept only
`model.layers.{30,31}.mlp.down_proj` in BF16 is a negative result.
Do not continue this exact direction.

It was chosen from the uniform full-W4A16 GPTQ log because those two
`down_proj` modules had the highest local GPTQ reconstruction loss. The
sharded eval shows that this criterion did not predict downstream accuracy:
the mixed artifact was worse than the uniform full-g64 artifact on the same
first 30 public rows.

## Artifacts

| Item | Path / commit |
|---|---|
| Uniform full g64 artifact | `/root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16_platform_acc-quantized` |
| Mixed artifact | `/root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16_g64_skip30_31_down-quantized` |
| Mixed quant log | `/root/autodl-fs/zyn/logs/full_w4a16_g64_skip30_31_down_quant.log` |
| Mixed shard-1 checkpoint | `185ca3f31 checkpoint full w4a16 shard 1 cumulative 74.67` |
| Stop checkpoint | `eae84c4f7 checkpoint full w4a16 stop mixed shard 2` |
| Shard CSV | `scripts/eval_shards.csv` |

The mixed artifact verifier passed. Layers 30 and 31 `mlp.down_proj` remained
BF16, while adjacent MLP projections and attention projections remained GPTQ
W4A16 g64.

## Results

Same local sharded setup, same first 30 public rows, concurrency 32, CUDA graph
enabled.

| Run | Samples | Acc | Output tokens | Duration |
|---|---:|---:|---:|---:|
| Uniform full g64 | 0-29 | 81.33 | 235,361 | 761.51 s |
| Mixed skip30/31 down | 0-29 | 74.67 | 297,743 | 765.46 s |
| FP8KV snapshot reference | 0-29 | 85.67 | 103,782 | n/a |

Task-level comparison on the same first shard:

| Task | Uniform full g64 | Mixed skip30/31 down | Change |
|---|---:|---:|---:|
| cwe | 90.00 | 90.00 | 0.00 |
| fwe | 100.00 | 100.00 | 0.00 |
| mcq | 50.00 | 50.00 | 0.00 |
| niah | 100.00 | 83.33 | -16.67 |
| qa | 66.67 | 50.00 | -16.67 |

The second shard was intentionally stopped after shard 1 because the candidate
was already below both the uniform full g64 run and the fp8kv reference.

## Diagnosis

This was not a platform-serving failure:

- CUDA graph was enabled (`disable_cuda_graph=False`) and captured.
- The server log contains repeated `cuda graph: True`.
- No server `ERROR` or `Traceback` appeared before the shard-1 result.
- The mixed quant artifact layout passed `tools/verify_full_w4a16_artifact.py`.

The failure is experimental: GPTQ reconstruction loss on individual modules was
not a reliable proxy for downstream task sensitivity. Keeping only late
`down_proj` modules in BF16 changed the generation distribution in a harmful
way, especially for retrieval/QA tasks, while it did not fix the long-output
problem that full W4A16 already showed.

Observed symptoms:

- Mixed output tokens increased from 235,361 to 297,743 on shard 1.
- NIAH regressed from 100.00 to 83.33.
- QA regressed from 66.67 to 50.00.
- Some long outputs still ran to the max-token region, so the issue is not
  solved by preserving only the two highest-loss `down_proj` tensors.

## Decision

Park this full-W4A16 mixed candidate. Short-term effort should move to the
known stronger route: MLP-only W4A16 plus the fp8kv flashinfer/cudagraph stack,
then measure that combined path with the same sharded evaluator before a
platform submission.

If full W4A16 is revisited later, choose candidates from downstream error
analysis rather than GPTQ module loss alone. The next full-specific sensitivity
tests should focus on attention projections, logits/norm boundaries, or a
controlled per-task ablation, not `layers 30,31 mlp.down_proj` alone.
