# v5j_dtype_bf16_chunk32k — stack throughput on the verified W4A16+bf16 base

> Single-variable extension of v5j_dtype_bf16 (which scored
> acc_ori=82.18 / final_score=23.18 on platform 2026-05-24 02:29).
> Adds chunked-prefill 32K to take the next throughput step.

## What this variant tests

v5j_dtype_bf16 proved the **W4A16 + Marlin + `--dtype bfloat16`** path
(acc_ori 82.18, +4.06 final_score over v5g's BF16 baseline). It used
conservative chunked-prefill 8192.

This variant stacks chunked-prefill 32K — the next safe throughput knob
per `exp_h_chunked_prefill_revert` (Δacc ≤ 0.22pp between 8K and 65K,
verified — it's a perf knob not an acc knob).

## Diff vs v5j_dtype_bf16

Two-token change in `prepare_env.sh`:

```diff
-    --chunked-prefill-size 8192
+    --chunked-prefill-size 32768 --max-prefill-tokens 32768
```

Everything else symlinked.

## Expected platform result

| Metric | v5j_dtype_bf16 (8K) | predicted (32K) |
|---|---|---|
| acc_ori | 82.18 | ~82 (perf knob) |
| S1 | 717.76 | ~500–550 |
| S8 | 1067.43 | ~850–900 |
| Smax | 2343.38 | ~2150–2250 |
| final_score | 23.18 | **~28–32** |

## Mem budget (84GB platform)

5 (model) + 0.80×84 KV (67) + 3 (32K buffer) = 75 ≤ 84 ✓ (9 GB headroom).

## Pre-submission smoke (highly recommended)

```bash
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k \
    --num-samples 30 --force-requant
```

Look for acc_ori ≥ 75 (within noise of 82) and S1 < 717.

## Pack

```bash
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k \
    --output soar_w4a16_bf16_chunk32k_$(date +%Y%m%d_%H%M%S).tar.gz
```
