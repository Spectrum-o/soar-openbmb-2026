# w4a16_lightning_skip_bf16 — mixed-precision on the bf16 path

> Fork of submission_w4a16_lightning_skip with:
>   - fp16 sed-patch REMOVED (causes dtype mismatch crash on bf16 path)
>   - --dtype float16 → --dtype bfloat16
>   - mem-fraction-static 0.80 explicit

## What this variant tests

The original `w4a16_lightning_skip` variant was built on the BROKEN fp16
path (it had the sed-patch + `--dtype float16`). On 2026-05-23 23:38 we
proved that combination crashes with `RuntimeError: query and key must
have the same dtype` at `infllm_cuda.varlen_fwd_stage1`.

This fork rebases lightning_skip on the WORKING bf16 path (verified by
v5j_dtype_bf16 at 23.18 final_score).

The mixed-precision idea: **dense attention layers + their MLPs are
W4A16-quantized, but Lightning attention layers' MLPs revert to BF16**.
Targets the architectural hypothesis that lightning recurrence h_t
accumulates quantization errors over long sequences → repetition
collapse on niah/cwe tasks.

## Diff vs w4a16_lightning_skip (original)

prepare_env.sh:
```diff
-# fp16 sed-patch block
-sed -i 's/torch.bfloat16/torch.float16/g' minicpm_backend.py
-sed -i 's/torch.bfloat16/torch.float16/g' minicpm_sparse_utils.py
+# SKIPPED. Path 1b confirmed bf16 backend works.

-SGLANG_SERVER_ARGS="... --dtype float16"
+SGLANG_SERVER_ARGS="... --mem-fraction-static 0.80 --dtype bfloat16"
```

Everything else (quantize_gptqmodel_w4a16.py, apply_lightning_skip_overlay.py,
prepare_model.sh, sglang bundle, calib data) is symlinked from the
original lightning_skip variant.

## Mem budget (84GB platform)

Lightning skip artifact is partially W4A16 + partially BF16. Estimate:
~32 layers, half are dense (W4A16 = ~2.5 GB) + half are Lightning
(MLP reverted to BF16 ≈ ~5 GB) → total model ~7-8 GB.

```
7.5 GB model + 0.80×84 KV (67) + 0.7 GB buffer (8K) = 75.2 GB ≤ 84 ✓
   ← 8.8 GB headroom
```

Comfortable. If we later stack chunked-prefill 32K, total = 77.2 GB
still fits.

## Expected platform

| Metric | v5j_dtype_bf16 (8K) | lightning_skip_bf16 (8K) |
|---|---|---|
| acc_ori | 82.18 | **predicted 84-88** ← lightning MLPs BF16 should cure cwe/niah |
| final_score | 23.18 | ? (depends on how much acc helps Smax) |

Risk: untested artifact loader behavior with the overlay. **GPU smoke
30 samples required before platform submission.**

## Smoke

```bash
bash scripts/local_eval.sh \
    --variant submission_w4a16_lightning_skip_bf16 \
    --num-samples 30 --force-requant
```
