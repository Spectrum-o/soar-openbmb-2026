# v5j_dtype_bf16_chunk32k_opfusion — stack op-fusion on the verified W4A16+bf16+32K base

> Extension of v5j_dtype_bf16_chunk32k. Applies the perf/op-fusion
> branch's minicpm.py overlay for additional decode-side throughput.

## What this variant tests

The op-fusion overlay does two things (commit 79b0c20f4):

1. **Drop float() upcast around rotary_emb** in MiniCPMAttention and
   MiniCPMLightningMixer. `apply_rope` already runs fp32 internally,
   so the explicit upcast is redundant work.

2. **Adopt residual-delay pattern** so RMSNorm dispatches through
   sglang's fused `add_rmsnorm` kernel.

Both changes target the decode hot path. Expected additional throughput
gain: **5-10% on S1, less on Smax** (large-batch attention is already
dominated by mat-muls, not norm/rope).

## Verification status

- **CPU pure-torch equivalence**: verified (commit 87c22ec88)
- **GPU end-to-end correctness with SALA kernels**: NEVER verified
- **Risk**: small probability the overlay is numerically incorrect on
  the platform's specific GPU+CUDA combination → fall back to chunk32k

## Diff vs v5j_dtype_bf16_chunk32k

Two additions:
1. `overlay/minicpm.py` (the fused-kernel-using replacement file from
   perf/op-fusion branch)
2. prepare_env.sh now has an "OP-FUSION OVERLAY" block that `cp`s the
   overlay over the bundled sglang's minicpm.py + verifies it imports cleanly

SGLANG_SERVER_ARGS unchanged from chunk32k.

## Pre-submission smoke (REQUIRED)

```bash
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion \
    --num-samples 30 --force-requant
```

Compare to chunk32k smoke (when available):
- acc_ori should be within 1pp of chunk32k's
- S1 should be 5-10% lower than chunk32k's

If acc DROPS materially, op-fusion is numerically wrong on this hardware —
do NOT submit; fall back to chunk32k.

## Expected platform

| Metric | chunk32k pred | **opfusion pred** |
|---|---|---|
| acc_ori | ~82 | ~82 ± 1 |
| S1 | ~500-550 | **~460-510** |
| final_score | ~28-32 | **~30-35** |

The fusion gains are modest (~5-10% decode speedup) but compound with
chunked-prefill 32K's prefill gains, so the total final_score may
edge into 30+ territory.

## Mem budget (84GB platform)

Same as chunk32k: 5+67+3 = 75 ≤ 84 ✓ (9 GB headroom).
op-fusion doesn't change mem footprint, only kernel dispatch.
