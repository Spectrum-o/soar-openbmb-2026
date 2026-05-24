# GPU next-session runbook v2 — post-v5j_dtype_bf16 breakthrough

> 2026-05-24 written after v5j_dtype_bf16 returned platform `acc_ori=82.18,
> final_score=23.18` (the first non-baseline gain). This doc supersedes
> `GPU_NEXT_SESSION_RUNBOOK.md` which was for the v5j-pending state.

## Status snapshot

| Variant | Platform result | Best score |
|---|---|---|
| baseline (no extras) | final_score 19.13 | rank floor |
| v5g (BF16 native, 8K) | acc_ori 83.04, **19.12** | (= baseline) |
| v5j (W4A16+Marlin, fp16) | ⛔ dtype mismatch | — |
| **v5j_dtype_bf16 (W4A16+Marlin, bf16, 8K)** | acc_ori 82.18, **23.18** | **CURRENT BEST** |

W4A16 + Marlin + `--dtype bfloat16` is the production-viable base. From
here we stack more throughput knobs.

## What's pre-staged on this branch

5 packed tarballs in `~/OneDrive/soar_submissions/` and at repo root:

| Variant | tarball md5 | What it stacks |
|---|---|---|
| v5j_dtype_bf16 (already submitted) | `0df44fb3fb9d9ff5163be2f06e6d33e6` | base |
| **chunk32k** | `98315cf8e778b4ee39a1946c0964831f` | +chunked-prefill 32K |
| **chunk65k** | `3f7dc6180e6828752c930e5928a62c06` | +chunked-prefill 65K |
| **chunk32k_opfusion** | `c5e2cca7323e346baf4045cbbe200b19` | chunk32k + RMSNorm/RoPE fusion |
| **lightning_skip_bf16** | `ebadc7eaf887e84861298d42c97cb8a0` | mixed-precision (dense W4A16 + lightning BF16) |

## Submission strategy

The optimization landscape now has two orthogonal axes:

```
            acc                          throughput
             ↑                                ↑
  lightning_skip_bf16             chunk32k → chunk65k
             ↑                                ↑
   (cures cwe/niah collapse)           (more KV per chunk)
             ↑                                ↑
                                  chunk32k + op-fusion
                                              ↑
                                  (fused RMSNorm + skip rope upcast)
```

**Throughput axis** (chunk32k → chunk65k → +opfusion) is the SAFER path:
each step is single-variable, all bf16-confirmed, mem budgets verified.

**acc axis** (lightning_skip_bf16) is the higher-variance path: changes
artifact structure, could blow up the gate if cwe/niah quant errors
weren't actually the bottleneck, or could push acc 5+ pp higher if they
were.

## Recommended submission order

### Slot 1: **chunk32k**

- Single-variable change from verified-working v5j_dtype_bf16
- chunked-prefill is a perf knob (exp_h verified Δacc ≤ 0.22pp)
- Expected: acc ~82, S1 -25-30%, **final_score 28-32**
- Risk: ~5% (only thing that could go wrong is mem buffer interactions
  we missed)
- md5: `98315cf8e778b4ee39a1946c0964831f`

### Slot 2: **chunk65k** (IF chunk32k succeeds)

- Stack chunked-prefill all the way (1849 W4A16 used 65K successfully)
- Expected: incremental +5-10% throughput
- Risk: ~10% (6.4 GB headroom is tighter than chunk32k's 9 GB)
- md5: `3f7dc6180e6828752c930e5928a62c06`
- **Don't submit before chunk32k confirms acc stays at ~82**

### Slot 3: **chunk32k_opfusion** (IF chunk32k acc stays clean)

- Stack op-fusion (RMSNorm + RoPE fused) on chunk32k
- Expected: +5-10% on S1 specifically (decode hot path)
- Risk: ~20% (GPU op-fusion not verified end-to-end on SALA)
- md5: `c5e2cca7323e346baf4045cbbe200b19`
- **Smoke 30 samples first** — if smoke acc drops materially, don't submit

### Slot N: **lightning_skip_bf16** (independent track)

- Mixed-precision experiment, orthogonal to chunked-prefill / op-fusion
- Could go either way: +5pp acc (if lightning quant was the bottleneck)
  or no change (if quant overhead is uniform)
- Risk: ~15% (overlay tool itself was new code; bf16 path of overlay
  loading has never been tested)
- md5: `ebadc7eaf887e84861298d42c97cb8a0`
- **Smoke 30 samples REQUIRED**

## GPU session protocol

When GPU is available, in this order:

### 1. Quick startup verify (5 min)

```bash
cd /root/autodl-tmp/zyn/sglang
git pull origin quant/w4a16
source sglang_minicpm_sala_env/bin/activate

# Verify all 4 new variants have correct configs (catches merge accidents)
for v in chunk32k chunk65k chunk32k_opfusion; do
    echo "=== $v ==="
    grep "export SGLANG_SERVER_ARGS" \
        submission_gptqmodel_no_fp16_patch_dtype_bf16_${v}/prepare_env.sh
done
echo "=== lightning_skip_bf16 ==="
grep "export SGLANG_SERVER_ARGS" submission_w4a16_lightning_skip_bf16/prepare_env.sh
```

Expected output: all configs should have `--dtype bfloat16` and `--mem-fraction-static 0.80`.

### 2. chunk32k smoke FIRST (50 min, blocks everything else)

```bash
nohup bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k \
    --num-samples 30 \
    --force-requant > /tmp/chunk32k_smoke.log 2>&1 &

# Monitor:
tail -f /tmp/chunk32k_smoke.log

# Or watch GPU peak:
watch -n 1 'nvidia-smi --query-gpu=memory.used --format=csv'
```

Critical observations during smoke:
- Server start (no dtype mismatch crash like v5j)
- Peak memory usage (should be < 76 GB on 96 GB AutoDL; predicts < 75 GB on 84 GB platform)
- Final acc_ori on 30 samples (expect ~78-84, anything < 75 raises red flag)

### 3. Parallel smokes (if chunk32k passes)

After chunk32k clears (~50 min), can run in parallel:

```bash
# Smoke lightning_skip_bf16 (independent variant, different artifact)
nohup bash scripts/local_eval.sh \
    --variant submission_w4a16_lightning_skip_bf16 \
    --num-samples 30 --force-requant > /tmp/lightning_skip_bf16_smoke.log 2>&1 &

# After lightning_skip finishes its quant step (which takes longer due to
# the overlay), can also smoke op-fusion:
nohup bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion \
    --num-samples 30 --force-requant > /tmp/opfusion_smoke.log 2>&1 &
```

### 4. Decision

After all smokes:

| Variant | Smoke result | Action |
|---|---|---|
| chunk32k | acc_ori ≥ 78 | Already packed; submit it as Slot 1 |
| chunk32k | acc_ori < 75 OR crash | DO NOT submit; investigate first |
| chunk65k | (skip smoke; chunked-prefill is perf knob) | Submit after chunk32k confirms platform |
| chunk32k_opfusion | acc within 1pp of chunk32k smoke | Safe to submit |
| chunk32k_opfusion | acc drops > 2pp | GPU op-fusion bug; DO NOT submit |
| lightning_skip_bf16 | acc_ori ≥ 78 + valid output | Submit; could push acc higher |
| lightning_skip_bf16 | crash OR garbage output | Overlay tool bug; investigate |

## Mem budget cheat sheet (84GB platform)

```
W4A16 model (5GB):    chunk-prefill   mem-fraction-static   buffer    total   headroom
                        8K              0.80                 0.7       72.7    11.3 ✓
                        32K             0.80                 3.0       75.0    9.0 ✓
                        65K             0.80                 5.6       77.6    6.4 ✓
                        65K             0.85                 5.6       82.0    2.0 ⚠ tight
                        65K             0.90                 5.6       86.0    -2.0 ✗ OOM

Lightning_skip (~7-8GB mixed): 8K + 0.80 + 0.7 = 75.5  → 8.5 GB headroom ✓
                               32K + 0.80 + 3   = 77.5  → 6.5 GB headroom ✓

BF16 model (18GB):    8K              0.65                  0.7       73.6   10.4 ✓
                       32K             0.65                  3.0       76.0    8.0 ✓
                       (NEVER use mem-fraction 0.80 with BF16 + chunk32k+)
```

## What NOT to do (still applies)

- Don't submit v5h variant (BF16 + 32K) with mem-fraction 0.80 — will OOM
- Don't submit bf16_official_args (BF16 + 65K + 0.80) — proved OOM
- Don't submit v5j again (W4A16 + Marlin + dtype=fp16) — proved dtype crash
- Don't smoke calibration knobs — proved <5-8pp upside, much less than dtype/quant changes
- Don't try chunked-prefill > 65K on any variant — 1849's verified ceiling

## Reference

- Hardware: RTX 6000D = **84 GB** (platform) vs RTX PRO 6000 Blackwell = 96 GB (AutoDL)
- Best score so far: 23.18 (v5j_dtype_bf16, 00:06 submission)
- Pre-staged tarballs all in `~/OneDrive/soar_submissions/` (Windows path: `C:\Users\zyn\OneDrive\soar_submissions\`)
- Previous handoffs: `experiments/HANDOFF_20260523_afternoon_situation.md`, `experiments/GPU_NEXT_SESSION_RUNBOOK.md`
- v5j failure analysis in `SUBMISSIONS.md` row dated 2026-05-23 22:30
