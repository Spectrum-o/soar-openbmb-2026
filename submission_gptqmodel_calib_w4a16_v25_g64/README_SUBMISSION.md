# v25_g64 — fork of v23 with group_size=64 instead of 128

> **PRECONDITION**: only submit if v23 platform returns `0 < acc_ori < 70`
> (the structural fixes worked but per-group quantization error is
> still meaningful). If v23 returns 0, this variant is unlikely to
> help — the issue is structural, not granularity.

## The diff vs v23

In `prepare_model.sh`, append `--group-size 64` to `EXTRA_ARGS`:

```diff
- EXTRA_ARGS+=(--no-offload-disk)
+ EXTRA_ARGS+=(--no-offload-disk --group-size 64)
```

## Why this might matter

GPTQ quantization clusters weights into groups of `group_size` columns
and computes one scale + zero point per group per row. group_size=128
(v23 default) clusters 128 weights together; group_size=64 doubles the
granularity (half the cluster size, twice as many scales).

Effect on accuracy:
- Finer groups → smaller per-cluster scale range → smaller quantization
  error on outlier weights within the cluster → ~1-3% accuracy gain on
  hard tasks (literature consensus on GPTQ + LLaMA-family models).

Effect on memory and perf:
- Approximately 2x scales tensor size (`scales` shape is
  `[in_features/group_size, out_features]`)
- ~2-3% slower Marlin GEMM (more memory accesses per output column)
- Final model size grows ~3% (scales become non-negligible)

Net trade: +1-3% acc / -2-3% perf / +3% memory. Reasonable trade if v23
is currently gated on acc.

## Variant dir structure

```
submission_gptqmodel_calib_w4a16_v25_g64/
├── README_SUBMISSION.md             ← tracked, this file
├── prepare_model.sh                 ← tracked, 1-line diff
├── prepare_env.sh                   → symlink to v23
├── quantize_gptqmodel_w4a16.py      → symlink to v23
├── perf_public_set.jsonl            → symlink to v23
├── flash_attn-...whl                → symlink to v23
└── sglang                           → symlink to v23
```

## Pack + submit

```bash
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v25_g64 \
    --check-only

python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v25_g64 \
    --suffix _v25_g64 --output-dir .
```

## What stays the same vs v23

- All H1 + H4 fixes (hardened qzeros, tokenizer overwrite)
- MLP-only quant scope
- SGLang launch args (chunked-prefill 8192 — single variable from v23)
- Calibration recipe (tail, 256 samples, left-truncation, chat template ON)
- bits=4

## Expected outcomes vs diagnosis

| Platform result | What it means | Next action |
|---|---|---|
| `acc_ori >= v23 + 5pp` | Finer granularity helped meaningfully | Cherry-pick into v23 base + combine with v25_calib_multi_adaptive for v26 |
| `acc_ori ≈ v23` | Granularity wasn't the bottleneck | Try v25_calib_multi_adaptive instead, or W8A16 if not yet tried |
| `acc_ori < v23` | Regressed. Unusual — would suggest interaction with the dynamic skip pattern or qzeros encoding at this group size. Check `[qzeros-fix] SUMMARY` for an anomaly in the patched count | Revert to g128 |
| `acc_ori ≈ v23` AND benchmark_duration -3% | Confirms g64's perf cost is real and acc didn't move; ship g128 | Stick with g128 |

## Marlin kernel compatibility

Marlin supports group_size ∈ {32, 64, 128, -1 (per-column)}. We pick 64
as a safe step from 128. Going to 32 would be even finer but Marlin's
32-grouped kernel is less commonly tuned and may have a larger perf hit.

## Pairs well with

If g64 helps, the natural follow-up is v26-perf = v25_g64 + chunked-prefill 65K
(uses the same v24-perf chunked-prefill pattern but with g64 baked in).
