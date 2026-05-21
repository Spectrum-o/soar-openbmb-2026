# v25_calib_multi_adaptive — fork of v23 with multi-adaptive calibration

> **PRECONDITION**: only submit if v23 platform returns `0 < acc_ori < 60`
> (correctness fixes worked but the model is gated on calibration quality).
> If v23 returns 0, this variant is unlikely to help — the issue is
> structural, not calibration recipe. If v23 returns >= 60, this is the
> path to push toward the 80 correctness gate.

## The diff vs v23

In `prepare_model.sh`, change two defaults:

```diff
- NUM_CALIB="${NUM_CALIB:-256}"
+ NUM_CALIB="${NUM_CALIB:-300}"

- CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-tail}"
+ CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"
```

## Why this might matter

v23's calibration uses **tail-only windowing**: for each of 256 prompts,
the LAST 8K tokens of the prompt go to the Hessian. This captures the
"answer-shaped" structure (question + answer markers near the end) but
ignores the long-context body of niah/cwe prompts (where the actual
needle / words are buried).

Multi-adaptive windowing (already supported by the quantize script via
`--calib-window-mode multi-adaptive`) splits each prompt into 1-3
8K-sized non-overlapping windows:
- short prompt (≤ 8K): full prompt
- medium (≤ 32K): tail only
- long (≤ 100K): tail + centered mid
- super-long (> 100K): tail + mid + one random mid

This roughly 1.4-2x the effective calibration sample count. Combined
with `NUM_CALIB=300` (up from 256), the Hessian sees ~420 windows total,
covering both the answer-shaped tails AND the long-context bodies.

Hypothesis: niah / cwe failures in v21 local were partly caused by
calibration NOT exposing the model's "haystack" attention pattern.
Multi-adaptive should reduce niah/cwe repetition collapse and push
those task pass rates from ~30-47% to ~60-80%.

## Variant dir structure

```
submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive/
├── README_SUBMISSION.md             ← tracked, this file
├── prepare_model.sh                 ← tracked, 2-line diff
├── prepare_env.sh                   → symlink to v23
├── quantize_gptqmodel_w4a16.py      → symlink to v23
├── perf_public_set.jsonl            → symlink to v23
├── flash_attn-...whl                → symlink to v23
└── sglang                           → symlink to v23
```

## Pack + submit

```bash
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive \
    --check-only

python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive \
    --suffix _v25_multi_adapt --output-dir .
```

## Timing considerations

Multi-adaptive with 300 input prompts produces ~420-500 calibration
samples. Per the quantize script docstring, this could push the 90-min
quant timeout. Compensating: NUM_CALIB is capped at 300 input (not 512
as the docstring's worst-case warning), keeping us under the timeout.

If quant runs hit the 90-min timeout on platform: lower `NUM_CALIB`
override to 200 (still produces ~280-330 windows after multi-adaptive
inflation, fewer than v23's 256 tail but covering wider context).

## Expected outcomes vs diagnosis

| Platform result | What it means | Next action |
|---|---|---|
| `acc_ori >= 70` | Multi-adaptive calib closed the ceiling gap | Cherry-pick into v23 base; ship as canonical recipe; combine with chunked-prefill 65K for v26-perf |
| 50 ≤ `acc_ori` < 70 | Some improvement; calibration helps but not the whole story | Continue with v25_g64 (finer granularity) |
| `acc_ori` ≈ v23's value | Calibration recipe wasn't the bottleneck | Try v25_g64 or W8A16 |
| `acc_ori` < v23 | Multi-adaptive REGRESSED. Possibly the wider context confused GPTQ | Revert to tail, look at per-task breakdown |
| Quant timed out | Sample count too high; lower `NUM_CALIB` | Re-pack with NUM_CALIB=200 baked in |

## Watchpoints in the platform log

```bash
grep "samples=\|calib examples" platform_v25.log | head -5
# expect: "starting GPTQModel W4A16 group_size=128 samples=N max_len=8192"
# where N is in the 280-500 range, NOT 256
```

If N == 256 on the platform but we set NUM_CALIB=300, the env override
didn't take effect — check that the prepare_model.sh diff actually
reached the platform via the tarball.
