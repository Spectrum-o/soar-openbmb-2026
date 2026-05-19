# SOAR Submission Log

Tracks every tarball produced for the OpenBMB SOAR 2026 platform.

> **Convention**: tarballs live at repo root, source folders are `submission_*/` (the `sglang/` subdir inside each is gitignored — regeneratable). Failed/superseded tarballs are kept locally for debugging but NOT committed.

## Status legend
- 🟢 passed both correctness gate (>= 80) and produced a final_score
- 🟡 passed correctness but suboptimal score
- 🔴 failed (startup error or correctness < 80 or 5h timeout)
- ⏳ submitted, waiting for result
- 📦 built but not yet submitted

## Submissions

| Date | Tarball | Source folder | Status | final_score | Notes |
|---|---|---|---|---|---|
| 2026-05-14 | `soar_w4a16_submission_20260514.tar.gz` | (not in repo) | 🔴 | — | Earliest W4A16 attempt; superseded |
| 2026-05-15 | `soar_stable_gptq_rtn_submission_20260515.tar.gz` | (not in repo) | 🔴 timeout | — | Plain GPTQ + `--disable-cuda-graph`; ran into 5h timeout. RTN script wrote `sym=False`. |
| 2026-05-15 | `soar_gptqmodel_marlin_fp8kv_submission_20260515.tar.gz` | (not in repo) | 🔴 | — | GPTQModel route + FP8 KV. FP8 KV path incompatible with MiniCPM sparse backend (FlashAttention only supports fp16/bf16). |
| 2026-05-15 | `soar_gptqmodel_marlin_fp8kv_submission_20260515_v2.tar.gz` | `submission_soar_w4a16/` | 🔴 | — | Retry of above; same FP8 KV incompatibility. |
| 2026-05-19 | `soar_official_rtn_w4a16_g128_marlin_cfg_submission_20260519.tar.gz` | (not in repo) | 🔴 startup | — | RTN + Marlin. Failed: `Unsupported quantization config: bits=4, sym=False`. RTN script wrote asymmetric config; Marlin requires `sym=True`. |
| 2026-05-19 | `soar_rtn_sym_w4a16_g128_marlin_submission_20260519.tar.gz` | `submission_rtn_sym_w4a16/` (pre-fix) | 🔴 acc | 0.0 | **v1 of symmetric RTN.** Started OK (sym=True fix worked), ran 5h, but `acc_ori=42.51` vs ~82 baseline → correctness gate failed → `final_score=0`. Root cause: scale formula bug (`/ half` instead of `/ (half - 1)`) clamped +ve outliers by 12.5%. Bench durations recorded: S1=599.97s, S8=969.73s, Smax=2259.68s. |
| 2026-05-19 | `soar_rtn_sym_w4a16_marlin_v2_chunk32k_submission_20260519.tar.gz` | `submission_rtn_sym_v2_chunk32k/` (pre-fix) | 🔴 superseded | — | Same scale bug as v1 — would also fail correctness. Replaced by `_scalefix` build. |
| 2026-05-19 | `soar_rtn_sym_w4a16_marlin_v3_mixedchunk_submission_20260519.tar.gz` | `submission_rtn_sym_v3_mixedchunk/` (pre-fix) | 🔴 superseded | — | Same scale bug as v1 — would also fail correctness. Replaced by `_scalefix` build. |
| 2026-05-19 | `soar_rtn_sym_w4a16_marlin_v4_aggressive_submission_20260519.tar.gz` | `submission_rtn_sym_v4_aggressive/` (pre-fix) | 🔴 superseded | — | Same scale bug as v1 — would also fail correctness. Replaced by `_scalefix` build. |
| 2026-05-19 | `soar_rtn_sym_w4a16_g128_marlin_submission_20260519_scalefix.tar.gz` | `submission_rtn_sym_w4a16/` | 📦 | — | **v1.1.** Fix for the scale formula bug (`scales = w_absmax / (half - 1)`). Submit this NEXT — disambiguates "scale bug" vs "RTN itself too lossy". |
| 2026-05-19 | `soar_rtn_sym_w4a16_marlin_v2_chunk32k_submission_20260519_scalefix.tar.gz` | `submission_rtn_sym_v2_chunk32k/` | 📦 | — | v2 with scale fix. Submit after v1.1 confirms correctness gate clears. |
| 2026-05-19 | `soar_rtn_sym_w4a16_marlin_v3_mixedchunk_submission_20260519_scalefix.tar.gz` | `submission_rtn_sym_v3_mixedchunk/` | 📦 | — | v3 with scale fix. |
| 2026-05-19 | `soar_rtn_sym_w4a16_marlin_v4_aggressive_submission_20260519_scalefix.tar.gz` | `submission_rtn_sym_v4_aggressive/` | 📦 | — | v4 with scale fix. |

## Reference

- **Baseline (no quant, BF16, no extra flags)**: `final_score = 19.13` — this is the number to beat.
- **SOAR scoring tiers**: S1 (concurrency=1), S8 (=8), Smax (∞), via `bench_serving.sh`. Same submission is scored at all three; "single-batch" and "multi-batch" prize categories are derived from these tiers.
- **Correctness gate**: `eval_model.py` against `perf_public_set.jsonl`, must score ≥ 80 baseline.
- **Hard constraints learned**:
  - FP8 KV cache (`--kv-cache-dtype fp8_*`) does NOT work with the MiniCPM sparse backend.
  - `sym=True` is required by `--quantization gptq_marlin` (Marlin kernel uses `uint4b8`).
  - `--disable-cuda-graph` is for diagnosing startup errors only — baseline runs WITH CUDA graph.
  - **For 4-bit symmetric quant with uint4b8: `scale = max(abs(w)) / (2^(bits-1) - 1)` = `/ 7`, NOT `/ 8`.** Using `/ 8` clamps the top positive bin and was the 2026-05-19 acc=42.51 bug.

## How to submit a queued (📦) tarball next

Steps when current ⏳ result comes back:

1. Update its row's Status / final_score in this table.
2. If 🟢, pick the next 📦 in the v2 → v3 → v4 order and upload.
3. If 🔴, examine the failure mode:
   - Startup error → check log, fix script, build new variant.
   - Correctness < 80 → RTN may be too lossy; consider switching to GPTQModel (proper Hessian-based GPTQ).
   - Score worse than v1 → flag may not help SALA workload; skip remaining variants in this branch.

## Build a tarball from a source folder

```bash
variant="v2_chunk32k"   # or v3_mixedchunk / v4_aggressive
src="submission_rtn_sym_${variant}"
tarball="soar_rtn_sym_w4a16_marlin_${variant}_submission_$(date +%Y%m%d).tar.gz"
(cd "${src}" && tar czf "../${tarball}" .)
```

The `sglang/` subdir inside each `submission_*/` is regenerated by copying `python/` (see `cp -r submission_rtn_sym_w4a16 submission_rtn_sym_<new>` for new variants).
