# Evening 2026-05-25 GPU session — runbook

> Written 2026-05-25 ~12:00 after opfusion_v1 platform failure
> (acc_ori=24.67, final_score=0.0). Successor to
> EVENING_2026_05_24_RUNBOOK.md.

## Anchor: where we stand on the platform

| Variant | acc_ori | final_score | S1 | Status |
|---|---|---|---|---|
| v5j_dtype_bf16 (chunk 8K) | 82.18 | 23.18 | 717.76 | ✅ |
| chunk32k_safe (mem-frac 0.70) | 80.31 | 22.90 | 720.76 | ✅ current best safe baseline |
| chunk32k_opfusion_v1 (mem-frac 0.80) | **24.67** | **0.0** | 707.66 | 🔴 broken residual overlay |

The 5/24 night queue (`gpu_smoke_all_pending.sh`) is **stale** — chunk32k_safe is now platform-verified, chunk65k_safe is redundant, chunk32k_fp8kv was dropped (commit `e7a962905`), and opfusion_v1's overlay has been rewritten.

## Why opfusion_v1 failed (already fixed, captured here for context)

`overlay/minicpm.py` v1 swapped SALA's residual path to the SGLang "fused add_rmsnorm" pattern. That pattern works for Llama-class models but invalidates SALA's `residual + hidden_states * scale_depth/√L` rescaling: with `scale_depth=1.4 / √32 ≈ 0.247` applied per sublayer, the fused version pushes scale_depth inside `add_rmsnorm` and lets residual accumulate across layers instead of resetting. CPU equivalence test (`87c22ec88`) only covered single-layer toy input; 32-layer × bf16 × long-context accumulation was unverified.

GPU smoke (30 samples) showed acc_ori=82.67 — false-positive because short-output tasks (fwe / option-letter MCQ) are insensitive to per-token drift. Platform's 150-sample set has more cwe/niah/long-QA → cumulative drift surfaces as repetition or wandering output.

## Priority 1 — smoke opfusion_v2 (the rewritten variant)

**Variant**: `submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion`

**Diff vs the v1 that failed**:
1. `overlay/minicpm.py` — only removes the RoPE upcast (6 lines: `orig_dtype = q.dtype`, `q, k = q.float(), k.float()`, `q, k = q.to(orig_dtype), k.to(orig_dtype)`, at 2 sites). Residual path is identical to working tree.
2. `prepare_env.sh` — `--mem-fraction-static 0.80` → `0.70` (single-variable parity with chunk32k_safe).

Old broken overlay kept at `overlay/minicpm.py.broken_residual_v1` for reference.

**Procedure**:
```bash
cd /root/autodl-tmp/zyn/sglang
git pull origin spec/eagle3-sala     # or whichever branch carries this fix
source sglang_minicpm_sala_env/bin/activate

# Step 1: preflight (cheap)
bash scripts/full_preflight.sh --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion

# Step 2: 30-sample smoke (~50 min)
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion \
    --num-samples 30
```

**Pass criteria** (ALL three must hold):
- `acc_ori ≥ 80` (parity with chunk32k_safe's 80.31). Below 78 = regression, stop.
- S1 within ±3% of chunk32k_safe's 720s (accept 700-742s).
- Server didn't crash mid-eval (no `Connection refused` in the log).

If smoke passes, pack + submit:
```bash
bash scripts/full_preflight.sh --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion --pack
# Copy to OneDrive → submit via SOAR platform
```

**Expected platform outcome**: acc_ori ~80 (matches chunk32k_safe), S1 ~700-715s, final_score ~23-24. Small but safe gain over 22.90 baseline.

**Failure modes to watch**:
- If smoke `acc_ori` drops to 60-78 unexpectedly: sparse-backend RoPE path may have hidden dtype dependency we missed. Inspect `submission_*_opfusion.log` for any RoPE-related warning.
- If smoke crashes with dtype mismatch in InfLLMv2: same SimpleGLA/Marlin boundary issue from v5j — should NOT happen with bf16 path but flag immediately.

## Priority 2 — EAGLE3 Phase 3 prep (no platform slot needed)

If opfusion_v2 smoke runs unattended, use parallel time to start EAGLE3 Phase 3 data collection:

1. Write `scripts/collect_eagle3_hidden_states.py`:
   - Driver: spin up SALA target via SGLang offline mode (or `requests.post` to a running `--port 30000` server).
   - For each prompt in `submission_*/perf_public_set.jsonl` (150 samples, long-context, on-distribution).
   - Capture `output_hidden_states` for dense layers only (not lightning — open question #4 in design doc says lightning state is recurrent and unlikely to be a clean draft signal).
   - Serialize to npz / safetensors per prompt under `/root/autodl-fs/zyn/eagle3_hidden_states/`.
   - Estimate: 150 samples × ~30K tokens × 4096 hidden × bf16 × 3 layers ≈ 45 GB. Manageable.

2. Open question for the training step (Phase 3.2): we need ~100K samples (perf_public_set's 150 is too few). Source options:
   - Public long-context QA sets (Long-RAG, LongBench).
   - Synthetic NIAH constructed locally (cheap to scale).
   - SALA's own pretraining mix if released (unlikely).
   Decision: punt to a separate research turn; for Phase 3.1 (collection script), 150 is enough to validate the pipeline.

This is foreground writing work; doesn't need GPU until you actually run the collector. Smoke can run concurrently.

## What to NOT do this session

- **Do NOT submit chunk65k variants** — superseded by chunk32k_safe + chunk32k_opfusion_v2. chunk65k_safe stays in the dir for archive only.
- **Do NOT submit chunk32k_fp8kv** — Path B patch insufficient (per commit `e7a962905`); needs Path C calibrated scales which is a multi-day project.
- **Do NOT submit lightning_skip_bf16 without smoke** — overlay tool × bf16 path is GPU-untested. If you have spare GPU time *after* opfusion_v2 + EAGLE3 collection, smoke this. Otherwise defer.
- **Do NOT touch `submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion/overlay/minicpm.py.broken_residual_v1`** — kept for post-mortem reference.

## Decision tree by smoke outcome

| opfusion_v2 smoke result | Action |
|---|---|
| acc_ori ≥ 80, S1 ≤ 720s, clean run | Pack + submit. Expected platform +0-1 final_score over chunk32k_safe. |
| acc_ori ≥ 80, S1 jump > 720s | RoPE upcast removal didn't help perf — submit anyway as parity check, then deprioritize this branch. |
| acc_ori 60-78 | Stop. RoPE path has hidden bf16↔fp32 dependency we don't understand. Open new investigation. |
| acc_ori < 60 | Stop. Revert overlay entirely, file an issue, move on to EAGLE3. |
| Crash | Read the log; if dtype mismatch → flag immediately. If OOM → mem-frac 0.70 wasn't enough, try 0.65. |

## Reference: chunk32k_safe is the anchor

Any variant that doesn't beat chunk32k_safe (22.90) by ≥ 1 final_score is **not worth the platform slot**. With ~5h per slot and limited daily slots, the bar is real.

## File pointers

- Variant dir: `submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion/`
- Broken v1 overlay (archive): `.../overlay/minicpm.py.broken_residual_v1`
- New v2 overlay: `.../overlay/minicpm.py`
- Predecessor doc: `experiments/EVENING_2026_05_24_RUNBOOK.md`
- EAGLE3 design: `experiments/EAGLE3_SALA_DESIGN.md`
- Platform log canonical: `SUBMISSIONS.md` (update with v1 + v2 results)
