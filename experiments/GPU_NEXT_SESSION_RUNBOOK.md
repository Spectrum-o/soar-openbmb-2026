# GPU next-session runbook — pre-staged 2026-05-23

> Read this when you next have GPU access. The prep work is done;
> this doc says what to actually run.

## TL;DR

```bash
# On the GPU box, freshly cloned or pulled
cd /root/autodl-tmp/zyn/sglang
git pull origin quant/w4a16
source sglang_minicpm_sala_env/bin/activate

# One command kicks off everything
bash scripts/gpu_session_next.sh

# 50 min later — v5j smoke done
bash scripts/gpu_session_decide_next.sh
```

That's the whole session. The runbook below is the why + what-comes-next.

## What's been pre-staged in this commit

| File | Purpose |
|---|---|
| `submission_bf16_native_chunk32k/prepare_env.sh` | **FIXED**: `--mem-fraction-static 0.65` added (was implicit ~0.88 → would OOM on 84GB platform). Now safe to submit. |
| `submission_gptqmodel_no_fp16_patch/prepare_env.sh` | **HARDENED**: `--mem-fraction-static 0.80` set explicitly (matches 1849; was implicit ~0.88 → working but with slim margin). |
| `submission_gptqmodel_no_fp16_patch_dtype_bf16/` | **NEW VARIANT**: v5j + `--dtype bfloat16`. For the case where v5j gives partial result (acc 50-70). Tests if dtype alone closes the gap. |
| `scripts/gpu_session_next.sh` | **NEW**: Stage 1 (verify configs) + Stage 2 (launch v5j smoke in background). NAS-logged. |
| `scripts/gpu_session_decide_next.sh` | **NEW**: Reads v5j smoke result + recommends next action. |

## The big picture

You're trying to make GPTQ work on the bf16 path (since v5g proved bf16 is
the unlock, and fp16 was the killer for 2 weeks). Three paths exist:

| Path | Variant | Tested by |
|---|---|---|
| **Path 1** | gptqmodel + Marlin + drop sed-patch + dtype=fp16 | v5j (this session's headline test) |
| **Path 1b** | same as Path 1 + dtype=bfloat16 | v5j_dtype_bf16 (conditional) |
| **Path 2** | llm-compressor + compressed-tensors + dtype=bfloat16 | gptq_official_args (already exists but P2 smoke FAILED) |
| **Path 3** | modify SGLang to cast Marlin output to bf16 | (engineering work, last resort) |

This session focuses on Path 1, with Path 1b as immediate fallback.

## Stage-by-stage walkthrough

### Stage 1 — Startup verifications (5 min)

`gpu_session_next.sh` automatically:
- Verifies v5h has the mem-fraction fix
- Verifies v5j has 0.80 mem-fraction explicit
- Verifies v5j_dtype_bf16 has bfloat16 dtype
- Greps the AWQ P2 failure log for root cause clues

This catches any merge mistake or accidental config change BEFORE wasting
GPU time on a broken variant.

### Stage 2 — v5j smoke (50 min, background)

```bash
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch \
    --num-samples 30 \
    --force-requant
```

This is the headline experiment. ~10-15 min quant + ~30-40 min eval = ~50 min total.

**While it runs, monitor**:
```bash
tail -f /root/autodl-fs/zyn/logs/gpu_session_*/v5j_smoke.log
```

**What you're looking for**:
- Process keeps running (no early crash)
- `acc_ori = XX.XX` line at the end

### Stage 3 — Decide next based on v5j result

Auto-tool: `bash scripts/gpu_session_decide_next.sh` — parses the log and tells you.

Manual decision tree if you want to read for yourself:

| v5j acc_ori | Path 1 status | Next action | Why |
|---|---|---|---|
| **≥ 70** | ✅ WORKS | **Pack v5j + submit on next slot** | Single config fix wins. Also build v5j+32K-chunked-prefill variant for follow-up slot |
| **50–70** | ⚠️ PARTIAL | **Smoke `v5j_dtype_bf16` immediately** | sed-patch is most of problem; check if bfloat16 dtype closes the gap |
| **< 50** | ❌ DEAD | **Skip to Path 2** (gptq_official_args) | sed-patch wasn't the only killer; Marlin path is structurally incompatible |
| **Crash (dtype mismatch)** | ❌ DEAD | **Skip to Path 1b or 2** | Marlin's fp16 output can't auto-cast; need explicit bf16 or different loader |
| **Crash (other)** | 🐛 BUG | **Read traceback** | Quant-prep bug, not architectural |

### Stage 4 — Conditional: depending on Stage 3

**If v5j worked** (≥70):
```bash
# Pack platform submission
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_no_fp16_patch \
    --suffix _v5j --output-dir .

# Build follow-up: v5j with 32K chunked-prefill (need to fork variant first)
# This is NOT pre-staged. Do this only after v5j success confirmed.
SRC=submission_gptqmodel_no_fp16_patch
DST=submission_gptqmodel_no_fp16_patch_chunk32k
mkdir -p "$DST"
for f in flash_attn-*.whl perf_public_set.jsonl prepare_model.sh quantize_gptqmodel_w4a16.py sglang; do
    ln -sf "../$SRC/$f" "$DST/$f"
done
cp "$SRC/prepare_env.sh" "$DST/prepare_env.sh"
cp "$SRC/README_SUBMISSION.md" "$DST/README_SUBMISSION.md" 2>/dev/null
# Edit DST/prepare_env.sh: change chunked-prefill 8192 → 32768
# Add --max-prefill-tokens 32768
# Keep mem-fraction-static 0.80 (5GB model + 32K buffer still fits comfortably)
sed -i 's|--chunked-prefill-size 8192|--chunked-prefill-size 32768 --max-prefill-tokens 32768|' "$DST/prepare_env.sh"
# Smoke 30 samples to verify chunked-prefill doesn't break W4A16 acc:
bash scripts/local_eval.sh --variant "$DST" --num-samples 30 --force-requant
```

**If v5j partial** (50-70):
```bash
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16 \
    --num-samples 30 --force-requant
# Then re-run gpu_session_decide_next.sh against the new log
```

**If v5j dead** (<50 or crash):
First debug AWQ P2 failure (Stage 1 already grepped it):
```bash
# Examine the actual failure
cat scripts/logs/5h_*_full/P2_*.log | grep -B2 -A20 -iE "Error|Traceback"

# Common causes + fixes:
# - ImportError: llmcompressor missing → bash submission_awq_llmcompressor/prepare_env.sh
# - Module 'compressed_tensors' not found → pip install compressed-tensors
# - KeyError on layer name → AWQModifier missed registering a layer; see Phase 4 lightning_skip code
```

Once AWQ fixed, gptq_official_args probably works:
```bash
bash scripts/local_eval.sh \
    --variant submission_gptq_official_args \
    --num-samples 30 --force-requant
```

## Mem budget cheat sheet for 84GB platform (RTX 6000D)

Always check before submitting:

```
W4A16 (5GB):   model + 0.80×84 KV + buffer
               5    + 67          + 0.7 (8K)  = 72.7 ✓
               5    + 67          + 3   (32K) = 75   ✓
               5    + 67          + 5.6 (65K) = 77.6 ✓ (= 1849 verified)

BF16 (18GB):   model + ?    KV + buffer
               18   + 67          + 0.7 (8K)  = 85.7 borderline (v5g works)
               18   + 67          + 3   (32K) = 88   ✗ OOM
               18   + 55          + 3   (32K) = 76   ✓ ← need mem-frac 0.65
               18   + 67          + 5.6 (65K) = 90.6 ✗ huge OOM
```

Rule: **BF16 + chunked-prefill ≥ 32K requires `--mem-fraction-static ≤ 0.65`.**

## What this session does NOT cover

- Lightning-skip variants (still need work)
- Re-running 5h calibration pipeline (already proven to be wrong lever)
- AWQ's specific debug (we just grep for clues; full fix is conditional)
- Op-fusion on bf16 path (need to rebuild on v5g base; deferred)

## If something goes wrong

- **Smoke crashes immediately**: check `scripts/logs/local_eval_quant_*.log`
  for quant-time failures vs `_server_*.log` for serve-time crashes
- **Process hangs**: `nvidia-smi` to see GPU utilization; if 0%, process is dead
- **Out of disk**: quant artifacts go to `/root/autodl-fs/zyn/models/`; clean
  old `-quantized` dirs before running new smokes if disk is tight
- **NAS not writable**: gpu_session_next.sh falls back to /tmp (warns); use
  `scp` to copy results before instance dies

## Reminder: things NOT to do

- **Don't submit v5h, v5e, v6 as currently in any platform-ready tarball**
  unless you re-verify mem-fraction is ≤ 0.65 (the old ones will OOM)
- **Don't waste another slot on bf16_official_args** (P1 already proved OOM)
- **Don't smoke calibration knobs again** (5h data: max +5-8pp, not worth it)
- **Don't pack moonshot** (AWQ + lightning_skip stacked) — needs both
  components GPU-validated first

## Reference

- Hardware: RTX 6000D = 84GB (platform) vs RTX PRO 6000 Blackwell = 96GB (AutoDL)
- v5g platform result: acc_ori 83.04, final_score 19.12 (= baseline 19.13)
- Previous handoff: `experiments/HANDOFF_20260523_afternoon_situation.md`
- 5h pipeline lessons: `experiments/5H_PIPELINE_HARDENING_HANDOFF.md`
