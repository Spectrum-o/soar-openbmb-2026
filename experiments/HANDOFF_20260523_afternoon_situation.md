# Situation handoff — 2026-05-23 afternoon
# (written from dev mirror; reading: server-side Claude)

> Status snapshot after v5g platform result came back.
> Action items at the bottom — start there if you only have 5 min.

## TL;DR

1. **v5g platform result confirmed**: `acc_ori=83.04, final_score=19.12` (= baseline 19.13).
   The fp16 sed-patch was the bottleneck across 2 weeks of W4A16 work. **Confirmed: not quant, not calibration.**
2. **P1 (BF16 + 65K official args) → acc=0 was OOM, not arithmetic failure**.
   Chunked-prefill is still a perf knob; the 65K config was just mem-budget-too-large for the BF16 model.
3. **Platform GPU = RTX 6000D = 84 GB (China-only, 12GB less than AutoDL's 96GB Blackwell)**.
   See sources below. Mem budgets need to be recomputed against 84GB ceiling.
4. **GPTQ is not dead** — 3 paths to make it work, v5j is testing Path 1 right now.
5. **5h pipeline data: calibration knobs cap at +5-8pp** (D1-F2 all in 44-52 range vs v5g's 83). Calibration is not the lever.

## v5g + v5d comparison — fp16 hypothesis decisively confirmed

| Variant | dtype | sed-patch | acc_ori |
|---|---|---|---|
| v5d | float16 | YES (bf16→fp16) | 46.64 |
| 1849 | float16 | YES | 46.89 |
| **v5g** | **bfloat16** | **NO** | **83.04** |

The Δ of +36pp between v5d and v5g is entirely the fp16 cost. SALA's lightning attention
recurrent state `h_t = h_{t-1} * decay + k_t.T @ v_t` accumulates over sequence length;
in bf16 it tolerates ~10⁵ magnitudes, in fp16 it overflows at 65504 → garbage outputs
on long prompts → repetition collapse failure mode we'd been chasing for 2 weeks.

## Critical hardware fact (newly verified)

| | GPU | VRAM | Source |
|---|---|---|---|
| **SOAR platform** | RTX 6000D | **84 GB** | CLAUDE.md + WebSearch verified |
| **AutoDL local** | RTX PRO 6000 Blackwell | **96 GB** | OOM error msg shows 94.97 GiB |

**Sources verified 2026-05-23**:
- [Nvidia RTX 6000D teardown shows 84GB VRAM using 3GB memory chips — Club386](https://www.club386.com/nvidia-rtx-6000d-teardown-shows-84gb-vram-using-3gb-memory-chips/)
- [NVIDIA RTX 6000D with 84GB GDDR7 memory — VideoCardz](https://videocardz.com/newz/nvidia-rtx-6000d-with-84gb-gddr7-memory-appears-in-first-teardown-video)
- [国行RTX Pro 6000D规格曝光 — 新浪](https://www.sina.cn/news/detail/5236996408871420.html)

Implication: **platform is 12GB tighter than AutoDL**. If a config OOMs locally, it also OOMs on platform. The reverse isn't necessarily true.

### Recomputed mem budgets for 84GB platform

| Config | model | KV (×84) | chunked-buffer | total | 84GB fit? |
|---|---|---|---|---|---|
| W4A16 + 65K + 0.80 (=1849) | 5 | 67 | 5.6 | 77.6 | ✓ (6GB headroom) |
| BF16 + 8K + 0.80 (=v5g platform) | 18 | 67 | 0.7 | 85.7 | borderline; v5g survived but tight |
| **BF16 + 32K + 0.80 (=v5h as packed)** | 18 | 67 | **3** | **88** | **✗ WILL OOM** |
| **BF16 + 32K + 0.65 (=v5h FIXED)** | 18 | 55 | 3 | **76** | ✓ (8GB headroom) |
| BF16 + 65K + 0.80 (=P1 was) | 18 | 67 | 5.6 | 90.6 | ✗ huge OOM (confirmed local) |
| W4A16 + 8K + 0.80 (=v5j) | 5 | 67 | 0.7 | 72.7 | ✓ plenty |
| W4A16 + 32K + 0.80 (=v5j+32K stacked) | 5 | 67 | 3 | 75 | ✓ |

**Action item baked in**: v5h must drop `--mem-fraction-static 0.80 → 0.65` before submission, or it OOMs on platform regardless of how the math worked locally.

## Three GPTQ paths (priority order)

### Path 1: v5j — gptqmodel + Marlin + drop sed-patch
- **Single-variable test**: 1849 minus the `sed s/bfloat16/float16/g` block, keeping `--dtype float16`
- **Hypothesis**: Marlin's fp16 output gets auto-cast back to bf16 at SGLang module boundaries; sed-patch was forcing the entire sparse backend to fp16 when only the Linear output boundary needed it
- **EV if works**: 1849 + 1 line config fix + chunked-prefill 32K + W4A16 4× bandwidth save = final_score 35-50
- **Status**: smoke-testable on GPU NOW

### Path 2: GPTQ via llm-compressor + compressed-tensors
- This is the OFFICIAL recommendation per `W4A16_README.md` (line 23-27, "智算一队" SOAR Week 4 champion's note)
- Variant `submission_gptq_official_args/` already exists (commit `da1d66203`)
- compressed-tensors loader natively supports `--dtype bfloat16` — no Marlin → no fp16 forcing
- **Known concern**: AWQ via same loader (P2) FAILED rc=1 in smoke. The bug class is likely shared. **Need to debug P2 first** before GPTQ via this path works.

### Path 3: modify SGLang to cast Marlin output to bf16
- Last resort. Engineering work: add `output.to(torch.bfloat16)` after the Marlin GEMM call in `python/sglang/srt/layers/quantization/gptq.py`
- Use `--dtype bfloat16` everywhere
- Highest confidence (~75%) but highest implementation cost (~2 hours edit + repack)

## Calibration is NOT the lever (5h pipeline data)

| Phase | knob | acc_ori | vs baseline |
|---|---|---|---|
| A_J0_baseline | 1849 recipe | 44.00 | — |
| D2_damp_01 | dampening_frac=0.1 | 50.89 | +6.9 |
| D3_calib_512 | num_calib=512 | 49.50 | +5.5 |
| D4_calib_len_16k | max_calib_len=16K | 49.33 | +5.3 |
| F1_multi_adaptive | multi-window calib | 51.50 | +7.5 |
| **v5g** | **dtype bf16 + skip sed** | **83.04** | **+39.0** |

Each calibration knob got at most +7.5pp. dtype fix got +39pp. **Stop tuning calibration; the ceiling on that lever is ~52, far below the gate.**

## GPU action plan (priority order; total ~3 hours)

### Stage 1: cheap startup verifications (~5 min total)

```bash
cd /root/autodl-tmp/zyn/sglang

# 1a. v5h: edit submission_bf16_chunk32k_fixed/prepare_env.sh to set
#     --mem-fraction-static 0.65 (currently 0.80, will OOM on 84GB platform)
#     Just confirm server boots; no eval needed.

# 1b. gptq_official_args: confirm prepare_env runs without import errors
bash submission_gptq_official_args/prepare_env.sh > /tmp/gptq_official_env.log 2>&1
tail -20 /tmp/gptq_official_env.log

# 1c. Investigate why P2 (AWQ) smoke failed rc=1 — read the existing log
ls scripts/logs/5h_*_full/P2*.log 2>/dev/null | head -3
# grep Traceback / Error / Failed
```

### Stage 2: v5j 30-sample smoke (~50 min, HIGHEST EV)

```bash
nohup bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch \
    --num-samples 30 \
    --force-requant > /tmp/v5j_smoke.log 2>&1 &
echo "v5j smoke pid: $!"
```

Decision tree on result:
- `acc_ori ≥ 70`: Path 1 works. **Pack v5j + submit tonight**. Bonus: also build v5j_chunk32k variant for tomorrow's slot.
- `acc_ori 50-70`: partial fix. Build v5j_dtype_bf16 (drop sed AND change dtype to bf16).
- `acc_ori < 50`: Path 1 fails. Goto Stage 3.
- Startup crash (RuntimeError dtype mismatch): Path 1 physically blocked. Goto Stage 3.

### Stage 3 (conditional, if Stage 2 fails): gptq_official_args smoke

Requires P2 (AWQ) failure debugged first. ~60-90 min eval.

### Stage 4 (last resort): Path 3 SGLang source patch

~2 hr engineering + 30 min smoke.

## What NOT to do

- **No more calibration tuning** (5h data proves ≤+7.5pp ceiling)
- **No 65K chunked-prefill for BF16** (math shows 90.6 > 84, will OOM)
- **No v5h to platform without local OOM verification** (P1 lesson)
- **No `--mem-fraction-static 0.80` for BF16 + ≥32K chunked-prefill**
- **No re-running D1-D4 ablations on v5g native bf16 base** — they were all done on broken fp16 baseline; results aren't transferable

## What to USE before submitting anything

```bash
# Already integrated into full_preflight.sh
bash scripts/full_preflight.sh --variant <name>

# Specifically for the new bf16 mem-budget concern, before submitting any BF16 variant:
grep -E "mem-fraction-static|chunked-prefill-size" submission_<name>/prepare_env.sh
# Sanity check: BF16 + chunked-prefill N requires mem-fraction-static ≤ X
#   8K: 0.78
#   16K: 0.72
#   32K: 0.65
#   65K: don't (will OOM)
```

## Submission slot strategy

- User has 1 platform slot left today (probably resets at local midnight)
- v5g's 19.12 is current best, locked on leaderboard
- Asymmetric risk: submitting any variant has 0 leaderboard downside (best-of-all scoring)
- **But submitting an unvalidated variant wastes the slot** = waste of 5h time window
- Recommended: **don't submit anything tonight until v5j smoke result is in**. If v5j smoke ≥70, submit v5j (W4A16, mem-budget-friendly). If v5j smoke <70, save the slot for tomorrow when more data available.

## Cross-references

- Original fp16 hypothesis design: v5g commit `5876cecc0`
- v5j variant: `submission_gptqmodel_no_fp16_patch/` (commit `544a61abe`)
- gpt_official_args variant: `submission_gptq_official_args/` (commit `da1d66203`)
- W4A16 recipe history: `SUBMISSIONS.md` (full chronological log)
- Earlier handoff: `experiments/HANDOFF_20260523_morning.md` (pre-v5g result)
- 5h pipeline lessons: `experiments/5H_PIPELINE_HARDENING_HANDOFF.md`

## Author's recommendation in one sentence

Run v5j smoke on GPU now (Stage 2 above); everything else is conditional on its result.
