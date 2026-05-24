# Evening 2026-05-24 GPU session — one-line execution guide

> Written 2026-05-24 evening as user gets GPU access. Follows after
> v5j_dtype_bf16 platform success (acc_ori 82.18, final_score 23.18)
> and chunk65k OOM lesson.

## TL;DR — what to do RIGHT NOW

```bash
# 1. SSH to AutoDL
cd /root/autodl-tmp/zyn/sglang   # or wherever the repo is
git pull origin quant/w4a16
source sglang_minicpm_sala_env/bin/activate

# 2. Kick off master smoke (4-5h serial)
nohup bash scripts/gpu_smoke_all_pending.sh > /tmp/master_smoke.log 2>&1 &
echo "master PID: $!" | tee /tmp/master_smoke.pid

# 3. Monitor (optional, can disconnect SSH after this)
tail -f /tmp/master_smoke.log
# Or watch GPU memory peaks:
watch -n 5 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv'
```

After 4-5h:
- Results pushed to `origin/quant/w4a16` as intermediate commits + final SUMMARY commit
- NAS dir `/root/autodl-fs/zyn/logs/gpu_session_evening_<TS>/SUMMARY.md` has the verdict table

## What gets smoked (5 variants, in priority order)

| # | Variant | Smoke purpose | Expected P(OK) |
|---|---|---|---|
| 1 | `chunk32k_safe` | Validate `mem-frac 0.70 + 32K chunked-prefill` combo | 90% |
| 2 | `chunk32k_fp8kv` | Validate FP8 KV Path B patch + uncalibrated scale=1.0 | 50% |
| 3 | `lightning_skip_bf16` | Validate overlay tool × bf16 path | 75% |
| 4 | `chunk32k_opfusion` | Validate GPU op-fusion numerical correctness | 70% |
| 5 | `chunk65k_safe` | Validate `mem-frac 0.70 + 65K` (the chunk65k OOM fix) | 80% |

Total wall time ~4-5h serial.

## How to interpret the SUMMARY.md after smokes finish

| Status | Meaning | Action |
|---|---|---|
| `OK` | acc ≥ 78 + clean run | submit to platform with confidence |
| `OK_acc_low` | acc 60-78 | acc dropped vs v5j_dtype_bf16 (82); investigate why, maybe still submit |
| `acc_BAD` | acc < 60 | DO NOT submit — quant artifact is wrong |
| `CRASH_dtype` | RuntimeError dtype mismatch | DO NOT submit — backend incompatibility |
| `CRASH_oom` | CUDA OOM | DO NOT submit — mem budget still too aggressive |
| `CRASH_assert` | FlashAttention or assertion fail | DO NOT submit — config rejected by kernel |
| `CRASH_other` | Other traceback | check the .log file in the NAS dir |
| `UNKNOWN` | no acc + no error | check log manually |

## Decision tree based on SUMMARY

### Scenario A: chunk32k_safe passes (OK) + fp8kv passes (OK)

**Best outcome.** Submit chunk32k_fp8kv tomorrow for biggest jump.
- Expected platform: final_score 28-38
- Backup: chunk32k_safe (already known +2-5 vs current)

### Scenario B: chunk32k_safe passes, fp8kv crashes

**Most likely.** Submit chunk32k_safe — locks in known +2-5.
- Then build Path C (calibrated KV scales) as follow-up GPU work for fp8kv

### Scenario C: chunk32k_safe passes, lightning_skip_bf16 passes with acc ≥ 84

**Surprise win.** Submit lightning_skip_bf16 — proves architectural acc improvement.
- Stack with chunk32k follow-up

### Scenario D: chunk32k_safe crashes

**Worst case for the night.** Investigate why 0.70 + 32K still doesn't work.
- Possible cause: actually a backend issue, not just mem-frac
- Submit nothing tomorrow; debug + repack first

## What's still NOT covered

- **op-fusion variant smoke ALONE doesn't tell you platform behavior.** It tests
  GPU numerical correctness only. If smoke acc is good, still chance platform
  has different GPU/CUDA combo that breaks fusion.
- **fp8kv passing smoke doesn't mean all platform workloads work.** The
  hidden eval set may have prompts that push KV magnitudes outside fp8_e4m3
  range. If smoke acc is borderline (78-80), still risky.
- **chunk65k_safe is at the END of the queue.** If chunk32k_safe fails, we
  shouldn't bother with chunk65k_safe (same root cause).

## Estimated GPU time per variant

```
prepare_env:     5 min   (pip install, sed-patches if any, fp8kv patch tool)
prepare_model:   10-15 min (W4A16 quant via gptqmodel)
server startup:  3 min
30-sample eval:  25-30 min
total:           ~50 min per variant
```

5 variants ≈ **4-5 hours total**. Run via `nohup` so SSH disconnect doesn't
kill it.

## Recovery if something goes wrong

If master script gets killed (instance reboot, OOM-killer, etc.):
```bash
# Find where it stopped
ls -la /root/autodl-fs/zyn/logs/gpu_session_evening_*/
# Resume from the next variant in VARIANTS array
# Or just re-run from scratch; smokes are idempotent (re-quant happens)
```

If a specific variant smoke fails repeatedly:
```bash
# Look at its log:
cat /root/autodl-fs/zyn/logs/gpu_session_evening_*/lightning_skip_bf16.log | tail -100

# Common fixes:
# - "command not found": missing env or pip install failed in prepare_env
# - "RuntimeError dtype": variant config has --dtype float16 by mistake; recheck
# - "OutOfMemoryError": even safer mem-frac needed (try 0.60)
```

## After smoke results land

Check `SUMMARY.md` (will be in commit log + NAS dir). Based on results:
1. Submit best variant via OneDrive Mac → SOAR platform
2. Build follow-up variant for next slot (e.g. chunk32k_safe_fp8kv if both passed)
3. Update SUBMISSIONS.md with platform result row

## Reference

- v5j_dtype_bf16 platform: acc_ori 82.18, final_score 23.18 (current best)
- chunk65k FAILED: OOM mid-eval at 09:20 (taught us mem-frac 0.80 + 32K+ too aggressive)
- 84GB platform is RTX 6000D (NOT 96GB Blackwell like AutoDL)
- Path B FP8 KV patch is from notes/fp8-kv-cache-investigation (2026-05-14)
