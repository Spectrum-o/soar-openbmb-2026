# SOAR BF16 + Chunk32K + Op-Fusion (v6) — top of the stack

> Built 2026-05-23 alongside v5d/v5e. Submits ONLY after v5e is verified
> working. v6 = v5e + op-fusion overlay on minicpm.py.

## What v6 stacks on top of v5e

1. **chunked-prefill 32K + max-prefill-tokens 32K** (from v5e)
2. **auto_map.AutoConfig stripped from config.json** (from v5d, the v3/v5/v5c root-cause fix)
3. **Full 1849-style env bundle** (transformers 4.57.1 pin, flash_attn, bundled SGLang)
4. **NEW: op-fusion patch on `sglang/srt/models/minicpm.py`** (from `origin/perf/op-fusion` HEAD)

## Op-fusion specifics

From SOAR champion notes 笔记 04 (智算一队):
1. Drop `float()` upcast around `rotary_emb` in MiniCPMAttention and MiniCPMLightningMixer (apply_rope already runs fp32 internally — the upcast was redundant)
2. Adopt residual-delay pattern so RMSNorm dispatches through SGLang's fused `add_rmsnorm` kernel (saves 2 kernel launches per decoder layer × 32 layers per decode step)

## How the overlay is applied

`prepare_env.sh` after sglang editable install:

```bash
cp overlay/minicpm.py ${SUBMISSION_DIR}/sglang/python/sglang/srt/models/minicpm.py
```

The editable install means Python reads source files from the
SUBMISSION_DIR location, so the `cp` is effective immediately. An import
smoke test verifies the overlay file is valid before SGLang launches.

## Provenance + risks

- Source commit: `origin/perf/op-fusion` HEAD (`87c22ec88 + 79b0c20f4`)
- Pure-torch CPU equivalence: **verified** by commit 87c22ec88 unit test
- GPU end-to-end correctness vs MiniCPM-SALA sparse kernels: **NOT verified**
- If acc drops > 2pp vs v5e, op-fusion is numerically wrong on this GPU+CUDA
  combo — fall back to v5e

## Submission gate

Submit v6 ONLY IF v5e returns a clean ~28-30 final_score. If v5e itself fails
(chunked-prefill 32K incompatible) or returns low acc, v6 inherits both
risks and we have nothing to gain.

## Expected outcome

| Result | final_score | Action |
|---|---|---|
| ✅ Works, acc preserved | **30-35** (op-fusion adds ~5-10% throughput on top of v5e) | best result of today, rank 18-20 |
| ✅ Loads but acc drops | low | op-fusion numerical issue → fall back to v5e |
| ⛔ Crashes | 0 | overlay broke something → fall back to v5e |
