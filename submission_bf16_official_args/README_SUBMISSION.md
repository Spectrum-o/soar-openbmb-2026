# SOAR BF16 + Official Serving Args (control for W4A16 migration)

> Built 2026-05-23 11:30 as **Experiment 1 (control)** for tonight's W4A16
> migration validation. Single-variable test: v5g + official `run_sala.sh`
> serving args.

## Purpose

v5g (BF16 native, chunked-prefill 8192) returned `acc_ori=83.04` on the SOAR
platform. The official `run_sala.sh` BF16 baseline launcher uses:
- `--chunked-prefill-size 65536`
- `--max-prefill-tokens 65536`
- `--max-running-requests 32`
- `--mem-fraction-static 0.80`

We never tested those args at platform-scale. This variant tests whether
those args preserve acc when applied to v5g's bf16-native config.

**Why this matters before W4A16**: the official W4A16 path (`run_sala_w4a16.sh`)
uses the SAME 4 args. If they break acc on bf16, they'll break it on W4A16 too.
Better to isolate the serving-args variable here, on a model we already trust.

## Diff vs submission_bf16_native_bf16 (v5g)

**Single change** in `prepare_env.sh`:
```diff
- export SGLANG_SERVER_ARGS="... --chunked-prefill-size 8192 ... --dtype bfloat16"
+ export SGLANG_SERVER_ARGS="... --chunked-prefill-size 65536 --max-prefill-tokens 65536 --max-running-requests 32 --mem-fraction-static 0.80 ... --dtype bfloat16"
```

Everything else byte-identical to v5g:
- No fp16 sed-patch (the v5g breakthrough)
- `--dtype bfloat16`
- transformers 4.57.1 pin + hub<1.0 cascade
- bundled flash_attn wheel
- bundled SGLang
- prepare_model.sh strips auto_map.AutoConfig (preserves v5d fix)

## Local validation matrix (tonight)

| Local acc_ori | Decision |
|---|---|
| ≈ 83 | Official serving args SAFE → use on W4A16 |
| 75-82 | Marginal — investigate which arg(s) cost acc |
| < 75 | Don't compound on W4A16 — submit v5h (32K) instead |

## When NOT to submit this directly to platform

This is intended as a **local control**. The actual platform submission target
is the W4A16 version (or v5h as fallback). Submit this only if:
- W4A16 local acc fails the gate AND
- BF16 + official args local acc clears with margin
- (i.e. the answer is "official args are safe, but W4A16 quant isn't")
