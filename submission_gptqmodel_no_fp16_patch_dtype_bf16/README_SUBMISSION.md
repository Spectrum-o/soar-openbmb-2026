# v5j_dtype_bf16 — single-variable test of `--dtype bfloat16` on Marlin path

> Conditional submission. Build only if v5j returns acc_ori in [50, 70].

## What this variant tests

The single-variable test that v5j missed.

Recap of what we know:
- **v5d** (BF16 model + sed-patch + `--dtype float16`): acc_ori 46.64
- **v5g** (BF16 model + NO sed-patch + `--dtype bfloat16`): acc_ori **83.04**
- **v5j** (W4A16 model + NO sed-patch + `--dtype float16`): smoke pending

v5g flipped TWO things at once (sed-patch + dtype). We don't know if EITHER
alone is sufficient, or if BOTH are needed.

v5j removes only the sed-patch (Marlin path, keeps fp16 dtype).

**This variant (v5j_dtype_bf16) removes only the dtype** (Marlin path, keeps
no-sed-patch). It tests: does Marlin's internal fp16 GEMM + downstream bf16
sparse backend (via SGLang cast) work as well as full bf16?

## Diff vs v5j

A single character change in `prepare_env.sh`:

```diff
-export SGLANG_SERVER_ARGS="... --dtype float16"
+export SGLANG_SERVER_ARGS="... --dtype bfloat16"
```

Everything else is symlinked to v5j (same quant script, same calibration,
same flash_attn wheel, same SGLang bundle, same prepare_model.sh).

## Decision tree (after v5j smoke)

| v5j result | Action on this variant |
|---|---|
| acc_ori ≥ 80 | This variant is REDUNDANT (v5j already wins). Don't pack. |
| acc_ori 50-70 | **Build and smoke this**. Tests if `--dtype bfloat16` is the missing piece. |
| acc_ori < 50 | This variant likely also fails (sed-patch alone wasn't enough). Skip. |
| v5j crashes RuntimeError dtype | This variant might WORK where v5j didn't (bf16 dtype could fix the boundary mismatch). Worth a smoke. |

## Mem budget check (84GB RTX 6000D platform)

Same as v5j: 5GB model + 0.80×84 KV (67GB) + 0.7GB buffer = 72.7GB ≤ 84GB.
11GB headroom. No OOM risk.

## Local smoke command

```bash
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16 \
    --num-samples 30 --force-requant 2>&1 | tee /tmp/v5j_bf16_smoke.log
```

## Pack command

```bash
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_no_fp16_patch_dtype_bf16 \
    --suffix _v5j_dtype_bf16 \
    --output-dir .
```
