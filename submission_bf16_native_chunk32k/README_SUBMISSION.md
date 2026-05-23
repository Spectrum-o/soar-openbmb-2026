# SOAR BF16 Native + Chunked-Prefill 32K (v5h)

> Built 2026-05-23 08:15. Submit ONLY IF v5g returns acc_ori jumping above
> ~60 (i.e., the fp16 conversion hypothesis is at least partially confirmed).

## Diff vs v5g

**Single-line change** in `prepare_env.sh:347`:

```diff
- export SGLANG_SERVER_ARGS="... --chunked-prefill-size 8192 ... --dtype bfloat16"
+ export SGLANG_SERVER_ARGS="... --chunked-prefill-size 32768 --max-prefill-tokens 32768 ... --dtype bfloat16"
```

Everything else identical to v5g:
- No fp16 sed-patch
- `--dtype bfloat16`
- No quantization
- auto_map.AutoConfig stripped via prepare_model.sh

## Submission gate

| v5g platform result | v5h decision |
|---|---|
| acc_ori ≥ 80 (gate cleared, non-zero final_score) | **Submit v5h** — locks in throughput on top of cleared gate |
| acc_ori 60-79 | **Submit v5h** — chunked-prefill might tip a marginal score |
| acc_ori ≈ 47 (no jump, fp16 wasn't the killer) | **DO NOT submit** — model itself caps, need different strategy |
| crashes | DO NOT submit — diagnose v5g first |

## Why this is the natural next step

`memory/project_chunked_prefill_win.md`: chunked-prefill 8K → 32K on SALA
gives +59% throughput / -60% TTFT locally. If v5g passes correctness gate,
v5h's bump should compound the throughput win.
