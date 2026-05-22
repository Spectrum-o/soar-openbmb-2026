# SOAR AWQ + Lightning-Skip Combo Submission (moonshot)

> Built 2026-05-23 03:00 (autonomous night work). The MOONSHOT variant:
> stacks the two highest-EV independent bets from tonight's analysis.

## Why this exists

The research synthesis (`experiments/QUANT_STRATEGY_RESEARCH_20260522.md`)
identified two independent paths beyond 1849's GPTQ baseline:

1. **AWQ** — activation-aware quantization. OpenBMB ships AWQ (not GPTQ)
   for MiniCPM-V 4.5. SALA's `scale_emb=12 + scale_depth=1.4` create
   exactly the outlier pattern AWQ targets. Expected: +5-10pp aggregate.

2. **Lightning-skip mixed precision** — SALA's lightning attention
   recurrence accumulates quant errors across time steps. Keeping
   lightning-adjacent MLPs in BF16 should break the cwe/niah repetition
   collapse. Expected: +5-10pp on long-context tasks specifically.

These two are ORTHOGONAL — AWQ is "different quant algorithm", lightning-skip
is "which layers to quantize". They can stack.

## Combined hypothesis

- 1849 baseline: acc_ori=46.89 (platform), ≈49 (local)
- + AWQ alone: ~52-56 (research-backed)
- + lightning-skip alone: ~52-57 (architectural hypothesis)
- + BOTH: ~58-66 (if effects add, with diminishing returns)
- Best case: **66-70 raw acc** — still ~10pp below the 80% gate, but the
  closest we've gotten

## How it works

Two-step prepare_model:
1. `quantize_llmcompressor_awq.py` (from `submission_awq_llmcompressor/`):
   - llm-compressor AWQ recipe (W4A16_ASYM, group_size=128)
   - calibration: perf_public_set, left-truncate, 256 × 8192
   - post-process: H4 tokenizer overwrite, auto_map.AutoConfig strip,
     compressed-tensors quantization_config block
   - output format: compressed-tensors / pack-quantized

2. `apply_lightning_skip_overlay.py` (extended to handle compressed-tensors):
   - reads `mixer_types` from config → identify lightning indices
   - reads original BF16 MLP weights for those layers
   - writes `model-lightning-skip-overlay.safetensors`
   - removes `.weight_packed/.weight_scale/.weight_zero_point` (compressed-tensors fmt)
     AND `.qweight/.qzeros/.scales` (gptq_marlin fmt, as defensive backup)
     for lightning MLPs from the index
   - updates `quantize_config.json` dynamic to skip lightning MLPs

## Format compatibility

`apply_lightning_skip_overlay.py` recognizes BOTH:
- **gptq_marlin** suffixes: `.qweight`, `.qzeros`, `.scales`, `.g_idx`
- **compressed-tensors** suffixes: `.weight_packed`, `.weight_scale`,
  `.weight_zero_point`, `.weight_shape`, `.weight_g_idx`

When working on a compressed-tensors AWQ artifact, only the compressed-tensors
suffixes will actually exist in the index; gptq_marlin suffixes are no-ops.
And vice versa.

## SGLang loader path

```
--quantization compressed-tensors --dtype bfloat16
```

Same as `submission_awq_llmcompressor`. The lightning-skip overlay
modifications are transparent to SGLang — it just sees the `dynamic` skip
rules and routes the lightning MLPs through `UnquantizedLinearMethod`
which finds the `.weight` tensors in the overlay shard.

## Risks (this is the LEAST tested variant)

| Risk | Severity | Mitigation |
|---|---|---|
| AWQ alone might not work on this platform | High | individual `submission_awq_llmcompressor` variant exists; test that first |
| Lightning-skip overlay logic for compressed-tensors hasn't been tested | High | individual `submission_w4a16_lightning_skip` (gptq-marlin format) exists; test that first |
| The combination introduces unknown interactions | Medium | each step is logged independently in prepare_model.sh output |
| AWQ + skip-lightning = quantize only dense layers' MLPs = ~25% of params | Medium | acceptable; cw/niah are the bottleneck not throughput |

## DO NOT SUBMIT UNTIL

1. `submission_awq_llmcompressor` has been validated on GPU (does AWQ
   output load + serve correctly?)
2. `submission_w4a16_lightning_skip` has been validated on GPU (does the
   gptq-marlin-format overlay logic work?)
3. Local `bash scripts/local_eval.sh --variant submission_awq_lightning_skip
   --num-samples 30 --force-requant` succeeds without crash

If both individual pieces work AND the combo loads + serves locally, then
this is the strongest candidate for the highest acc score this project
can produce.

## Tunables (same as `submission_awq_llmcompressor`)

```
CALIB_JSONL, NUM_CALIB, MAX_CALIB_LEN, AWQ_SCHEME, MLP_ONLY=1
```

## Hard Constraints

```bash
python3 tools/hard_constraints_lint.py --variant submission_awq_lightning_skip
```

Expected: 11 PASS / 0 FAIL (llm_compressor mode constraints).
