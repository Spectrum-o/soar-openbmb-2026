# SOAR AWQ via llm-compressor Submission

> Built 2026-05-23 (night). The strongest a-priori bet from the research-agent
> synthesis (`experiments/QUANT_STRATEGY_RESEARCH_20260522.md`):
> **OpenBMB ships AWQ (not GPTQ) for MiniCPM-V**, and MiniCPM-SALA's
> `scale_emb=12` + `scale_depth=1.4` create exactly the kind of activation
> outliers AWQ was designed for.

## What this variant does

Replaces the GPTQModel pipeline with llm-compressor + AWQModifier:

```python
AWQModifier(ignore=[...self_attn... + lm_head])
QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=[...])
oneshot(model, processor=tokenizer, dataset=calib, recipe, save_compressed=True)
```

Output is **compressed-tensors / pack-quantized** format. SGLang loads it via:
```
--quantization compressed-tensors --dtype bfloat16
```
(NOT `gptq_marlin`, NOT `--dtype float16`.)

## Differences from 1849 GPTQ pipeline

| Knob | 1849 (GPTQ) | **v_awq (llm-compressor)** |
|---|---|---|
| Algorithm | GPTQ (Hessian-based) | **AWQ (activation-aware)** |
| Tool | gptqmodel 7.0.0 | **llm-compressor >= 0.7** |
| Output format | gptq_marlin (uint4b8) | **compressed-tensors / pack-quantized** |
| SGLang loader | `gptq_marlin` | **`compressed-tensors`** |
| dtype at load | `float16` (Marlin requires) | **`bfloat16`** (native) |
| Symmetric vs Asymmetric | sym=True (uint4b8) | **W4A16_ASYM** (production standard) |
| qzeros fix needed? | Yes (0x77 → 0x88) | **No** (different layout) |
| H4 tokenizer overwrite | Yes | **Yes** (llmcompressor.save also re-serializes) |
| Module set | MLP-only (registered hack) | **MLP-only via ignore patterns** (cleaner) |
| Calibration | perf_public_set, left-trunc, 256×8192 | **Same** |
| transformers pin | 4.57.1 | **4.57.1** (same H5 stack) |
| flash_attn bundled | Yes | **Yes** (same wheel) |
| auto_map.AutoConfig stripped | Yes (row 44) | **Yes** (same — verified by linter) |

## Hard Constraints (verified by `tools/hard_constraints_lint.py`)

All 15 constraints PASS for this variant (run `python3 tools/hard_constraints_lint.py --variant submission_awq_llmcompressor` to verify).

Key items handled in `quantize_llmcompressor_awq.py`:
- **H4 (row 44)**: `copy_runtime_assets()` overwrites tokenizer files from base BF16 unconditionally
- **Row 44**: `write_sglang_compatible_config()` strips `auto_map.AutoConfig`
- **Row 45**: same function strips `has_sparse_attention` and other derived properties
- **H5**: `prepare_env.sh` force-reinstalls transformers==4.57.1 with hub<1.0 cascade

## Expected outcome

Per research synthesis (`experiments/QUANT_STRATEGY_RESEARCH_20260522.md`):
- AWQ typically +1-3pp over GPTQ on instruction-tuned models
- On models with activation outliers (SALA): +5-10pp typical
- OpenBMB official AWQ recipe for MiniCPM-V exists → suggests AWQ is genuinely
  suited to this model family

If 1849 (GPTQ) gave acc=58.61 on platform, AWQ could push to **62-68 acc** on
same platform. Still below 80% gate, but **significantly closer**. If combined
with chunked-prefill 32K (v_awq2 follow-up) and op-fusion (v_awq3),
could plausibly approach the gate.

## Risks

| Risk | Mitigation |
|---|---|
| llmcompressor 0.7+ not on cn mirrors | install_with_cn_fallbacks tries Tsinghua / Aliyun / PyPI |
| AWQModifier API in llmcompressor varies by version | quant script has fallback to GPTQModifier if AWQModifier import fails |
| `--quantization compressed-tensors` SGLang path less tested for SALA | Bundled SGLang has the same minicpm.py as 1849; should work |
| AWQ on hybrid-attention (lightning) untested | Default to `MLP_ONLY=1` (skip lightning attention quant) |
| compressed-tensors format on Marlin kernel needs verification | Defer to GPU run; falls back to slower kernel if mismatch |

## Tunable env vars for prepare_model.sh

```bash
CALIB_JSONL=/path/to/perf_public_set.jsonl   # override calib source
NUM_CALIB=256                                  # number of calibration prompts
MAX_CALIB_LEN=8192                             # max tokens per prompt
AWQ_SCHEME=W4A16_ASYM                          # or W4A16_SYM for sym variant
MLP_ONLY=1                                     # set 0 to also quantize attention
```

## Wall time estimate

- prepare_env: ~5-10 min (installs llmcompressor + transformers downgrade + flash_attn)
- prepare_model (quantize): ~60-180 min (AWQ is faster than GPTQ on calib)
- SGLang launch + 5h eval: same as 1849

Total: ~6-7h on platform.

## Submission ordering

This is a **major path change** (different tool, different format, different loader).
Submit ONLY when:
1. The BF16 path (v5d/v5e/v6) has settled the rank-20 floor question
2. You have a submission slot to spare for a high-variance bet

If v5e reaches ~28-30 final_score → rank 20 secured → AWQ becomes a stretch
for higher score.
If BF16 path can't push past gate → AWQ is the next-best alternative for
moving acc upward (which then unlocks throughput → final_score gains).

## Validation before submitting

```bash
# Lint should be all PASS
python3 tools/hard_constraints_lint.py --variant submission_awq_llmcompressor

# Pack
python3 tools/pack_submission.py --variant submission_awq_llmcompressor --check-only
python3 tools/pack_submission.py --variant submission_awq_llmcompressor \
    --output soar_awq_llmcompressor_$(date +%Y%m%d_%H%M).tar.gz
```

Tarball expected ~250 MB (bundles flash_attn 240 MB + sglang 10 MB).
