# SOAR AWQ via llm-compressor + Official Serving Args (W4A16 candidate)

> Built 2026-05-23 11:30 as **Experiment 2 (primary W4A16 candidate)** for
> tonight's migration validation. AWQ via llm-compressor (compressed-tensors
> format) + the verbatim official `run_sala_w4a16.sh` serving args.

## Purpose

v5g (BF16, chunked-prefill 8192) cleared the SOAR correctness gate at
`acc_ori=83.04`, proving:
- The bf16 family is correct (vs fp16 family which capped at acc_ori~47)
- The fp16 sed-patch on minicpm_backend.py was the killer (now permanently gone)

The OFFICIAL W4A16 path per `W4A16_README.md` + `run_sala_w4a16.sh`:
- llm-compressor + GPTQ (we use AWQModifier + QuantizationModifier here —
  see "AWQ vs GPTQ note" below)
- compressed-tensors output format (NOT gptq_marlin)
- bf16 native at the compute layer
- chunked-prefill 65536 + max-prefill 65536 + max-running 32 + mem-fraction 0.80

This variant uses the existing `submission_awq_llmcompressor` quant pipeline
(which already has all v5g+1849 fixes: H4 tokenizer overwrite, auto_map strip,
transformers 4.57.1 pin, bundled flash_attn) and ONLY changes the serving args
to the official 65K/0.80 recipe.

## Diff vs submission_awq_llmcompressor

**Single change** in `prepare_env.sh`:
```diff
- export SGLANG_SERVER_ARGS="... --chunked-prefill-size 8192 ... --quantization compressed-tensors --dtype bfloat16"
+ export SGLANG_SERVER_ARGS="... --chunked-prefill-size 65536 --max-prefill-tokens 65536 --max-running-requests 32 --mem-fraction-static 0.80 ... --quantization compressed-tensors --dtype bfloat16"
```

Everything else byte-identical (quantize script, prepare_model, calibration data).

## AWQ vs GPTQ note

Official `W4A16_README.md` says "llm-compressor + GPTQ". This variant uses
llm-compressor's `AWQModifier + QuantizationModifier` recipe instead. Reasoning:
- AWQ uses activation-aware channel scaling — claimed sweet spot for SALA's
  `scale_emb=12` outlier channels (research memo)
- OpenBMB ships AWQ (not GPTQ) for MiniCPM-V 4.5
- Both produce compressed-tensors format → same SGLang loader path
- Both should preserve the bf16 compute layer the same way

If tonight's local eval shows AWQ acc_ori is too low, we have a
straight-GPTQ-via-llmcompressor fallback to test (`submission_gptq_llmcompressor_official_args/`,
to be built if needed). For now AWQ is our primary; it's been the more
researched recipe for SALA-like distributions.

## Local validation matrix (tonight)

| Local acc_ori | Expected platform | Decision |
|---|---|---|
| ≥ 78 | ≥ 75 | **Submit this** — W4A16 official path works |
| 70-77 | 67-74 | Marginal; compare to BF16+official baseline by final_score formula |
| < 70 | < 67 | Try GPTQ-via-llmcompressor; if still bad, submit BF16+official |

## Why no MLP-only by default

`prepare_model.sh` defaults to `MLP_ONLY=1` (matches 1849's hybrid-attention
safety). But official `quantize_to_w4a16.py` quantizes ALL Linears. For tonight's
**first** experiment, leave MLP-only ON (lower risk, faster fallback if attention
quant breaks lightning recurrence). If MLP-only acc clears, follow-up experiment
toggles MLP_ONLY=0 to chase more bandwidth savings.

To run with full attention quant set `MLP_ONLY=0` before invoking prepare_model.

## Local quant cache

`scripts/local_eval.sh` caches the quantized model at
`/root/autodl-fs/zyn/models/submission_awq_official_args-quantized/`. Pass
`--force-requant` to rebuild.
