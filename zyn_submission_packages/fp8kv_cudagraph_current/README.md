# zyn fp8 KV CUDA graph current package

This package records the current runnable submission state used by the live
full-eval run on 2026-05-26.

## Entry Point

Use the repository `run_sala.sh`. It now mirrors the live server settings:

```bash
MODEL_PATH=/path/to/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized \
QUANTIZATION_PARAM_PATH=/path/to/minicpm_fp8_e4m3_kv_scales.json \
PORT=31111 \
bash run_sala.sh
```

The script enables:

- `--attention-backend minicpm_flashinfer`
- `--quantization gptq_marlin`
- `--kv-cache-dtype fp8_e4m3`
- `--dtype bfloat16`
- `--chunked-prefill-size 32768`
- `--max-prefill-tokens 32768`
- `--mem-fraction-static 0.70`
- `--cuda-graph-bs 1 2 4 8 12 16 24 32`
- `--quantization-param-path` when `QUANTIZATION_PARAM_PATH` is set

It also prepends the venv `bin` directory to `PATH` so FlashInfer JIT can find
`ninja`.

## Runtime Verified

Live server launch path:

`/root/autodl-tmp/zyn/sglang_runtime`

Live model path:

`/autodl-fs/data/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized`

Runtime dependency notes:

- `transformers==4.57.1`
- `torch==2.9.1+cu128`
- keep Transformers on 4.x

## Evidence

Current pushed eval snapshots:

- `zyn_eval_runs/fp8kv_cg_e4m3_20260526_180812_snapshot_0057`
- `zyn_eval_runs/fp8kv_cg_e4m3_20260526_180812_snapshot_0059`
- `zyn_eval_runs/fp8kv_cg_e4m3_20260526_180812_snapshot_0060`

The live logs confirm:

- `Using KV cache dtype: torch.float8_e4m3fn`
- CUDA graph capture completed
- decode is running with `cuda graph: True`

## KV Scale Calibration

The safe GPTQ model does not contain calibrated `k_scale` or `v_scale` tensors
for FP8 KV. Generate a scale JSON on the selected W4 model before packaging.
Prefer collecting through SGLang itself so the stats come from the same
`gptq_marlin` W4 serving path. The collection script defaults to BF16 KV cache
while measuring K/V amax, then writes scales for FP8 KV serving:

```bash
MODEL_PATH=/autodl-fs/data/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized \
DATA_PATH=/autodl-fs/data/zyn/calib_sets/w4_kv_selected_20260601.jsonl \
bash scripts/zyn_collect_kv_scales_sglang.sh
```

Package the resulting JSON and set `QUANTIZATION_PARAM_PATH` in the platform
server launch. Without it, SGLang defaults to scale `1.0`, which is the
accuracy risk this branch is fixing.
