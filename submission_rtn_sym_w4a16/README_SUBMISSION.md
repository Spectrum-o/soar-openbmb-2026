# SOAR Symmetric RTN W4A16 + Marlin Submission

This is the **fixed** version of the previous `soar_official_rtn_w4a16_g128_marlin_cfg_submission_*` package.

## What changed (vs the previous failing submission)

The previous package failed at server startup with:

```
ValueError: Unsupported quantization config: bits=4, sym=False
```

Root cause: `quantize_gptq_rtn.py` wrote `sym=False` in the model config because the underlying RTN math was **asymmetric** (`scale = (max - min) / 15`, plus per-group zero point from `-min`).

SGLang's `--quantization gptq_marlin` only accepts `(bits=4, sym=True) -> uint4b8`; asymmetric configs are rejected at `gptq.py:278`.

This package:

1. Replaces `quantize_gptq_rtn.py` with **symmetric** RTN:
   - `scale = max(abs(w)) / 8`  (per group)
   - `signed_q = clamp(round(w / scale), -8, 7)`
   - `stored_q = signed_q + 8`  → range `[0, 15]`, matches `uint4b8`
   - `qzeros` filled with the bias value `8` everywhere
2. Writes `sym=True` in both `config.json::quantization_config` and `quantize_config.json`.
3. Switches `--quantization gptq` → `--quantization gptq_marlin` in `SGLANG_SERVER_ARGS` (now valid because sym=True).
4. Drops `--disable-cuda-graph` (Marlin GEMM is CUDA-graph-friendly).

Other things unchanged from the stable RTN baseline:
- Same `prepare_env.sh` / `prepare_model.sh` shape.
- No external calibration dataset; no GPTQModel / llmcompressor dependency.
- Same MiniCPM-SALA backend bf16 → fp16 patch.

## Contents

| File | Purpose |
|---|---|
| `prepare_env.sh` | Install `sglang/python`, patch fp16, export `SGLANG_SERVER_ARGS` |
| `prepare_model.sh` | Dispatch the platform `--input/--output` to the quantizer |
| `quantize_gptq_rtn_sym.py` | Symmetric RTN → GPTQ tensors (`qweight`, `scales`, `qzeros`, `g_idx`) |
| `sglang/python/` | SGLang source from this checkout |

## Platform Flow

1. Platform sources `prepare_env.sh`.
2. Platform runs `bash prepare_model.sh --input <stock model> --output <processed model>`.
3. Platform launches SGLang with the exported `SGLANG_SERVER_ARGS`.

## Correctness expectation

Symmetric RTN typically loses **< 2 points** vs the BF16 baseline on benchmark accuracy. The SOAR correctness gate is 80 (baseline ~82). If `eval_model.py` drops below 80, fall back to a GPTQModel-based path (proper Hessian-based GPTQ).

## Local sanity check (optional)

Before submitting, on the GPU instance:

```bash
# Quantize
bash prepare_model.sh \
    --input  /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
    --output /root/autodl-tmp/models/MiniCPM-SALA-W4A16-RTN-SYM

# Verify config
cat /root/autodl-tmp/models/MiniCPM-SALA-W4A16-RTN-SYM/quantize_config.json
# Expect: "sym": true, "bits": 4, "group_size": 128

# Boot SGLang manually with the same args
source prepare_env.sh
python3 -m sglang.launch_server \
    --model-path /root/autodl-tmp/models/MiniCPM-SALA-W4A16-RTN-SYM \
    --trust-remote-code \
    --port 31111 \
    --tp-size 1 \
    --max-running-requests 32 \
    --enable-metrics \
    ${SGLANG_SERVER_ARGS}
```

Look for `The server is fired up and ready to roll!`. If you see the old `bits=4, sym=False` error again, the platform is probably running a stale cached package.
