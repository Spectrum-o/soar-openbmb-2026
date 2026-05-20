# SOAR W4A16 Submission — GPTQModel with SOAR-distribution calibration

This is the latest W4A16 attempt, learning from the previous failures:

| Attempt | Approach | Result | Diagnosis |
|---|---|---|---|
| v1 (RTN, /half bug) | Naive RTN, `scale = abs_max / 8` | acc=42.51, score=0 | Wrong scale formula |
| v1.1 (RTN, /(half-1)) | Naive RTN, `scale = abs_max / 7` | acc=42.18, score=0 | RTN math fine, but RTN itself too lossy on outlier-heavy weights |
| v2 (GPTQModel, BatchEncoding bug) | GPTQModel + bundled calibration | prepare_model failed | GPTQModel 7.x rejected HuggingFace `BatchEncoding`; fixed by emitting plain dict tensors |
| v3 (GPTQModel, optional module bug) | GPTQModel + custom SALA module tree | prepare_model failed | GPTQModel required `self_attn.o_gate` on every layer; fixed by removing optional SALA modules from the common tree |
| v4 (GPTQModel, inherited template bug) | Subclassed `MiniCPMGPTQ` | prepare_model failed | Inherited `layer_modules` still contained `self_attn.o_gate`; fixed by subclassing `BaseGPTQModel` and registering `auto.MODEL_MAP` |
| v5 (full-attention GPTQ) | Quantized q/k/v/o + MLP | acc=0, score=0 | Served successfully but model semantics collapsed; likely attention/Lightning projections too sensitive or incompatible with fused SGLang loader |
| v6 (MLP-only, serialized derived config) | GPTQModel MLP-only + SOAR-distribution calibration | SGLang startup failed | `has_sparse_attention` was written into config.json even though SGLang exposes it as a read-only derived property |
| **this submission** | GPTQModel MLP-only + SOAR-distribution calibration | TBD | Keep attention/Lightning BF16; quantize only `mlp.gate_proj/up_proj/down_proj`; do not serialize derived config properties |

## Why this route is safer for correctness

1. **GPTQ (Hessian-based) instead of RTN.** RTN per-group `abs_max` lets a few outliers dominate group scales, coarsening every other weight in the group. GPTQ uses second-order Hessian info to reduce the impact of quantization error compared with a one-shot round-to-nearest pass.

2. **Calibration data drawn from `perf_public_set.jsonl`** rather than 4 hardcoded templates. The previous GPTQModel attempt used synthetic prompts; this one defaults to actual SOAR distribution data bundled in the submission (256 samples by default; the 150 public rows are cycled deterministically, max_calib_len=4096).

3. **Custom MiniCPM-SALA registration in GPTQModel.** SALA's `model_type = "minicpm_sala"` is not in the GPTQModel registry. This submission registers `MiniCPMSALAGPTQ` (subclass of `BaseGPTQModel`, not `MiniCPMGPTQ`) in both known `MODEL_MAP` locations with an MLP-only module list: `gate_proj`, `up_proj`, and `down_proj`. Attention/Lightning projections stay BF16 because the full-attention GPTQ route served successfully but scored 0.

4. **Hard constraints respected** (learned from prior failures):
   - `--quantization gptq_marlin` (Marlin kernel only supports `(bits=4, sym=True)` → `uint4b8`)
   - `sym=True`, `desc_act=False`, `group_size=128`
   - NO `--kv-cache-dtype fp8_*` (verified incompatible with MiniCPM sparse backend)
   - NO `--disable-cuda-graph` (baseline runs with CUDA graph)
   - `quantization_config.dynamic` skips all `self_attn.*` modules so SGLang loads attention weights as BF16 while serving quantized MLP layers through Marlin.

## Contents

| File | Purpose |
|---|---|
| `prepare_env.sh` | Install sglang + `gptqmodel>=7.0,<8.0` + transformers + accelerate; skip PyPI if already installed; install bundled `flash_attn` wheel; fp16-patch sparse backend; export `SGLANG_SERVER_ARGS`. |
| `prepare_model.sh` | Pass `--input/--output` to the quantizer; auto-use bundled `perf_public_set.jsonl` unless `CALIB_JSONL` overrides it. |
| `quantize_gptqmodel_w4a16.py` | The actual quantizer. Registers `minicpm_sala`, loads model, runs GPTQ, writes Marlin-ready config. |
| `perf_public_set.jsonl` | Bundled SOAR-distribution calibration data, so platform runs do not fall back to synthetic calibration. |
| `flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl` | Bundled prebuilt wheel for SOAR Python 3.10 + torch 2.9 + CUDA 12.8; avoids slow GitHub download/source build on the platform. |
| `sglang/python/` | Current SGLang source. |

## Tunable env vars for prepare_model.sh

```bash
CALIB_JSONL=/path/to/perf_public_set.jsonl   # override calibration source
NUM_CALIB=256                                  # number of calibration prompts
MAX_CALIB_LEN=4096                             # max tokens per prompt
```

For higher accuracy (cost: longer quantization time):
```bash
NUM_CALIB=1024 MAX_CALIB_LEN=8192 bash prepare_model.sh --input ... --output ...
```

## Local sanity check (no GPU needed)

```bash
python3 quantize_gptqmodel_w4a16.py \
    --input  /any/path \
    --output /any/output \
    --dry-run
```

Prints the resolved configuration and a sample of calibration prompts.
Useful for verifying the script is wired correctly before burning
GPU time.

## Local eval workflow (recommended before submitting)

```bash
# On the GPU node:
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --num-samples 200
```

This runs quantize → launch sglang → eval_model.py → prints acc_ori.
Iterate locally until acc_ori ≥ 80, then upload to the platform.

## Estimated platform timeline

- prepare_env: seconds to a few minutes if dependencies are already present; otherwise installs GPTQModel from a China PyPI mirror and installs bundled flash-attn locally
- prepare_model (quantization): typically 45-75 min on the platform; hard-capped at 90 min
- SGLang warmup: 1-2 min
- bench_serving (S1 + S8 + Smax): ~80 min
- eval_model.py: ~10 min

Total: ~2-2.5 hours. Well inside the 5h platform budget.

## Fallback (if this still fails)

If acc_ori is still below 80 after this submission, the next escalation is:
1. **AWQ** (activation-aware weight quantization) — different algorithm,
   sometimes works better than GPTQ on outlier-heavy layers.
2. **Mixed precision** — keep attention layers in BF16, quantize only MLP.
3. **8-bit** instead of 4-bit (`--bits 8 --group-size 128`) — much
   less lossy, but smaller throughput gain.
