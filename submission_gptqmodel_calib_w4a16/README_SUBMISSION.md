# SOAR W4A16 Submission — GPTQModel with SOAR-distribution calibration

This is the **third** W4A16 attempt, learning from the previous two failures:

| Attempt | Approach | Result | Diagnosis |
|---|---|---|---|
| v1 (RTN, /half bug) | Naive RTN, `scale = abs_max / 8` | acc=42.51, score=0 | Wrong scale formula |
| v1.1 (RTN, /(half-1)) | Naive RTN, `scale = abs_max / 7` | acc=42.18, score=0 | RTN math fine, but RTN itself too lossy on outlier-heavy weights |
| **this submission** | GPTQModel + SOAR-distribution calibration | TBD | — |

## Why this should clear correctness gate (≥ 80 acc_ori)

1. **GPTQ (Hessian-based) instead of RTN.** RTN per-group `abs_max` lets a few outliers dominate group scales, coarsening every other weight in the group. GPTQ uses second-order Hessian info to distribute quantization error wisely — typical accuracy improvement: 5-15 points over RTN on LLMs. The winners' notebook (周冠军笔记 04, 智算一队) explicitly states "GPTQ + Marlin" beats RTN, and that "calibration set composition iterated several rounds" was critical.

2. **Calibration data drawn from `perf_public_set.jsonl`** rather than 4 hardcoded templates. The previous GPTQModel attempt used synthetic prompts; this one defaults to actual SOAR distribution data (256 samples by default, max_calib_len=4096). The winners explicitly identified this as the key lever.

3. **Custom MiniCPM-SALA registration in GPTQModel.** SALA's `model_type = "minicpm_sala"` is not in the GPTQModel registry (verified against the main branch's `models/__init__.py`). This submission registers `MiniCPMSALAGPTQ` (subclass of `MiniCPMGPTQ`) with a module_tree that includes the Lightning Attention modules (`z_proj`, `q_norm`, `k_norm`, `o_norm`) marked as "do not quantize". Without this, GPTQModel would either fail to find the model class or quantize gating projections (high-risk for accuracy).

4. **Hard constraints respected** (learned from prior failures):
   - `--quantization gptq_marlin` (Marlin kernel only supports `(bits=4, sym=True)` → `uint4b8`)
   - `sym=True`, `desc_act=False`, `group_size=128`
   - NO `--kv-cache-dtype fp8_*` (verified incompatible with MiniCPM sparse backend)
   - NO `--disable-cuda-graph` (baseline runs with CUDA graph)

## Contents

| File | Purpose |
|---|---|
| `prepare_env.sh` | Install sglang + `gptqmodel>=7.0,<8.0` + transformers + accelerate; fp16-patch sparse backend; export `SGLANG_SERVER_ARGS`. |
| `prepare_model.sh` | Pass `--input/--output` to the quantizer; auto-include `--calib-jsonl /root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl` if present. |
| `quantize_gptqmodel_w4a16.py` | The actual quantizer. Registers `minicpm_sala`, loads model, runs GPTQ, writes Marlin-ready config. |
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

- prepare_env: 3-5 min (install GPTQModel + compile kernels)
- prepare_model (quantization): 15-30 min (256 samples × 4096 tokens × 32 layers)
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
