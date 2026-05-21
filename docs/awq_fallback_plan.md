# AWQ fallback plan

> Read this if GPTQ + Marlin + calibration tuning keeps failing to
> clear platform's acc_ori ≥ 80 gate.

## Why AWQ?

Activation-aware Weight Quantization (AWQ) is structurally different
from GPTQ:
- GPTQ uses Hessian-based weight reconstruction per group, requiring
  ~256 calibration samples and 14+ min of compute on SALA.
- AWQ uses per-channel scaling search informed by activation magnitudes,
  requires ~10x less calibration data, and is **less sensitive to
  domain shift** — exactly the property we'd want if our problem turns
  out to be calibration-distribution-induced.

The SOAR Week 4 champion's notes explicitly mention AWQ as a viable path,
and the May 14 artifact we already have on disk uses llm-compressor's
AWQ implementation.

## Pre-existing assets

- **`/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/`** — the artifact
  from 2026-05-14 produced via `quantize_to_w4a16.py` + llm-compressor.
  Format: **compressed-tensors**, NOT gptq_marlin.
  Status: never benchmarked. We do not know if it loads cleanly in
  SGLang's compressed-tensors loader or what its acc is.
- **`quantize_to_w4a16.py`** at repo root — the script that produced it.
  Uses llm-compressor's `oneshot()` API with a wikitext calibration
  default.

## Step-by-step plan to validate AWQ

This is sequential — each step's outcome decides whether the next is
worth doing.

### Step 1: load the May 14 artifact in SGLang (cheap, 10 min)

The artifact is 5GB, sglang load should be ~30s. Need to confirm
SGLang's `compressed-tensors` loader recognizes it.

```bash
cd /root/autodl-tmp/zyn/soar/sglang
source sglang_minicpm_sala_env/bin/activate

# Quick smoke test — bring up server, hit /health, send one prompt
python3 -m sglang.launch_server \
    --model-path /root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16 \
    --trust-remote-code \
    --port 31112 \
    --tp-size 1 \
    --quantization compressed-tensors \
    --dtype float16 \
    --attention-backend minicpm_flashinfer \
    --dense-as-sparse \
    --skip-server-warmup \
    &

# Wait for health
sleep 60
curl -sf http://127.0.0.1:31112/health && echo " server up"

# Single completion test
curl -s -X POST http://127.0.0.1:31112/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "any",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 50
    }' | jq -r '.choices[0].message.content'

# Kill
kill %1
```

**Expected outcomes**:
- 200 OK + reasonable response (e.g. "Hello") → AWQ artifact is viable;
  go to Step 2.
- 500 / KeyError / unrecognized quant format → AWQ artifact is broken;
  go to Step 3.

### Step 2: full eval on May 14 artifact (45 min)

If Step 1 produced reasonable output:

```bash
# Reuse local_eval.sh but override the quant artifact + sglang args.
# Easiest: hack the path constants temporarily, then revert.
# OR — use the standalone bench:
QUANT_OUT=/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16 \
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --skip-quant \
    --eval-data /root/autodl-tmp/zyn/soar/sglang/submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \
    --num-samples 150 \
    2>&1 | tee /root/autodl-fs/zyn/logs/expG_awq_may14_$(date +%Y%m%d_%H%M%S).log
```

But `local_eval.sh` derives `QUANT_OUT` from `--variant`. Need to either
pass `--quant-out` (check the flag) or temporarily symlink.

Cleaner: write a small `submission_awq_w4a16/` wrapper that calls into
the existing artifact via prepare_env's SGLANG_SERVER_ARGS pointed at
compressed-tensors. The variant dir only needs:
- `prepare_env.sh` exporting the right SGLANG_SERVER_ARGS
- `prepare_model.sh` as a symlink-passthrough (BF16 style) to the
  existing AWQ artifact

**Expected outcomes**:
- ≥ 80% → pack a tarball, submit. Done.
- 50-80% → AWQ is in the same boat as GPTQ; same problem (repetition
  collapse), needs Exp C/D from PLAN.md applied to it.
- < 50% → AWQ artifact is itself broken (likely the wikitext calibration
  is far off SOAR distribution). Re-quant with SOAR-distribution calib.

### Step 3: re-quant with AWQ + SOAR calibration (1-2 hours)

If May 14 doesn't load or scores too low:

```bash
cd /root/autodl-tmp/zyn/soar/sglang
source sglang_minicpm_sala_env/bin/activate

# Need llm-compressor (may already be installed; if not, ~30s install)
uv pip install "llmcompressor>=0.4" datasets

# Run AWQ with our perf_public_set as calibration (instead of wikitext)
python3 quantize_to_w4a16.py \
    --model /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
    --output /root/autodl-fs/zyn/models/MiniCPM-SALA-AWQ-SOAR \
    --calib-dataset submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \
    --calib-config "" \
    --calib-split train \
    --calib-text-column question \
    --num-samples 256 \
    --max-seq-len 8192
```

Wall: ~1-2 hours on RTX PRO 6000 (llm-compressor is slower than gptqmodel).

Then eval as in Step 2 with the new artifact path.

## Failure mode catalog

| Symptom                                                | Likely cause             | Fix                       |
|--------------------------------------------------------|--------------------------|---------------------------|
| `Unsupported quantization format: compressed-tensors` | SGLang version too old   | Bundle newer sglang       |
| `KeyError: model.layers.X.YYY.weight`                  | layout mismatch          | Check artifact structure  |
| Server starts, all outputs empty                       | Dtype mismatch (fp16/bf16)| Pass `--dtype float16`    |
| Generation collapses / loops (same as GPTQ)           | Calibration issue        | Apply Exp C/D from PLAN.md|

## What we'd give up by switching to AWQ

- Marlin's bandwidth-saving GEMM kernel (gptq_marlin). AWQ uses a
  different kernel path; may be slower at small batch sizes.
- The qzeros + Marlin saga we've already debugged. Starting fresh in
  compressed-tensors land means a new bug surface.

But: if acc clears 80%, throughput is a follow-up problem, not a
blocker.

## Cost summary

| Step | Wall time | Compute risk |
|------|-----------|--------------|
| 1    | ~10 min   | low — just loads existing artifact |
| 2    | ~45 min   | medium — uses a platform slot only if we submit |
| 3    | 1-2 hours | high — re-quant with new params, untested code path |
