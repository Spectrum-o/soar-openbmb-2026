# SOAR Full W4A16 Experiment — GPTQModel q/k/v/o + MLP

This variant intentionally tests the risky route: quantize both MiniCPM-SALA
attention/Lightning projections (`self_attn.q_proj/k_proj/v_proj/o_proj`) and
MLP projections (`mlp.gate_proj/up_proj/down_proj`) with GPTQModel W4A16.

It is separate from the stable MLP-only and FP8KV variants. Earlier notes say a
full-attention GPTQ attempt served but scored `acc=0`, so this branch is for
controlled re-testing with the current calibration, tokenizer, qzeros, and
SGLang loader fixes.

## What Changes

- `quantize_gptqmodel_w4a16.py` registers MiniCPM-SALA with full common linear
  modules: q/k/v/o + gate/up/down.
- Optional gates (`self_attn.o_gate`, `self_attn.z_proj`) and norm modules stay
  BF16 via `quantization_config.dynamic` skips.
- `prepare_model.sh` defaults to `GROUP_SIZE=128`; use `GROUP_SIZE=64` for a
  slower higher-quality run.
- `prepare_env.sh` is reused from the stable GPTQModel W4A16 package and does
  not enable FP8 KV. First prove full W4A16 independently, then layer FP8KV in a
  follow-up variant if accuracy survives.

## Suggested Local Run

```bash
GROUP_SIZE=128 NUM_CALIB=256 MAX_CALIB_LEN=8192   bash submission_gptqmodel_full_w4a16/prepare_model.sh   --input /root/autodl-fs/models/OpenBMB/MiniCPM-SALA   --output /root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16-quantized
```

Then launch with conservative serving args from `prepare_env.sh` and run the
local smoke/eval scripts before considering platform submission.

## Risk

This may quantize numerically sensitive sparse/Lightning attention projections.
Expect a real chance of accuracy collapse. Keep this branch isolated from the
verified FP8KV path.
