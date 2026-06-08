# Full-W4A16 Selective BF16 Recovery: Last7 Attention + Guarded RoPE Fusion

This variant starts from the platform-proven all-time-high
`submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120/`
package and changes only the MiniCPM RoPE call sites.

It is deliberately guarded: `prepare_model.sh` first runs
`probe_minicpm_rope_fusion.py --model-config "${INPUT_DIR}/config.json"` on the
platform GPU. The probe reads the actual MiniCPM-SALA config, checks dense and
lightning RoPE shapes, and requires bitwise equality against the current
fp32-upcast path. If the model shape or kernel path is not supported, the
submission fails before GPTQ quantization.

## Platform Anchor

`soar_gptqmodel_full_w4a16_20260526_0239.tar.gz`:

- `acc=97.83`, `acc_ori=78.27`, `final_score=22.85`
- `S1=621.91`, `S8=1002.06`, `Smax=2315.78`

The speed win is real (`S1` is about 13% faster than v5j), but raw accuracy is
about 1.7pp below the likely 80 gate.

## Proven Parent

`soar_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_20260529_1035.tar.gz`:

- `acc=99.83`, `acc_ori=79.87`, `final_score=21.73`
- `S1=650.28`, `S8=1023.59`, `Smax=2334.43`
- Platform reached inferencing after about 27 minutes and completed in about
  2h29m, leaving useful headroom under the 5h platform limit.

## Default Recovery

After full W4A16 quantization, `prepare_model.sh` runs:

```bash
python3 apply_lightning_skip_overlay.py \
  --layers "${FULL_SELECTIVE_LAYERS:-last7-lightning}" \
  --modules "${FULL_SELECTIVE_MODULES:-attn}"
```

Default behavior restores only the last 7 lightning layers' attention
`q_proj/k_proj/v_proj/o_proj` weights to BF16. MLPs remain W4A16.

This is a speed recovery follow-up to the platform-proven `last8-lightning`
attention package (`acc_ori=80.51`, `S1=708.97`). It removes one restored
lightning layer to see whether the model stays above the 80 gate with lower
BF16 attention cost.

The direct parent for this RoPE probe is the platform-proven RMS-only
op-fusion + calib300 multi package. RMS fusion and GPTQ calibration stay
unchanged; this package tests whether Python-side RoPE fp32 materialization can
be removed without changing MiniCPM outputs.

## Calibration Change

Only these defaults change:

```bash
NUM_CALIB=300
CALIB_WINDOW_MODE=multi-adaptive
```

The existing 150 public rows are still the only bundled calibration source. The
quantizer deterministically cycles rows to reach `NUM_CALIB`, then
`multi-adaptive` creates 1-3 windows per prompt so GPTQ sees tail, middle, and
one extra long-context window for very long rows. This trades a longer
`prepare_model.sh` phase for better Hessian coverage of haystack and long QA
activations.

## Compatibility Notes

- Checkpoint tensor names remain HF-style:
  `self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`.
- SGLang runtime skip rules use fused names:
  `self_attn.qkv_proj$` and `self_attn.o_proj$`.
- The overlay physically removes orphan `.qweight/.qzeros/.scales/.g_idx`
  tensors from safetensors shards, because SGLang iterates physical keys after
  file-level index filtering.
- `overlay/minicpm.py` removes the explicit RoPE fp32 upcast at the dense and
  lightning attention sites, but the package is not allowed to run unless
  `probe_minicpm_rope_fusion.py` proves the no-upcast path is bitwise identical
  to MiniCPM's current fp32-upcast path for the actual input model config.
- The probe requires MiniCPM-SALA identity, `mixer_types` length matching
  `num_hidden_layers`, supported CUDA RoPE head dimensions, and both dense and
  lightning shapes when those layer types use RoPE.
- The RMSNorm residual-delay form preserves the MiniCPM formula
  `residual + sublayer_output * scale_depth / sqrt(num_layers)` by delaying the
  scaled sublayer output into the next fused `RMSNorm(x, residual)` call. The
  unit test covers both injected outputs and toy attention/MLP modules that
  depend on the normed hidden states.

## Knobs

- `FULL_SELECTIVE_LAYERS=last4-lightning`, `last8-lightning`, or explicit
  `23,27,29-31`
- `FULL_SELECTIVE_MODULES=attn`, `qkv`, `o_proj`, `mlp`, or comma lists such as
  `attn,down_proj`

## Submission Risk

This is a speed probe on top of the current best RMS-opfusion point. It should
not replace the proven package unless the RoPE probe passes and platform
accuracy stays within noise while benchmark duration improves. If the strict
probe fails, this branch is blocked; do not submit an unguarded no-upcast RoPE
package.
