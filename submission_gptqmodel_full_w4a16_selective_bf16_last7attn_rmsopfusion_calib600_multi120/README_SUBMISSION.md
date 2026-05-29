# Full-W4A16 Selective BF16 Recovery: Last7 Attention + RMS Op-Fusion + Calib600 Multi

This variant starts from the platform-proven
`submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion/` package
and changes only the GPTQ calibration recipe.

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

The direct parent for this calibration probe is the platform-proven RMS-only
op-fusion package above. RMS fusion stays unchanged; this package tests whether
spending more of the prepare-time budget on calibration improves the remaining
accuracy gap.

## Calibration Change

Only these defaults change:

```bash
NUM_CALIB=600
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
- `overlay/minicpm.py` keeps the explicit RoPE fp32 upcast at both attention
  sites. The old failed op-fusion package removed those casts and changed the
  RMSNorm residual path in the same overlay, so that failure does not isolate
  the RMSNorm change.
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

This is a quality probe on top of the current best RMS-opfusion point. It should
not replace the proven package unless platform accuracy improves enough to
offset any extra quantization variance. If prepare time approaches the
120-minute quant timeout, the lower-risk follow-up is `NUM_CALIB=400` with the
same `multi-adaptive` windowing, or the already proven RMS-opfusion package.
