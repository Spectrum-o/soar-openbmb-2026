# Full-W4A16 Selective BF16 Recovery: Last7 Attention

This variant starts from `submission_gptqmodel_full_w4a16/` and adds a narrow
post-quant overlay.

## Platform Anchor

`soar_gptqmodel_full_w4a16_20260526_0239.tar.gz`:

- `acc=97.83`, `acc_ori=78.27`, `final_score=22.85`
- `S1=621.91`, `S8=1002.06`, `Smax=2315.78`

The speed win is real (`S1` is about 13% faster than v5j), but raw accuracy is
about 1.7pp below the likely 80 gate.

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

## Compatibility Notes

- Checkpoint tensor names remain HF-style:
  `self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`.
- SGLang runtime skip rules use fused names:
  `self_attn.qkv_proj$` and `self_attn.o_proj$`.
- The overlay physically removes orphan `.qweight/.qzeros/.scales/.g_idx`
  tensors from safetensors shards, because SGLang iterates physical keys after
  file-level index filtering.

## Knobs

- `FULL_SELECTIVE_LAYERS=last4-lightning`, `last8-lightning`, or explicit
  `23,27,29-31`
- `FULL_SELECTIVE_MODULES=attn`, `qkv`, `o_proj`, `mlp`, or comma lists such as
  `attn,down_proj`

## Submission Risk

This is a speed-route experiment. It should not replace the known working
non-FP8 package unless platform accuracy clears the gate and timing stays close
to the full-W4A16 anchor.
