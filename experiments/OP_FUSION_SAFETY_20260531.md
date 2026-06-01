# MiniCPM-SALA Operator Fusion Safety Gate

## Hard invariant

Operator fusion is acceptable only if it preserves the model computation. If a
change relies on lower precision, a different accumulation order with visible
error, a removed cast, or a different residual formula, treat it as a numerical
change first and a speed optimization second. It must not be submitted as a
"safe fusion" package.

For this competition queue the submit gate is stricter than "close enough":
the fusion must be theoretically computation-preserving for the MiniCPM-SALA
path being served. A kernel whose own tests use loose tolerances is not enough
evidence for an accuracy-neutral submit package.

## Current code evidence

MiniCPM has three relevant fusion candidates:

1. RMSNorm residual-delay fusion in `MiniCPMDecoderLayer`
   - This is the only fusion currently allowed in submit candidates.
   - It keeps the MiniCPM formula
     `residual + sublayer_output * scale_depth / sqrt(num_hidden_layers)` by
     delaying the scaled sublayer output into the next fused
     `RMSNorm(x, residual)` call.
   - Platform-proven package: `last7attn_rmsopfusion`, which completed with
     `acc_ori=79.87` and kept RoPE fp32 upcast.

2. RoPE fp32-upcast removal
   - Current MiniCPM dense attention and lightning attention both explicitly do:
     `orig_dtype = q.dtype`, `q, k = q.float(), k.float()`,
     `self.rotary_emb(...)`, then cast back.
   - The old failed op-fusion package removed this cast and changed RMS/residual
     in the same overlay. Platform hidden/long-context accuracy collapsed.
   - Under the hard invariant, this is not safe. Do not remove the upcast unless
     a full long-context equivalence run proves it.

3. Lightning QKNorm fusion
   - MiniCPM currently applies separate per-head RMSNorm calls:
     `q = self.q_norm(q.reshape(-1, self.head_dim))` and the same for `k`.
   - Other SGLang models can use `apply_qk_norm(...)`, which dispatches to the
     JIT `fused_inplace_qknorm` kernel when supported.
   - However, upstream `test_qknorm.py` accepts `atol=1e-2, rtol=1e-2`. That is
     approximate numerical equivalence, not a guarantee that acc is unchanged.
   - Therefore QKNorm JIT fusion is not a safe submit candidate yet.

## Submit rule

For current MiniCPM-SALA submissions:

- Keep RMS-only op-fusion if the package is based on the proven overlay.
- Keep both RoPE fp32 upcasts.
- Do not enable `apply_qk_norm` / JIT QKNorm fusion in submit candidates.
- FP8KV overlays must keep `maybe_remap_kv_scale_name`, otherwise injected
  `k_scale` / `v_scale` tensors can fail to load or silently use bad scale.

The static guard is `tests/test_minicpm_opfusion_safety.py`.
It now covers the active FP8KV mid-calibration queue (`174343`, `214103`,
`215412`, and `221735`) so a future overlay edit cannot silently remove RoPE
fp32 upcast, enable approximate QKNorm JIT fusion, or drop the FP8KV scale-name
remap from the packages we are actually submitting.
