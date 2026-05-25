# OP-FUSION PLATFORM FAILURE — confirmed 2026-05-25

> Critical finding: the chunk32k_opfusion REAL variant (overlay actually applied)
> got `acc_ori = 24.67, final_score = 0.0` on platform — a **~55pp collapse**
> from the chunk32k_safe baseline (80.31). Validates the explicit warning in
> `submission_*_opfusion/prepare_env.sh`: *"GPU correctness never verified
> end-to-end with SALA's specific kernels — if acc drops materially vs the
> no-overlay base, op-fusion is numerically wrong on this platform's GPU+CUDA
> combo. Fall back to chunk32k (no overlay)."*

## Platform score

```
[2026-05-25 01:32:10] [PENDING]
[2026-05-25 01:32:11] [PREPARING]
[2026-05-25 01:32:11] [DOWNLOADING] prepare_env.sh applied overlay/minicpm.py
[2026-05-25 01:53:41] [INFERENCING] server up
[2026-05-25 03:16:21] [SUCCESS]

Score:
  acc        = 30.83
  acc_ori    = 24.67    ← 55pp BELOW chunk32k_safe (80.31)
  final_score = 0.0     ← FAILED platform correctness gate (<80)
  benchmark_duration: S1=707.66s S8=1063.85s Smax=2352.32s
```

S1/S8/Smax durations are within ~1% of chunk32k_safe (720/1081/2386) — confirms
server ran healthy, throughput is normal. The collapse is **purely in accuracy
output**, not in performance.

## What the overlay actually changes (overlay/minicpm.py diff vs python/sglang/srt/models/minicpm.py)

Two changes:

### 1. RoPE fp32 upcast removed

Original (no overlay):
```python
orig_dtype = q.dtype
q, k = q.float(), k.float()         # upcast bf16 → fp32
# ... apply rotary_emb ...
q, k = q.to(orig_dtype), k.to(orig_dtype)  # cast back
```

With overlay: the three lines are deleted. `apply_rope` runs on raw bf16.

The justification was *"apply_rope already runs fp32 internally"*. The platform
result proves this is **false** for SALA's setup — without the explicit upcast,
RoPE numerical error compounds over long context.

### 2. Residual-delay pattern for RMSNorm fusion

Original (no overlay):
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
# ... attention ...
hidden_states = residual + hidden_states * (self.config.scale_depth / sqrt(N))
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
# ... mlp ...
hidden_states = residual + hidden_states * (self.config.scale_depth / sqrt(N))
return hidden_states, None
```

With overlay:
```python
scale = self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
# ... attention ...
hidden_states = hidden_states * scale
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
# ... mlp ...
hidden_states = hidden_states * scale
return hidden_states, residual
```

Intent: pass residual into RMSNorm to use SGLang's fused `add_rmsnorm` kernel.

Likely-broken aspect: the **scale factor placement** changed. In the original,
the scale multiplies the attention/mlp **output** before adding to residual.
In the overlay, the scale multiplies the **hidden_states post-attention/mlp**
which then becomes the new residual via `post_attention_layernorm(hidden_states, residual)`.
But `add_rmsnorm` adds `hidden_states + residual` THEN norms — so the post-scale
hidden_states gets added to the pre-scale residual without the prior scale-add
being applied to it. The math doesn't match the model definition.

The overlay's `op_fusion_verify.patch` (90 lines) tested CPU equivalence on
synthetic tensors — but **not** the actual model graph with SALA's hybrid
layers (dense minicpm4 + lightning), where the residual flow differs by layer
type.

## Why local 30-sample smoke saw `acc_ori = 82.67` (E1 of phase3, 2026-05-24)

The 30-sample local smoke uses `perf_public_set.jsonl`'s first 30 rows after
round-robin reordering. The numerical drift accumulates over **long context**:

- Short prompts (most of the first 30 in our smoke set) don't expose the bug.
- The platform's hidden eval set hits much longer contexts where the drift
  compounds → repetition-collapse or wrong-answer failure modes.

**Lesson: 30-sample local smoke is NOT sufficient validation for any change
that touches numerical paths** (RoPE, RMSNorm, residual, scale). Need ≥100
samples + at least one long-context (>16K) item before trusting the result.

## What to NOT do going forward

1. **Do not submit any variant with overlay/minicpm.py applied** until either:
   - The overlay's math is re-derived against SALA's full layer types (dense + lightning),
     OR
   - End-to-end GPU correctness is verified vs no-overlay baseline at ≥200 samples
     with prompts ≥32K tokens.
2. **Do not trust 30-sample smoke acc** when the change touches RoPE / RMSNorm /
   residual / scale — those errors only surface in long-context decode.
3. **The chunk32k_opfusion tarball** (if it exists in `~/OneDrive/soar_submissions/`)
   must be marked DO NOT SUBMIT.

## What stays valid

- **chunk32k_safe** (no overlay): platform `acc_ori=80.31, final_score=22.9` —
  this remains the production champion as of 2026-05-25.

## Cross-refs

- Overlay source: `submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion/overlay/minicpm.py`
- prepare_env.sh comment block warning about correctness: lines 346-356
- CPU-only verify patch: `experiments/op_fusion_verify.patch` (insufficient)
- phase3 E1 result commit: `aa80f7e phase3 E1 opfusion REAL: OK acc_ori=82.67`
- This handoff supersedes the +1-3 final_score prediction made in commits `ff07e49` and `aa80f7e`.
