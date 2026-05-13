# Operator Fusion Analysis for MiniCPM-SALA

Investigated 2026-05-14 while user was asleep. Concrete findings + recommended
next steps. **No code changes made**, just research notes.

## Champion's hint (SOAR Week 4 - 智算一队)

> "原始代码中 RMSNorm 归一化 + RoPE 旋转位置编码，是典型的可以融合的算子，
> sglang 库也提供了相应的融合算子库可以适配利用"

So the champion explicitly mentioned RMSNorm and RoPE fusion. Let's see where
SALA stands on each.

## Finding 1: RMSNorm + residual-add is NOT fused (clearest win)

### Current code

`python/sglang/srt/models/minicpm.py:486-505`:

```python
# Self Attention
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(...)
hidden_states = residual + hidden_states * (
    self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
)

# Fully Connected
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states * (
    self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
)
```

Two unfused RMSNorm calls per layer × 32 layers = **64 unfused RMSNorms** per
forward pass. Each is two passes over the hidden state (read input, compute
variance, write output, then later read for residual add).

### Available fused op in sglang

`python/sglang/srt/layers/layernorm.py:103-133`:

```python
def forward_cuda(self, x, residual=None, **kwargs):
    ...
    if residual is not None:
        fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
        return x, residual
    out = rmsnorm(x, self.weight.data, self.variance_epsilon)
    return out
```

`fused_add_rmsnorm` does `residual_add → rmsnorm` **in-place** in a single
kernel. Used by every other LLM model in sglang (llama, qwen, gemma, etc).

### Why MiniCPM doesn't use it: the scale_depth trick

MiniCPM scales each residual contribution by `scale_depth / sqrt(num_layers)`.
This is unique to MiniCPM's training recipe. `fused_add_rmsnorm` does
`x = rmsnorm(x + residual)` without scaling.

### Recommended fix paths (ranked)

#### Path A: pre-scale weights at load time (cleanest, low risk)

The math: `residual + hidden * c` is mathematically equivalent to
`residual + hidden'` if we baked `c` into the layer that produces `hidden`.
The output of `self.self_attn(...)` ultimately comes from `o_proj`'s weight.
Multiply `o_proj.weight` and `mlp.down_proj.weight` by `c` at load time.

Pros: zero kernel work, fully equivalent math, single-pass change.
Cons: changes weight magnitudes (verify W4A16 quantization still works
correctly with scaled weights — might compress them slightly differently;
re-quantize after this change if applied together with W4A16).

#### Path B: custom Triton kernel with built-in scale

Write a `fused_scaled_add_rmsnorm(x, scale * residual, weight, eps)` Triton
kernel. ~30 lines of code, similar to existing rms_norm.

Pros: no weight magnitude tampering, easier to A/B benchmark.
Cons: another kernel to maintain.

#### Path C: don't fuse, leave it (do nothing)

Acceptable if other optimizations dominate. RMSNorm is fast already; the win
here is mainly memory bandwidth (one tensor pass instead of two).

### Estimated impact

Hard to say without measurement, but rough envelope:
- Each unfused RMSNorm pair is ~2-5 μs on H/B-class GPUs
- 64 per forward × thousands of forward passes during prefill
- **Estimated prefill speedup: 2-5%** (small but free)

## Finding 2: RoPE is already torch.compile'd; further fusion is harder

`python/sglang/srt/layers/rotary_embedding.py:135-140`:

```python
self._apply_rotary_emb_wrapped = _apply_rotary_emb
if torch_compile_available:
    self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(
        self._apply_rotary_emb_wrapped
    )
```

So apply_rotary is already JIT-compiled by torch. Standalone fusion (e.g.
RoPE + QKV reshape) would require touching the rotary_embedding module —
non-trivial and shared with other models.

**Verdict: skip for now**. The RoPE inputs and RMSNorm outputs are separated
by `qkv_proj` (a large GEMM), so they can't be trivially fused into one kernel
without rewriting qkv_proj.

## Finding 3: scale_depth multiply is itself unfused

The `hidden_states * (scale_depth / sqrt(num_layers))` line creates an extra
tensor pass. Could be folded into o_proj/down_proj output during torch.compile
if we mark the constant. But this is dominated by Path A above (which removes
the multiply entirely from runtime).

## Recommended action plan for user

1. **First — verify W4A16 win is real** (champion's #1 move). That's the
   biggest single-shot impact. Branch `quant/w4a16` is ready.

2. **Then — implement Path A (weight pre-scaling) on a new branch**.
   Steps:
   - Hook into model weight loading
   - Identify `o_proj.weight` and `mlp.down_proj.weight` for each layer
   - Multiply by `scale_depth / sqrt(num_hidden_layers)`
   - Change the forward code to use `fused_add_rmsnorm`:
     ```python
     hidden_states, residual = self.input_layernorm(hidden_states, residual)
     hidden_states = self.self_attn(...)  # output is now pre-scaled
     # residual is updated in-place by fused_add_rmsnorm next iteration
     ```
   - Bench. If math is correct, output throughput should match baseline plus a
     small win from reduced memory traffic.

3. **If Path A is too risky** (W4A16 + weight scaling compound), do Path B
   (custom kernel). More code but composes cleanly with W4A16.

4. **Skip RoPE fusion** — diminishing returns vs effort.

## Files investigated

- `python/sglang/srt/layers/layernorm.py` (RMSNorm class, fused_add_rmsnorm)
- `python/sglang/srt/layers/rotary_embedding.py` (RoPE)
- `python/sglang/srt/models/minicpm.py` (forward code with unfused pattern)
- `python/sglang/srt/models/minicpm3.py` (same unfused pattern — not solved upstream)
