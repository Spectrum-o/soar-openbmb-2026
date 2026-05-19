"""Numerical equivalence test for MiniCPM op-fusion rewrite.

Validates that the new residual-delay + fused-style add-then-norm forward
path produces the same output as the original manual residual-add +
separate RMSNorm path.

Does NOT depend on sglang or load SALA weights. Uses a pure-torch RMSNorm
that mimics the contract of sglang's RMSNorm:
    norm(x)            -> normed x
    norm(x, residual)  -> (normed (x + residual), x + residual)

The point is to verify *math equivalence* of the forward rewrite. The
real fused_add_rmsnorm CUDA kernel correctness is sglang's own
responsibility.

Runs on CPU or any GPU.
"""

import math
import sys

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Mimics sglang.srt.layers.layernorm.RMSNorm contract for this test."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        inv_rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * inv_rms).to(orig_dtype) * self.weight

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        if residual is None:
            return self._rmsnorm(x)
        residual = residual + x
        return self._rmsnorm(residual), residual


def make_layer(h, eps, device, dtype):
    layer = nn.Module()
    layer.input_layernorm = RMSNorm(h, eps=eps)
    layer.post_attention_layernorm = RMSNorm(h, eps=eps)
    layer.attn_proj = nn.Linear(h, h, bias=False)
    layer.mlp_proj = nn.Linear(h, h, bias=False)
    return layer.to(device=device, dtype=dtype)


def baseline_layer_forward(layer, hidden_states, scale):
    """Original MiniCPM forward — explicit residual + scaled add."""
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = torch.tanh(layer.attn_proj(hidden_states))
    hidden_states = residual + hidden_states * scale

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = torch.tanh(layer.mlp_proj(hidden_states))
    hidden_states = residual + hidden_states * scale
    return hidden_states


def fused_layer_forward(layer, hidden_states, residual, scale):
    """Rewritten MiniCPM forward — residual-delay, sublayer outputs pre-scaled."""
    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    hidden_states = torch.tanh(layer.attn_proj(hidden_states))
    hidden_states = hidden_states * scale

    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    hidden_states = torch.tanh(layer.mlp_proj(hidden_states))
    hidden_states = hidden_states * scale
    return hidden_states, residual


def run(device, dtype, num_layers=4, hidden_size=256, num_tokens=32, seed=0):
    torch.manual_seed(seed)
    eps = 1e-6
    scale_depth = 1.4
    scale = scale_depth / math.sqrt(num_layers)

    layers = nn.ModuleList(
        [make_layer(hidden_size, eps, device, dtype) for _ in range(num_layers)]
    )
    final_norm = RMSNorm(hidden_size, eps=eps).to(device=device, dtype=dtype)

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    h_base = x.clone()
    for layer in layers:
        h_base = baseline_layer_forward(layer, h_base, scale)
    out_base = final_norm(h_base)

    h_fused = x.clone()
    residual = None
    for layer in layers:
        h_fused, residual = fused_layer_forward(layer, h_fused, residual, scale)
    out_fused, _ = final_norm(h_fused, residual)

    diff = (out_base.float() - out_fused.float()).abs()
    return out_base, out_fused, diff


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print()

    configs = [
        ("float32",  torch.float32,  1e-5, 1e-5),
        ("bfloat16", torch.bfloat16, 5e-3, 1e-2),
        ("float16",  torch.float16,  5e-3, 1e-2),
    ]

    all_pass = True
    for name, dtype, atol, rtol in configs:
        try:
            out_base, out_fused, diff = run(device, dtype)
        except Exception as e:
            print(f"[{name}] EXCEPTION: {type(e).__name__}: {e}")
            all_pass = False
            continue
        ok = torch.allclose(out_base, out_fused, atol=atol, rtol=rtol)
        all_pass = all_pass and ok
        print(
            f"[{name}] max_abs={diff.max().item():.3e}  "
            f"mean_abs={diff.mean().item():.3e}  "
            f"allclose(atol={atol}, rtol={rtol})={ok}"
        )

    print()
    print("RESULT:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
