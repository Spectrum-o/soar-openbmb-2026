#!/usr/bin/env python3
"""Math helpers for per-head FP8 KV scales on scalar-scale attention wrappers.

FlashInfer's paged MiniCPM path currently exposes scalar `k_scale/v_scale`.
Per-head scales can still be represented around the wrapper:

    K_real = K_fp8 * k_scale[head]
    V_real = V_fp8 * v_scale[head]

For each query head that maps to a KV head, this is equivalent to:

    attention(Q, K_real, V_real)
    == postscale_v(attention(prescale_q(Q, k_scale), K_fp8, V_fp8), v_scale)

This module is intentionally standalone and CPU-testable. It does not patch the
runtime by itself; it defines the exact tensor transformations that a MiniCPM
runtime experiment should apply before/after the FlashInfer paged wrapper if
non-scalar checkpoint scales become available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def q_head_scale_from_kv_scale(kv_scale: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Expand `[num_kv_heads]` scale values to `[num_q_heads]` GQA layout."""
    kv_scale = torch.as_tensor(kv_scale)
    if kv_scale.ndim == 0:
        return kv_scale.reshape(1).expand(num_q_heads)
    if kv_scale.ndim != 1:
        raise ValueError(f"kv_scale must be scalar or rank-1, got shape {tuple(kv_scale.shape)}")
    num_kv_heads = kv_scale.numel()
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads={num_q_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )
    return kv_scale.repeat_interleave(num_q_heads // num_kv_heads)


def scale_query_for_per_head_k(q: torch.Tensor, k_scale: torch.Tensor) -> torch.Tensor:
    """Apply per-KV-head K dequant scale to Q before a scalar-scale wrapper.

    `q` must have shape `[..., num_q_heads, head_dim]`.
    """
    if q.ndim < 2:
        raise ValueError(f"q must have at least 2 dimensions, got {tuple(q.shape)}")
    q_head_scale = q_head_scale_from_kv_scale(k_scale.to(device=q.device), q.shape[-2])
    view_shape = [1] * q.ndim
    view_shape[-2] = q.shape[-2]
    return q * q_head_scale.reshape(view_shape).to(dtype=q.dtype)


def scale_output_for_per_head_v(o: torch.Tensor, v_scale: torch.Tensor) -> torch.Tensor:
    """Apply per-KV-head V dequant scale to wrapper output.

    `o` must have shape `[..., num_q_heads, head_dim]`.
    """
    if o.ndim < 2:
        raise ValueError(f"o must have at least 2 dimensions, got {tuple(o.shape)}")
    q_head_scale = q_head_scale_from_kv_scale(v_scale.to(device=o.device), o.shape[-2])
    view_shape = [1] * o.ndim
    view_shape[-2] = o.shape[-2]
    return o * q_head_scale.reshape(view_shape).to(dtype=o.dtype)


def scale_kv_for_cache_write(
    kv: torch.Tensor,
    scale: torch.Tensor,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> torch.Tensor:
    """Divide real K/V by per-head scale before FP8 cache storage.

    Supports either shaped `[..., num_kv_heads, head_dim]` tensors or flattened
    `[..., num_kv_heads * head_dim]` tensors.  The return shape matches `kv`.
    """
    scale = torch.as_tensor(scale, device=kv.device)
    if scale.ndim == 0:
        return kv / scale.to(dtype=kv.dtype)
    if scale.ndim != 1:
        raise ValueError(f"scale must be scalar or rank-1, got shape {tuple(scale.shape)}")

    if kv.ndim >= 2 and kv.shape[-2] == scale.numel():
        view_shape = [1] * kv.ndim
        view_shape[-2] = scale.numel()
        view_shape[-1] = 1
        return kv / scale.reshape(view_shape).to(dtype=kv.dtype)

    if num_kv_heads is None:
        num_kv_heads = scale.numel()
    if head_dim is None:
        if kv.shape[-1] % num_kv_heads != 0:
            raise ValueError(
                f"cannot infer head_dim from kv shape {tuple(kv.shape)} and {num_kv_heads} heads"
            )
        head_dim = kv.shape[-1] // num_kv_heads

    if scale.numel() != num_kv_heads:
        raise ValueError(
            f"scale has {scale.numel()} entries but num_kv_heads={num_kv_heads}"
        )
    if kv.shape[-1] != num_kv_heads * head_dim:
        raise ValueError(
            f"flat kv last dim {kv.shape[-1]} != num_kv_heads * head_dim "
            f"({num_kv_heads} * {head_dim})"
        )

    flat_scale = scale.repeat_interleave(head_dim).to(dtype=kv.dtype)
    return kv / flat_scale.reshape([1] * (kv.ndim - 1) + [kv.shape[-1]])


def reference_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Small CPU reference for GQA attention.

    Shapes:
      q: `[num_q_tokens, num_q_heads, head_dim]`
      k: `[num_kv_tokens, num_kv_heads, head_dim]`
      v: `[num_kv_tokens, num_kv_heads, head_dim]`
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be rank-3 tensors")
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head_dim differ")

    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    group = num_q_heads // num_kv_heads
    outputs = []
    for q_head in range(num_q_heads):
        kv_head = q_head // group
        scores = q[:, q_head, :] @ k[:, kv_head, :].transpose(0, 1)
        probs = torch.softmax(scores * sm_scale, dim=-1)
        outputs.append(probs @ v[:, kv_head, :])
    return torch.stack(outputs, dim=1)


def bridge_attention_with_scaled_cache(
    q: torch.Tensor,
    k_scaled_cache: torch.Tensor,
    v_scaled_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Reference wrapper-equivalent path for per-head scaled FP8 cache values."""
    q_prescaled = scale_query_for_per_head_k(q, k_scale)
    out_raw = reference_gqa_attention(q_prescaled, k_scaled_cache, v_scaled_cache, sm_scale)
    return scale_output_for_per_head_v(out_raw, v_scale)


def load_per_head_scales_from_report(path: Path, layer: int) -> dict[str, torch.Tensor]:
    """Load per-head scale arrays for one layer from a kv_calibrate JSON report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    layer_report = report.get("layers", {}).get(str(layer))
    if not isinstance(layer_report, dict):
        raise KeyError(f"layer {layer} not found in {path}")

    result: dict[str, torch.Tensor] = {}
    for projection in ("k", "v"):
        per_head = layer_report.get(f"{projection}_per_head")
        if not isinstance(per_head, dict) or "scale" not in per_head:
            raise KeyError(f"layer {layer} has no {projection}_per_head.scale")
        result[f"{projection}_scale"] = torch.tensor(per_head["scale"], dtype=torch.float32)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, help="Optional fp8kv_scales_report.json")
    ap.add_argument("--layer", type=int, help="Layer id to extract from --report")
    ap.add_argument("--json", action="store_true", help="Print extracted scales as JSON")
    args = ap.parse_args()

    if args.report is None:
        print(__doc__.strip())
        return 0
    if args.layer is None:
        raise SystemExit("--layer is required with --report")

    scales = load_per_head_scales_from_report(args.report, args.layer)
    if args.json:
        print(json.dumps({k: v.tolist() for k, v in scales.items()}, indent=2))
    else:
        print(f"layer {args.layer}")
        for name, tensor in scales.items():
            print(f"{name}: shape={tuple(tensor.shape)} min={tensor.min().item():.6g} max={tensor.max().item():.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
