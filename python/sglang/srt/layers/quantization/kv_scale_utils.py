from __future__ import annotations

from typing import Optional, Union

import torch


def divide_kv_cache_by_scale_(
    cache: torch.Tensor,
    scale: Union[float, torch.Tensor, None],
    *,
    num_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """Divide KV values by scalar or per-head scale before quantized storage."""
    if scale is None:
        return cache

    if not torch.is_tensor(scale):
        cache.div_(scale)
        return cache

    scale = scale.to(device=cache.device)
    if scale.numel() == 1:
        cache.div_(scale.reshape(()))
        return cache
    if scale.ndim != 1:
        raise ValueError(
            f"KV cache scale must be scalar or rank-1, got shape {tuple(scale.shape)}"
        )

    scale_heads = scale.numel()
    if num_heads is None:
        num_heads = scale_heads
    if scale_heads != num_heads:
        raise ValueError(
            f"KV cache scale has {scale_heads} heads but num_heads={num_heads}"
        )

    if cache.ndim >= 3 and cache.shape[-2] == num_heads:
        view_shape = [1] * cache.ndim
        view_shape[-2] = num_heads
        view_shape[-1] = 1
        cache.div_(scale.reshape(view_shape))
        return cache

    if cache.ndim >= 2:
        if head_dim is None:
            if cache.shape[-1] % num_heads != 0:
                raise ValueError(
                    f"cannot infer head_dim from KV cache shape {tuple(cache.shape)} "
                    f"and num_heads={num_heads}"
                )
            head_dim = cache.shape[-1] // num_heads
        if cache.shape[-1] == num_heads * head_dim:
            flat_scale = scale.repeat_interleave(head_dim)
            cache.div_(flat_scale.reshape([1] * (cache.ndim - 1) + [cache.shape[-1]]))
            return cache

    raise ValueError(
        f"cannot broadcast KV cache scale shape {tuple(scale.shape)} to "
        f"cache shape {tuple(cache.shape)}"
    )


def multiply_kv_cache_by_scale_(
    cache: torch.Tensor,
    scale: Union[float, torch.Tensor, None],
    *,
    num_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """Multiply KV values by scalar or per-head scale after quantized storage."""
    if scale is None:
        return cache

    if not torch.is_tensor(scale):
        cache.mul_(scale)
        return cache

    scale = scale.to(device=cache.device)
    if scale.numel() == 1:
        cache.mul_(scale.reshape(()))
        return cache
    if scale.ndim != 1:
        raise ValueError(
            f"KV cache scale must be scalar or rank-1, got shape {tuple(scale.shape)}"
        )

    scale_heads = scale.numel()
    if num_heads is None:
        num_heads = scale_heads
    if scale_heads != num_heads:
        raise ValueError(
            f"KV cache scale has {scale_heads} heads but num_heads={num_heads}"
        )

    if cache.ndim >= 3 and cache.shape[-2] == num_heads:
        view_shape = [1] * cache.ndim
        view_shape[-2] = num_heads
        view_shape[-1] = 1
        cache.mul_(scale.reshape(view_shape))
        return cache

    if cache.ndim >= 2:
        if head_dim is None:
            if cache.shape[-1] % num_heads != 0:
                raise ValueError(
                    f"cannot infer head_dim from KV cache shape {tuple(cache.shape)} "
                    f"and num_heads={num_heads}"
                )
            head_dim = cache.shape[-1] // num_heads
        if cache.shape[-1] == num_heads * head_dim:
            flat_scale = scale.repeat_interleave(head_dim)
            cache.mul_(flat_scale.reshape([1] * (cache.ndim - 1) + [cache.shape[-1]]))
            return cache

    raise ValueError(
        f"cannot broadcast KV cache scale shape {tuple(scale.shape)} to "
        f"cache shape {tuple(cache.shape)}"
    )


def make_kv_cache_scale_loader(layer: torch.nn.Module, per_head_attr: str):
    def load_scale(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if not isinstance(loaded_weight, torch.Tensor):
            loaded_weight = loaded_weight[:]
        loaded_weight = loaded_weight.detach()
        if loaded_weight.numel() == 1:
            param.data.fill_(loaded_weight.item())
            setattr(layer, per_head_attr, None)
            return
        if loaded_weight.ndim != 1:
            raise ValueError(
                "FP8 KV cache scale must be scalar or rank-1 per-head tensor, "
                f"got shape {tuple(loaded_weight.shape)}"
            )

        per_head_scale = loaded_weight.to(
            device=param.device, dtype=torch.float32
        ).contiguous()
        setattr(layer, per_head_attr, per_head_scale)
        param.data.fill_(per_head_scale.max().item())

    return load_scale


def scalar_scale_value(scale: torch.Tensor) -> float:
    return float(scale.detach().to("cpu").item())


def valid_per_head_scale(layer: torch.nn.Module, attr: str):
    scale = getattr(layer, attr, None)
    if scale is None:
        return None
    if scale.ndim != 1:
        raise ValueError(
            f"{attr} must be a rank-1 per-head scale, got shape {tuple(scale.shape)}"
        )
    if torch.any(scale <= 0):
        raise ValueError(f"{attr} must contain only positive scale values")
    return scale.detach().to(device=layer.k_scale.device, dtype=torch.float32)


def finalize_kv_cache_scales(layer: torch.nn.Module, *, fp8_fnuz: bool) -> tuple[float, float]:
    k_per_head = valid_per_head_scale(layer, "k_scale_per_head")
    v_per_head = valid_per_head_scale(layer, "v_scale_per_head")

    if k_per_head is not None or v_per_head is not None:
        k_scalar = scalar_scale_value(layer.k_scale)
        v_scalar = scalar_scale_value(layer.v_scale)
        if k_per_head is None:
            k_per_head = (
                torch.full_like(v_per_head, k_scalar)
                if k_scalar > 0.0
                else v_per_head.clone()
            )
        if v_per_head is None:
            v_per_head = (
                torch.full_like(k_per_head, v_scalar)
                if v_scalar > 0.0
                else k_per_head.clone()
            )
        if k_per_head.numel() != v_per_head.numel():
            raise ValueError(
                "k_scale_per_head and v_scale_per_head must have the same "
                f"number of heads, got {k_per_head.numel()} and {v_per_head.numel()}"
            )
        if fp8_fnuz:
            k_per_head = k_per_head * 2
            v_per_head = v_per_head * 2

        layer.k_scale_per_head = k_per_head
        layer.v_scale_per_head = v_per_head
        k_scale = float(k_per_head.max().item())
        v_scale = float(v_per_head.max().item())
    else:
        k_scalar = scalar_scale_value(layer.k_scale)
        v_scalar = scalar_scale_value(layer.v_scale)
        if k_scalar > 0.0 and v_scalar > 0.0:
            k_scale = k_scalar
            v_scale = v_scalar
        elif k_scalar < 0.0 and v_scalar < 0.0:
            k_scale = 1.0
            v_scale = 1.0
        else:
            scale_to_duplicate = max(k_scalar, v_scalar)
            k_scale = scale_to_duplicate
            v_scale = scale_to_duplicate
        if fp8_fnuz:
            k_scale *= 2
            v_scale *= 2

    if not isinstance(k_scale, float) or not isinstance(v_scale, float):
        raise ValueError("Only support per-tensor scalar fallback for fp8 KV cache")

    layer.k_scale.copy_(k_scale)
    layer.v_scale.copy_(v_scale)
    layer.k_scale_float = k_scale
    layer.v_scale_float = v_scale
    return k_scale, v_scale


def expand_kv_scale_to_q_heads(
    kv_scale: torch.Tensor, num_q_heads: int
) -> torch.Tensor:
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


def scale_flat_query_for_per_head_k(
    q: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    num_q_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if q.shape[-1] != num_q_heads * head_dim:
        raise ValueError(
            f"flat query last dim {q.shape[-1]} != num_q_heads * head_dim "
            f"({num_q_heads} * {head_dim})"
        )
    q_view = q.reshape(-1, num_q_heads, head_dim)
    q_head_scale = expand_kv_scale_to_q_heads(
        k_scale.to(device=q.device), num_q_heads
    ).to(dtype=q.dtype)
    return (q_view * q_head_scale.reshape(1, num_q_heads, 1)).reshape_as(q)


def scale_grouped_output_for_per_head_v(
    output: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    num_q_heads: int,
    v_head_dim: int,
) -> torch.Tensor:
    if output.shape[-2] > num_q_heads or num_q_heads % output.shape[-2] != 0:
        raise ValueError(
            f"grouped output shape {tuple(output.shape)} is not compatible with "
            f"num_q_heads={num_q_heads}"
        )
    output_view = output.reshape(-1, num_q_heads, v_head_dim)
    q_head_scale = expand_kv_scale_to_q_heads(
        v_scale.to(device=output.device), num_q_heads
    ).to(dtype=output.dtype)
    return (output_view * q_head_scale.reshape(1, num_q_heads, 1)).reshape(
        -1, num_q_heads * v_head_dim
    )
