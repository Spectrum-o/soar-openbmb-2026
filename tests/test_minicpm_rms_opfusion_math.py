#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def unfused_layers(
    hidden_states: torch.Tensor,
    attn_outputs: list[torch.Tensor],
    mlp_outputs: list[torch.Tensor],
    weights: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    eps: float,
) -> torch.Tensor:
    for (w1, w2), attn_out, mlp_out in zip(weights, attn_outputs, mlp_outputs):
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, w1, eps)
        # The real attention output is a function of the normed states; for this
        # algebra test we inject the same output into both graphs.
        hidden_states = residual + attn_out * scale

        residual = hidden_states
        hidden_states = rms_norm(hidden_states, w2, eps)
        hidden_states = residual + mlp_out * scale
    return rms_norm(hidden_states, weights[-1][1], eps)


def unfused_layers_with_modules(
    hidden_states: torch.Tensor,
    attn_weights: list[torch.Tensor],
    mlp_weights: list[torch.Tensor],
    weights: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    eps: float,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    module_inputs = []
    for (w1, w2), attn_w, mlp_w in zip(weights, attn_weights, mlp_weights):
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, w1, eps)
        attn_in = hidden_states
        attn_out = torch.tanh(attn_in.float() @ attn_w.float()).to(hidden_states.dtype)
        hidden_states = residual + attn_out * scale

        residual = hidden_states
        hidden_states = rms_norm(hidden_states, w2, eps)
        mlp_in = hidden_states
        mlp_out = torch.sin(mlp_in.float() @ mlp_w.float()).to(hidden_states.dtype)
        hidden_states = residual + mlp_out * scale
        module_inputs.append((attn_in, mlp_in))
    return rms_norm(hidden_states, weights[-1][1], eps), module_inputs


def rms_opfusion_layers(
    hidden_states: torch.Tensor,
    attn_outputs: list[torch.Tensor],
    mlp_outputs: list[torch.Tensor],
    weights: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    eps: float,
) -> torch.Tensor:
    residual = None
    for (w1, w2), attn_out, mlp_out in zip(weights, attn_outputs, mlp_outputs):
        if residual is None:
            residual = hidden_states
            hidden_states = rms_norm(hidden_states, w1, eps)
        else:
            residual = hidden_states + residual
            hidden_states = rms_norm(residual, w1, eps)
        hidden_states = attn_out * scale

        residual = hidden_states + residual
        hidden_states = rms_norm(residual, w2, eps)
        hidden_states = mlp_out * scale

    residual = hidden_states + residual
    return rms_norm(residual, weights[-1][1], eps)


def rms_opfusion_layers_with_modules(
    hidden_states: torch.Tensor,
    attn_weights: list[torch.Tensor],
    mlp_weights: list[torch.Tensor],
    weights: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    eps: float,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    residual = None
    module_inputs = []
    for (w1, w2), attn_w, mlp_w in zip(weights, attn_weights, mlp_weights):
        if residual is None:
            residual = hidden_states
            hidden_states = rms_norm(hidden_states, w1, eps)
        else:
            residual = hidden_states + residual
            hidden_states = rms_norm(residual, w1, eps)
        attn_in = hidden_states
        attn_out = torch.tanh(attn_in.float() @ attn_w.float()).to(hidden_states.dtype)
        hidden_states = attn_out * scale

        residual = hidden_states + residual
        hidden_states = rms_norm(residual, w2, eps)
        mlp_in = hidden_states
        mlp_out = torch.sin(mlp_in.float() @ mlp_w.float()).to(hidden_states.dtype)
        hidden_states = mlp_out * scale
        module_inputs.append((attn_in, mlp_in))

    residual = hidden_states + residual
    return rms_norm(residual, weights[-1][1], eps), module_inputs


class TestMiniCPMRmsOpfusionMath(unittest.TestCase):
    def test_residual_delay_is_algebraically_equivalent(self):
        torch.manual_seed(20260529)
        dtype = torch.bfloat16
        num_layers = 5
        shape = (17, 64)
        scale = 1.4 / math.sqrt(32)
        eps = 1e-6

        hidden_states = torch.randn(*shape, dtype=dtype)
        attn_outputs = [torch.randn(*shape, dtype=dtype) for _ in range(num_layers)]
        mlp_outputs = [torch.randn(*shape, dtype=dtype) for _ in range(num_layers)]
        weights = [
            (
                torch.randn(shape[-1], dtype=dtype).abs() + 0.5,
                torch.randn(shape[-1], dtype=dtype).abs() + 0.5,
            )
            for _ in range(num_layers)
        ]

        expected = unfused_layers(
            hidden_states.clone(), attn_outputs, mlp_outputs, weights, scale, eps
        )
        actual = rms_opfusion_layers(
            hidden_states.clone(), attn_outputs, mlp_outputs, weights, scale, eps
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_residual_delay_preserves_module_inputs(self):
        torch.manual_seed(20260529)
        dtype = torch.bfloat16
        num_layers = 8
        shape = (11, 32)
        scale = 1.4 / math.sqrt(32)
        eps = 1e-6

        hidden_states = torch.randn(*shape, dtype=dtype)
        attn_weights = [
            torch.randn(shape[-1], shape[-1], dtype=dtype) / math.sqrt(shape[-1])
            for _ in range(num_layers)
        ]
        mlp_weights = [
            torch.randn(shape[-1], shape[-1], dtype=dtype) / math.sqrt(shape[-1])
            for _ in range(num_layers)
        ]
        weights = [
            (
                torch.randn(shape[-1], dtype=dtype).abs() + 0.5,
                torch.randn(shape[-1], dtype=dtype).abs() + 0.5,
            )
            for _ in range(num_layers)
        ]

        expected, expected_inputs = unfused_layers_with_modules(
            hidden_states.clone(), attn_weights, mlp_weights, weights, scale, eps
        )
        actual, actual_inputs = rms_opfusion_layers_with_modules(
            hidden_states.clone(), attn_weights, mlp_weights, weights, scale, eps
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for (expected_attn, expected_mlp), (actual_attn, actual_mlp) in zip(
            expected_inputs, actual_inputs
        ):
            torch.testing.assert_close(actual_attn, expected_attn, rtol=0, atol=0)
            torch.testing.assert_close(actual_mlp, expected_mlp, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
