#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kv_scale_utils",
    REPO_ROOT / "python" / "sglang" / "srt" / "layers" / "quantization" / "kv_scale_utils.py",
)
kv_scale_utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kv_scale_utils
assert SPEC.loader is not None
SPEC.loader.exec_module(kv_scale_utils)


def create_dummy_scale_weights(layer: torch.nn.Module) -> None:
    layer.k_scale = torch.nn.Parameter(
        torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
    )
    layer.v_scale = torch.nn.Parameter(
        torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
    )
    layer.k_scale_per_head = None
    layer.v_scale_per_head = None
    layer.k_scale.weight_loader = kv_scale_utils.make_kv_cache_scale_loader(
        layer, "k_scale_per_head"
    )
    layer.v_scale.weight_loader = kv_scale_utils.make_kv_cache_scale_loader(
        layer, "v_scale_per_head"
    )


class TestFp8KvRuntimeScaleSupport(unittest.TestCase):
    def test_divide_kv_cache_by_scalar_scale_in_place(self):
        cache = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        out = kv_scale_utils.divide_kv_cache_by_scale_(cache, 2.0)
        self.assertIs(out, cache)
        self.assertTrue(torch.allclose(cache, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))

    def test_divide_kv_cache_by_per_head_scale_for_shaped_cache(self):
        cache = torch.ones(3, 2, 4)
        kv_scale_utils.divide_kv_cache_by_scale_(cache, torch.tensor([2.0, 4.0]))
        self.assertTrue(torch.allclose(cache[:, 0, :], torch.full((3, 4), 0.5)))
        self.assertTrue(torch.allclose(cache[:, 1, :], torch.full((3, 4), 0.25)))

    def test_divide_kv_cache_by_per_head_scale_for_flat_cache(self):
        cache = torch.ones(3, 8)
        kv_scale_utils.divide_kv_cache_by_scale_(
            cache, torch.tensor([2.0, 4.0]), num_heads=2, head_dim=4
        )
        self.assertTrue(torch.allclose(cache[:, :4], torch.full((3, 4), 0.5)))
        self.assertTrue(torch.allclose(cache[:, 4:], torch.full((3, 4), 0.25)))

    def test_multiply_kv_cache_by_per_head_scale_for_shaped_cache(self):
        cache = torch.ones(3, 2, 4)
        kv_scale_utils.multiply_kv_cache_by_scale_(cache, torch.tensor([2.0, 4.0]))
        self.assertTrue(torch.allclose(cache[:, 0, :], torch.full((3, 4), 2.0)))
        self.assertTrue(torch.allclose(cache[:, 1, :], torch.full((3, 4), 4.0)))

    def test_per_head_scale_roundtrip_for_non_flashinfer_fallback(self):
        original = torch.randn(5, 2, 4)
        cache = original.clone()
        scale = torch.tensor([0.5, 2.0])
        kv_scale_utils.divide_kv_cache_by_scale_(cache, scale)
        kv_scale_utils.multiply_kv_cache_by_scale_(cache, scale)
        self.assertTrue(torch.allclose(cache, original, atol=1e-6, rtol=1e-6))

    def test_divide_kv_cache_rejects_unbroadcastable_scale(self):
        cache = torch.ones(3, 7)
        with self.assertRaisesRegex(ValueError, "cannot infer head_dim"):
            kv_scale_utils.divide_kv_cache_by_scale_(cache, torch.tensor([2.0, 4.0]))

    def test_divide_kv_cache_rejects_wrong_per_head_count(self):
        cache = torch.ones(3, 2, 4)
        with self.assertRaisesRegex(ValueError, "has 3 heads but num_heads=2"):
            kv_scale_utils.divide_kv_cache_by_scale_(
                cache, torch.tensor([1.0, 2.0, 3.0]), num_heads=2
            )

    def test_scale_flat_query_expands_kv_scale_to_gqa_heads(self):
        q = torch.ones(2, 8)
        out = kv_scale_utils.scale_flat_query_for_per_head_k(
            q, torch.tensor([2.0, 4.0]), num_q_heads=4, head_dim=2
        )
        self.assertTrue(torch.allclose(out[:, 0:4], torch.full((2, 4), 2.0)))
        self.assertTrue(torch.allclose(out[:, 4:8], torch.full((2, 4), 4.0)))

    def test_scale_grouped_output_reconstructs_split_head_groups(self):
        grouped = torch.ones(4, 2, 2)
        out = kv_scale_utils.scale_grouped_output_for_per_head_v(
            grouped, torch.tensor([3.0, 5.0]), num_q_heads=4, v_head_dim=2
        )
        self.assertEqual(tuple(out.shape), (2, 8))
        self.assertTrue(torch.allclose(out[:, 0:4], torch.full((2, 4), 3.0)))
        self.assertTrue(torch.allclose(out[:, 4:8], torch.full((2, 4), 5.0)))

    def test_scale_grouped_output_preserves_interleaved_head_group_order(self):
        grouped = torch.tensor(
            [
                [[10.0], [20.0]],  # token 0, q heads 0-1
                [[30.0], [40.0]],  # token 0, q heads 2-3
                [[50.0], [60.0]],  # token 1, q heads 0-1
                [[70.0], [80.0]],  # token 1, q heads 2-3
            ]
        )
        out = kv_scale_utils.scale_grouped_output_for_per_head_v(
            grouped, torch.tensor([2.0, 3.0]), num_q_heads=4, v_head_dim=1
        )
        expected = torch.tensor(
            [
                [20.0, 40.0, 90.0, 120.0],
                [100.0, 120.0, 210.0, 240.0],
            ]
        )
        self.assertTrue(torch.allclose(out, expected))

    def test_per_head_scale_rejects_non_divisible_gqa_mapping(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            kv_scale_utils.expand_kv_scale_to_q_heads(torch.tensor([1.0, 2.0]), 3)

    def test_kv_cache_method_loads_scalar_scales_unchanged(self):
        layer = torch.nn.Module()
        create_dummy_scale_weights(layer)

        layer.k_scale.weight_loader(layer.k_scale, torch.tensor(0.25))
        layer.v_scale.weight_loader(layer.v_scale, torch.tensor(0.5))
        kv_scale_utils.finalize_kv_cache_scales(layer, fp8_fnuz=False)

        self.assertIsNone(layer.k_scale_per_head)
        self.assertIsNone(layer.v_scale_per_head)
        self.assertAlmostEqual(layer.k_scale_float, 0.25)
        self.assertAlmostEqual(layer.v_scale_float, 0.5)

    def test_kv_cache_method_preserves_per_head_scales_with_scalar_fallback(self):
        layer = torch.nn.Module()
        create_dummy_scale_weights(layer)

        layer.k_scale.weight_loader(layer.k_scale, torch.tensor([0.25, 0.5]))
        layer.v_scale.weight_loader(layer.v_scale, torch.tensor([1.0, 2.0]))
        kv_scale_utils.finalize_kv_cache_scales(layer, fp8_fnuz=False)

        self.assertEqual(layer.k_scale_per_head.tolist(), [0.25, 0.5])
        self.assertEqual(layer.v_scale_per_head.tolist(), [1.0, 2.0])
        self.assertAlmostEqual(layer.k_scale_float, 0.5)
        self.assertAlmostEqual(layer.v_scale_float, 2.0)
        self.assertAlmostEqual(layer.k_scale.item(), 0.5)
        self.assertAlmostEqual(layer.v_scale.item(), 2.0)

    def test_kv_cache_method_duplicates_single_per_head_kv_scale(self):
        layer = torch.nn.Module()
        create_dummy_scale_weights(layer)

        layer.k_scale.weight_loader(layer.k_scale, torch.tensor([0.25, 0.5]))
        kv_scale_utils.finalize_kv_cache_scales(layer, fp8_fnuz=False)

        self.assertEqual(layer.k_scale_per_head.tolist(), [0.25, 0.5])
        self.assertEqual(layer.v_scale_per_head.tolist(), [0.25, 0.5])
        self.assertAlmostEqual(layer.k_scale_float, 0.5)
        self.assertAlmostEqual(layer.v_scale_float, 0.5)


if __name__ == "__main__":
    unittest.main()
