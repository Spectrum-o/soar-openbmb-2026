#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fp8kv_per_head_bridge",
    REPO_ROOT / "tools" / "fp8kv_per_head_bridge.py",
)
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
assert SPEC.loader is not None
SPEC.loader.exec_module(bridge)


class TestFp8KvPerHeadBridge(unittest.TestCase):
    def test_q_head_scale_from_kv_scale_repeats_gqa_groups(self):
        scale = torch.tensor([2.0, 4.0])
        expanded = bridge.q_head_scale_from_kv_scale(scale, num_q_heads=6)
        self.assertEqual(expanded.tolist(), [2.0, 2.0, 2.0, 4.0, 4.0, 4.0])

    def test_cache_write_scale_supports_shaped_kv(self):
        kv = torch.ones(3, 2, 4)
        scale = torch.tensor([2.0, 4.0])
        out = bridge.scale_kv_for_cache_write(kv, scale)
        self.assertTrue(torch.allclose(out[:, 0, :], torch.full((3, 4), 0.5)))
        self.assertTrue(torch.allclose(out[:, 1, :], torch.full((3, 4), 0.25)))

    def test_cache_write_scale_supports_flat_kv(self):
        kv = torch.ones(3, 8)
        scale = torch.tensor([2.0, 4.0])
        out = bridge.scale_kv_for_cache_write(kv, scale, num_kv_heads=2, head_dim=4)
        self.assertTrue(torch.allclose(out[:, :4], torch.full((3, 4), 0.5)))
        self.assertTrue(torch.allclose(out[:, 4:], torch.full((3, 4), 0.25)))

    def test_bridge_attention_matches_direct_per_head_dequant_attention(self):
        torch.manual_seed(0)
        q = torch.randn(5, 4, 3, dtype=torch.float64)
        k_real = torch.randn(7, 2, 3, dtype=torch.float64)
        v_real = torch.randn(7, 2, 3, dtype=torch.float64)
        k_scale = torch.tensor([0.5, 2.0], dtype=torch.float64)
        v_scale = torch.tensor([4.0, 0.25], dtype=torch.float64)

        k_cache = bridge.scale_kv_for_cache_write(k_real, k_scale)
        v_cache = bridge.scale_kv_for_cache_write(v_real, v_scale)

        direct = bridge.reference_gqa_attention(q, k_real, v_real, sm_scale=0.7)
        bridged = bridge.bridge_attention_with_scaled_cache(
            q, k_cache, v_cache, k_scale, v_scale, sm_scale=0.7
        )

        self.assertTrue(torch.allclose(direct, bridged, atol=1e-10, rtol=1e-10))

    def test_load_per_head_scales_from_report(self):
        report = {
            "layers": {
                "9": {
                    "k_per_head": {"scale": [1.0, 2.0]},
                    "v_per_head": {"scale": [3.0, 4.0]},
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            scales = bridge.load_per_head_scales_from_report(path, layer=9)

        self.assertEqual(scales["k_scale"].tolist(), [1.0, 2.0])
        self.assertEqual(scales["v_scale"].tolist(), [3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
