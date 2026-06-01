#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kv_calibrate", REPO_ROOT / "scripts" / "kv_calibrate.py"
)
kv_calibrate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(kv_calibrate)


class TestKvCalibrateHeadStats(unittest.TestCase):
    def test_compute_per_head_absmax_from_flattened_projection(self):
        tensor = torch.tensor(
            [
                [1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
                [-7.0, 1.0, -2.0, 8.0, -3.0, 4.0],
            ]
        )
        result = kv_calibrate.compute_per_head_absmax(
            tensor, num_kv_heads=2, head_dim=3
        )
        self.assertEqual(result, [7.0, 8.0])

    def test_update_projection_stats_accumulates_per_head_max(self):
        stats = {
            "k_max": 0.0,
            "v_max": 0.0,
            "k_head_max": None,
            "v_head_max": None,
            "n_samples": 0,
        }
        kv_calibrate.update_projection_stats(
            stats,
            "k",
            torch.tensor([[1.0, -2.0, 10.0, -1.0]]),
            num_kv_heads=2,
            head_dim=2,
        )
        kv_calibrate.update_projection_stats(
            stats,
            "k",
            torch.tensor([[-3.0, 4.0, 2.0, -5.0]]),
            num_kv_heads=2,
            head_dim=2,
        )
        self.assertEqual(stats["k_max"], 10.0)
        self.assertEqual(stats["k_head_max"], [4.0, 10.0])

    def test_build_head_report_quantifies_scalar_waste(self):
        report = kv_calibrate.build_head_report([2.0, 10.0, 4.0], safe_max=2.0)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["scale"], [1.0, 5.0, 2.0])
        self.assertAlmostEqual(report["max_over_min_nonzero_abs"], 5.0)
        self.assertAlmostEqual(report["scalar_over_median_head_scale"], 2.5)
        self.assertAlmostEqual(report["median_head_utilization_under_scalar"], 0.4)

    def test_build_scale_tensor_values_defaults_to_scalar(self):
        stats = {"k_max": 10.0, "k_head_max": [2.0, 10.0, 4.0]}
        values, report, granularity = kv_calibrate.build_scale_tensor_values(
            stats, "k", safe_max=2.0, emit_per_head=False
        )
        self.assertEqual(values, [5.0])
        self.assertIsNotNone(report)
        self.assertEqual(granularity, "scalar")

    def test_build_scale_tensor_values_can_emit_per_head(self):
        stats = {"k_max": 10.0, "k_head_max": [2.0, 10.0, 4.0]}
        values, report, granularity = kv_calibrate.build_scale_tensor_values(
            stats, "k", safe_max=2.0, emit_per_head=True
        )
        self.assertEqual(values, [1.0, 5.0, 2.0])
        self.assertIsNotNone(report)
        self.assertEqual(granularity, "per_head")


if __name__ == "__main__":
    unittest.main()
