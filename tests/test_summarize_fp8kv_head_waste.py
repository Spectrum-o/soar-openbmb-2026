#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_fp8kv_head_waste",
    REPO_ROOT / "tools" / "summarize_fp8kv_head_waste.py",
)
summary_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary_tool
assert SPEC.loader is not None
SPEC.loader.exec_module(summary_tool)


class TestSummarizeFp8KvHeadWaste(unittest.TestCase):
    def sample_report(self):
        return {
            "layers": {
                "0": {
                    "k_scale": 2.0,
                    "v_scale": 4.0,
                    "n_samples": 64,
                    "k_per_head": {
                        "median_head_utilization_under_scalar": 0.25,
                        "scalar_over_median_head_scale": 4.0,
                        "max_over_min_nonzero_abs": 8.0,
                    },
                    "v_per_head": {
                        "median_head_utilization_under_scalar": 0.80,
                        "scalar_over_median_head_scale": 1.25,
                        "max_over_min_nonzero_abs": 2.0,
                    },
                },
                "9": {
                    "k_scale": 1.0,
                    "n_samples": 64,
                    "k_per_head": {
                        "median_head_utilization_under_scalar": 0.55,
                        "scalar_over_median_head_scale": 1.818,
                    },
                },
            }
        }

    def test_collect_projection_waste_sorts_by_worst_utilization(self):
        rows = summary_tool.collect_projection_waste(self.sample_report())
        self.assertEqual([(row.layer, row.projection) for row in rows], [(0, "k"), (9, "k"), (0, "v")])
        self.assertAlmostEqual(rows[0].median_util, 0.25)
        self.assertAlmostEqual(rows[0].scalar_over_median, 4.0)
        self.assertEqual(rows[0].n_samples, 64)

    def test_classify_recommends_runtime_work_when_severe(self):
        rows = summary_tool.collect_projection_waste(self.sample_report())
        self.assertEqual(
            summary_tool.classify(rows, severe_util=0.5),
            "runtime_scale_work_worth_testing",
        )

    def test_classify_handles_missing_per_head_stats(self):
        self.assertEqual(summary_tool.classify([], severe_util=0.5), "missing_per_head_stats")

    def test_json_report_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text('{"layers": {"0": {}}}', encoding="utf-8")
            self.assertEqual(summary_tool.load_report(path), {"layers": {"0": {}}})


if __name__ == "__main__":
    unittest.main()
