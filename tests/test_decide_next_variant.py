#!/usr/bin/env python3
"""Unit tests for scripts/decide_next_variant.py.

Run from repo root:
    python3 -m unittest tests.test_decide_next_variant -v

Mirrors test_parse_quant_diagnostic.py: unittest + tempfile + stdlib.
Each test builds a synthetic platform log fragment, runs decide(),
and asserts the recommendation matches the expected V24_PLAN.md
decision tree branch.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import decide_next_variant as a module (it's in scripts/ not tools/)
_spec = importlib.util.spec_from_file_location(
    "decide_next_variant",
    REPO_ROOT / "scripts" / "decide_next_variant.py",
)
_dnv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dnv)

from tools import parse_quant_diagnostic as pqd  # noqa: E402


def _make_log(content: str) -> Path:
    """Write content to a tempfile, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Synthetic log fragments matching different V24_PLAN.md branches
# ---------------------------------------------------------------------------
SUCCESS_FRAGMENT = """\
[prepare_env] start 2026-05-22 03:00:00
[versions] python=3.10.12
[versions] torch=2.9.1
[versions] transformers=4.57.1
[versions] gptqmodel=7.0.0
[prepare_env] forcing gptqmodel==7.0.0 (currently: 7.0.0)
[prepare_model] DIAGNOSTIC: calib jsonl path=/sub/perf_public_set.jsonl
[prepare_model] DIAGNOSTIC: calib jsonl size=23925601 bytes
[prepare_model] DIAGNOSTIC: calib jsonl row count=150
[quantize] saving to /tmp/out
[qzeros-fix] output_dir inventory (/tmp/out):
[qzeros-fix]   model-00001-of-00003.safetensors  (4,983,672,832 bytes)
[qzeros-fix] scanning 3 .safetensors weight file(s)
[qzeros-fix] SUMMARY:
[qzeros-fix]   qzeros tensors seen:        96
[qzeros-fix]   patched (had 0x77777777):   96
[qzeros-fix] POST-CHECK model-00002.safetensors::layers.0.mlp.gate_proj.qzeros first_val=-2004318072 (0x88888888)
[qzeros-fix] OK
[quantize] done.
"""

ZERO_OLD_TRANSFORMERS_FRAGMENT = """\
[versions] python=3.10.12
[versions] torch=2.9.1
[versions] transformers=4.46.0
[versions] gptqmodel=7.0.0
[prepare_env] gptqmodel 7.0.0 already installed; skipping PyPI download
[prepare_model] DIAGNOSTIC: calib jsonl size=23925601 bytes
[prepare_model] DIAGNOSTIC: calib jsonl row count=150
[quantize] saving to /tmp/out
[qzeros-fix] output_dir inventory (/tmp/out):
[qzeros-fix]   model-00001-of-00003.safetensors  (4,983,672,832 bytes)
[qzeros-fix] scanning 3 .safetensors weight file(s)
[qzeros-fix] SUMMARY:
[qzeros-fix]   qzeros tensors seen:        96
[qzeros-fix]   patched (had 0x77777777):   96
[qzeros-fix] OK
[quantize] done.
"""

ZERO_NEW_TRANSFORMERS_FRAGMENT = """\
[versions] python=3.10.12
[versions] torch=2.9.1
[versions] transformers=4.57.1
[versions] gptqmodel=7.0.0
[qzeros-fix] SUMMARY:
[qzeros-fix]   qzeros tensors seen:        96
[qzeros-fix]   patched (had 0x77777777):   96
[qzeros-fix] OK
[quantize] done.
"""

QZEROS_FATAL_FRAGMENT = """\
[versions] python=3.10.12
[versions] transformers=4.57.1
[versions] gptqmodel=7.0.5
[quantize] saving to /tmp/out
[qzeros-fix] output_dir inventory (/tmp/out):
[qzeros-fix]   model.safetensors  (6,536,000,000 bytes)
[qzeros-fix] scanning 1 .safetensors weight file(s)
[qzeros-fix] SUMMARY:
[qzeros-fix]   qzeros tensors seen:        0
[qzeros-fix] FATAL: zero .qzeros tensors found in any shard.
"""

CRASHED_FRAGMENT = """\
[versions] python=3.10.12
[versions] transformers=4.57.1
[prepare_env] patched minicpm_backend.py for fp16 quantized run
Traceback (most recent call last):
  File "...", line 123, in forward
RuntimeError: query and key must have the same dtype
"""

PARTIAL_FRAGMENT = ZERO_NEW_TRANSFORMERS_FRAGMENT  # same log structure


class TestDecisionTree(unittest.TestCase):
    def _decide(self, log_text: str, acc: float | None):
        log = _make_log(log_text)
        parsed = pqd.parse_log(log)
        return _dnv.decide(parsed, acc)

    def test_high_acc_recommends_v24_perf(self):
        variant, _, _ = self._decide(SUCCESS_FRAGMENT, acc=45.0)
        self.assertEqual(variant, "v24_perf")

    def test_moderate_acc_partial_recommends_calib_multi_adaptive(self):
        # 0 < acc < 30 → v25_calib_multi_adaptive per the decision tree
        variant, _, _ = self._decide(PARTIAL_FRAGMENT, acc=20.0)
        self.assertEqual(variant, "v25_calib_multi_adaptive")

    def test_zero_acc_old_transformers_recommends_pin_transformers(self):
        variant, _, _ = self._decide(ZERO_OLD_TRANSFORMERS_FRAGMENT, acc=0.0)
        self.assertEqual(variant, "v24_pin_transformers")

    def test_zero_acc_new_transformers_recommends_no_dtype_key(self):
        variant, _, _ = self._decide(ZERO_NEW_TRANSFORMERS_FRAGMENT, acc=0.0)
        self.assertEqual(variant, "v24_no_dtype_key")

    def test_qzeros_fatal_recommends_manual_fix(self):
        variant, recommendation, _ = self._decide(QZEROS_FATAL_FRAGMENT, acc=0.0)
        self.assertEqual(variant, "MANUAL_FIX")
        self.assertIn("FATAL", recommendation)

    def test_crash_before_qzeros_recommends_manual_fix(self):
        # Has RuntimeError in the FATAL list, no qzeros block at all
        variant, recommendation, _ = self._decide(CRASHED_FRAGMENT, acc=None)
        self.assertEqual(variant, "MANUAL_FIX")

    def test_high_acc_at_boundary_30_recommends_v24_perf(self):
        variant, _, _ = self._decide(SUCCESS_FRAGMENT, acc=30.0)
        self.assertEqual(variant, "v24_perf")

    def test_low_acc_at_boundary_29_recommends_calib_multi_adaptive(self):
        variant, _, _ = self._decide(PARTIAL_FRAGMENT, acc=29.0)
        self.assertEqual(variant, "v25_calib_multi_adaptive")


class TestExtractAccFromLog(unittest.TestCase):
    def test_extract_acc_ori_json_form(self):
        text = '''
        Some preamble.
        Score: {
          "acc": 0.0,
          "acc_ori": 0.0,
          "final_score": 0.0
        }
        '''
        self.assertEqual(_dnv.extract_acc_from_log(text), 0.0)

    def test_extract_acc_ori_dot_form(self):
        text = "...acc_ori = 47.5..."
        self.assertEqual(_dnv.extract_acc_from_log(text), 47.5)

    def test_extract_average_score(self):
        text = "Average Score 47.24"
        self.assertEqual(_dnv.extract_acc_from_log(text), 47.24)

    def test_no_match_returns_none(self):
        text = "completely unrelated text"
        self.assertIsNone(_dnv.extract_acc_from_log(text))


class TestDecisionTreeRobustness(unittest.TestCase):
    """The platform UI may not include the full prepare_env stdout in
    the downloadable log. Sometimes you only get the final state machine
    lines + Score JSON. decide() should fail gracefully without crashing."""

    MINIMAL_PLATFORM_LOG_SUCCESS = """\
[2026-05-22 03:30:00] [PENDING]    task accepted
[2026-05-22 03:30:01] [PREPARING]
[2026-05-22 03:32:15] [INFERENCING]
[2026-05-22 06:30:00] [SUCCESS]
Score:
{
  "acc": 42.5,
  "acc_ori": 42.5,
  "final_score": 53.13,
  "benchmark_duration": {"S1": 626, "S8": 998, "Smax": 2290}
}
"""

    MINIMAL_PLATFORM_LOG_ZERO = """\
[2026-05-22 03:30:00] [PENDING]
[2026-05-22 03:30:01] [PREPARING]
[2026-05-22 06:30:00] [SUCCESS]
Score:
{
  "acc": 0.0,
  "acc_ori": 0.0,
  "final_score": 0.0
}
"""

    def _decide_from_text(self, log_text: str, acc: float | None = None):
        log = _make_log(log_text)
        parsed = pqd.parse_log(log)
        # If acc was not passed, extract from the text
        if acc is None:
            acc = _dnv.extract_acc_from_log(log_text)
        return _dnv.decide(parsed, acc), parsed

    def test_minimal_success_log_still_recommends_perf(self):
        # Even without [versions] / [qzeros-fix] blocks, acc=42.5 >= 30
        # should route to v24_perf.
        (variant, _, _), parsed = self._decide_from_text(
            self.MINIMAL_PLATFORM_LOG_SUCCESS
        )
        self.assertEqual(variant, "v24_perf")
        # And the parsed log has no transformers version (graceful degrade)
        self.assertNotIn("transformers", parsed.versions)

    def test_minimal_zero_log_routes_to_some_variant_or_manual(self):
        # No [qzeros-fix] section in this minimal log at all → qzeros_fix
        # state is "unfinished" (not ended_ok, no fatal). Branch 4
        # (pipeline crashed early) triggers → MANUAL_FIX.
        (variant, _, _), parsed = self._decide_from_text(
            self.MINIMAL_PLATFORM_LOG_ZERO
        )
        self.assertIn(variant, ("MANUAL_FIX", "v24_no_dtype_key", "v24_pin_transformers"),
                      f"unexpected variant {variant} for minimal zero log")
        # acc was extracted from the JSON
        self.assertNotIn("transformers", parsed.versions)

    def test_acc_extracted_from_platform_json_form(self):
        # Sanity: the Score JSON form is parseable
        log_text = self.MINIMAL_PLATFORM_LOG_SUCCESS
        acc = _dnv.extract_acc_from_log(log_text)
        self.assertEqual(acc, 42.5)

    def test_empty_log_does_not_crash(self):
        (variant, recommendation, _), _ = self._decide_from_text("")
        self.assertIn(variant, ("MANUAL_FIX", "UNCLEAR"))
        self.assertTrue(recommendation)  # non-empty message


class TestVariantDirInventory(unittest.TestCase):
    """Ensure VARIANTS dict references variants that exist on disk."""

    def test_all_referenced_dirs_exist(self):
        for variant_key, dirname in _dnv.VARIANTS.items():
            path = REPO_ROOT / dirname
            self.assertTrue(
                path.is_dir(),
                f"VARIANTS['{variant_key}'] = '{dirname}' but {path} doesn't exist"
            )

    def test_v24_perf_recommended_variant_is_built(self):
        # The variant most likely to be recommended (v23 success case)
        # must exist on disk.
        self.assertIn("v24_perf", _dnv.VARIANTS)
        path = REPO_ROOT / _dnv.VARIANTS["v24_perf"]
        self.assertTrue(path.is_dir(), f"v24_perf variant missing at {path}")


if __name__ == "__main__":
    unittest.main()
