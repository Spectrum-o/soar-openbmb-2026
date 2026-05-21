#!/usr/bin/env python3
"""Unit tests for the calibration / analysis tools.

Run from repo root:
    python3 -m unittest tests.test_calibration_tools -v

Tests are stdlib-only (unittest + tempfile). They cover:
  - slice_windows_for_prompt: the windowing policy used by both
    `quantize_gptqmodel_w4a16.py` (production) and
    `tools/calib_set_preview.py` (preview)
  - build_calib_set helpers: bucket_for, cycle_to_size, sample_uniform_mix
  - parse_experiment_logs: regex anchors against fixture log lines
  - analyze_predictions: pass/fail classification + per-task summary
  - quant_config_validator: file-presence + grep-based checks against
    a synthesized fake variant tree

If you change `_slice_windows_for_prompt` in the quantize script, update
this test to match (and update `tools/calib_set_preview.py` too).
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

# Add repo root to sys.path so `tools.xxx` imports work.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import tool modules
from tools import calib_set_preview, build_calib_set, parse_experiment_logs, analyze_predictions, quant_config_validator  # noqa: E402


# ---------------------------------------------------------------------------
# Window slicing — matches `_slice_windows_for_prompt` in the quantize script
# ---------------------------------------------------------------------------
class TestSliceWindows(unittest.TestCase):
    L = 8192

    def test_short(self):
        """N <= L: single window covering everything."""
        rng = random.Random(0)
        self.assertEqual(
            calib_set_preview.slice_windows_for_prompt(0, self.L, rng), [(0, 0)]
        )
        self.assertEqual(
            calib_set_preview.slice_windows_for_prompt(1000, self.L, rng),
            [(0, 1000)],
        )
        self.assertEqual(
            calib_set_preview.slice_windows_for_prompt(self.L, self.L, rng),
            [(0, self.L)],
        )

    def test_medium(self):
        """L < N <= 4L: single tail window."""
        rng = random.Random(0)
        for n in (self.L + 1, 2 * self.L, 4 * self.L):
            ws = calib_set_preview.slice_windows_for_prompt(n, self.L, rng)
            self.assertEqual(len(ws), 1)
            s, e = ws[0]
            self.assertEqual(e - s, self.L)
            self.assertEqual(e, n)  # tail-aligned

    def test_long(self):
        """4L < N <= 12.5L: tail + centered mid."""
        rng = random.Random(0)
        n = 8 * self.L
        ws = calib_set_preview.slice_windows_for_prompt(n, self.L, rng)
        self.assertEqual(len(ws), 2)
        # tail
        self.assertEqual(ws[0], (n - self.L, n))
        # mid (centered)
        mid_start = (n - self.L) // 2
        self.assertEqual(ws[1], (mid_start, mid_start + self.L))

    def test_super_long(self):
        """N > 12.5L: tail + mid + random."""
        rng = random.Random(42)
        n = 20 * self.L
        ws = calib_set_preview.slice_windows_for_prompt(n, self.L, rng)
        self.assertEqual(len(ws), 3)
        # Tail and mid are deterministic
        self.assertEqual(ws[0], (n - self.L, n))
        mid_start = (n - self.L) // 2
        self.assertEqual(ws[1], (mid_start, mid_start + self.L))
        # Random window: must be size L, in-range, not overlapping tail
        s, e = ws[2]
        self.assertEqual(e - s, self.L)
        self.assertGreaterEqual(s, self.L)
        self.assertLessEqual(e, n - self.L)

    def test_no_overlap_in_super_long(self):
        """The random mid window must not overlap tail or centered mid."""
        rng = random.Random(42)
        for seed in range(20):
            rng = random.Random(seed)
            n = 13 * self.L
            ws = calib_set_preview.slice_windows_for_prompt(n, self.L, rng)
            # check pairwise non-overlap (sorted by start)
            spans = sorted(ws)
            for i in range(len(spans) - 1):
                self.assertLessEqual(
                    spans[i][1],
                    spans[i + 1][0],
                    f"overlap at seed={seed}: {spans}",
                )

    def test_in_bounds(self):
        """All windows must be within [0, n]."""
        rng = random.Random(1)
        for n in (1, 100, self.L, 5 * self.L, 13 * self.L, 100 * self.L):
            ws = calib_set_preview.slice_windows_for_prompt(n, self.L, rng)
            for s, e in ws:
                self.assertGreaterEqual(s, 0)
                self.assertLessEqual(e, n)
                self.assertGreaterEqual(e, s)


# ---------------------------------------------------------------------------
# build_calib_set helpers
# ---------------------------------------------------------------------------
class TestBuildCalibSet(unittest.TestCase):
    BINS = [8192, 32768, 102400]

    def test_bucket_for_edges(self):
        """Boundary correctness for bucket assignment."""
        self.assertEqual(build_calib_set.bucket_for(0, self.BINS), "short")
        self.assertEqual(build_calib_set.bucket_for(8192, self.BINS), "short")
        self.assertEqual(build_calib_set.bucket_for(8193, self.BINS), "medium")
        self.assertEqual(build_calib_set.bucket_for(32768, self.BINS), "medium")
        self.assertEqual(build_calib_set.bucket_for(32769, self.BINS), "long")
        self.assertEqual(build_calib_set.bucket_for(102400, self.BINS), "long")
        self.assertEqual(build_calib_set.bucket_for(102401, self.BINS), "super")
        self.assertEqual(build_calib_set.bucket_for(10**6, self.BINS), "super")

    def test_cycle_to_size_already_large(self):
        rng = random.Random(0)
        rows = [{"i": i} for i in range(100)]
        out = build_calib_set.cycle_to_size(rows, 50, rng)
        self.assertEqual(len(out), 50)

    def test_cycle_to_size_under_filled(self):
        """When source is smaller than target, deterministically cycle."""
        rng = random.Random(0)
        rows = [{"i": i} for i in range(10)]
        out = build_calib_set.cycle_to_size(rows, 30, rng)
        self.assertEqual(len(out), 30)
        # Each original row should appear at least 2 times (shuffled then cycled)
        counts: dict[int, int] = {}
        for r in out:
            counts[r["i"]] = counts.get(r["i"], 0) + 1
        for i in range(10):
            self.assertGreaterEqual(counts[i], 2)

    def test_cycle_to_size_empty(self):
        rng = random.Random(0)
        self.assertEqual(build_calib_set.cycle_to_size([], 50, rng), [])

    def test_sample_uniform_mix(self):
        rng = random.Random(0)
        buckets = {
            "short": [{"b": "short", "i": i} for i in range(10)],
            "medium": [{"b": "medium", "i": i} for i in range(20)],
            "long": [{"b": "long", "i": i} for i in range(30)],
            "super": [],  # empty bucket
        }
        out = build_calib_set.sample_uniform_mix(buckets, 18, rng)
        # 3 non-empty buckets, target 18 → 6 per bucket
        from collections import Counter
        counts = Counter(r["b"] for r in out)
        self.assertEqual(set(counts.keys()), {"short", "medium", "long"})
        for c in counts.values():
            self.assertLessEqual(c, 6)

    def test_extract_text_fields(self):
        self.assertEqual(build_calib_set.extract_text({"question": "abc"}), "abc")
        self.assertEqual(build_calib_set.extract_text({"prompt": "def"}), "def")
        with self.assertRaises(KeyError):
            build_calib_set.extract_text({"no_text_field": "x"})


# ---------------------------------------------------------------------------
# parse_experiment_logs regex
# ---------------------------------------------------------------------------
class TestParseExperimentLogs(unittest.TestCase):
    def test_average_score(self):
        line = "Average Score: 49.00%"
        m = parse_experiment_logs._RE["average_score"].search(line)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "49.00")

    def test_generation_time(self):
        line = "Generation completed in 2383.17 seconds"
        m = parse_experiment_logs._RE["generation_time"].search(line)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "2383.17")

    def test_tokens_in_out(self):
        line = "Total Tokens: In=8644316, Out=1326812"
        m_in = parse_experiment_logs._RE["total_in_toks"].search(line)
        m_out = parse_experiment_logs._RE["total_out_toks"].search(line)
        self.assertEqual(m_in.group(1), "8644316")
        self.assertEqual(m_out.group(1), "1326812")

    def test_calib_chat_off(self):
        line = "[prepare_model] chat template DISABLED (DISABLE_CHAT_TEMPLATE=1)"
        self.assertIsNotNone(parse_experiment_logs._RE["calib_chat_off"].search(line))

    def test_calib_window_mode(self):
        line = "[calib] multi-adaptive: 150 prompts -> 270 windows (L=8192); per-bucket: ..."
        m = parse_experiment_logs._RE["calib_window"].search(line)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "150")
        self.assertEqual(m.group(2), "270")

    def test_quant_start(self):
        line = "[quantize] starting GPTQModel W4A16 group_size=128 samples=256 max_len=8192"
        m = parse_experiment_logs._RE["quant_start"].search(line)
        self.assertEqual(m.group(1), "4")
        self.assertEqual(m.group(2), "128")
        self.assertEqual(m.group(3), "256")
        self.assertEqual(m.group(4), "8192")

    def test_qzeros_patched(self):
        line = "[qzeros-fix] total 96 qzeros tensors patched 0x77777777 -> 0x88888888"
        m = parse_experiment_logs._RE["qzeros_patched"].search(line)
        self.assertEqual(m.group(1), "96")

    def test_filename_local_eval(self):
        out = parse_experiment_logs.parse_filename(
            Path("local_eval_eval_submission_gptqmodel_calib_w4a16_1779291723.log")
        )
        self.assertEqual(out["stage"], "eval")
        self.assertEqual(out["variant"], "submission_gptqmodel_calib_w4a16")
        self.assertEqual(out["timestamp"], "1779291723")

    def test_filename_tee(self):
        out = parse_experiment_logs.parse_filename(
            Path("expB_no_chat_template_20260521_011939.log")
        )
        # Our pattern just matches "expB" as the variant name; that's fine.
        self.assertEqual(out["stage"], "tee")


# ---------------------------------------------------------------------------
# analyze_predictions classification
# ---------------------------------------------------------------------------
class TestAnalyzePredictions(unittest.TestCase):
    def test_pass_classification(self):
        self.assertTrue(analyze_predictions.is_pass({"score": 1}))
        self.assertTrue(analyze_predictions.is_pass({"score": 0.7}))
        self.assertTrue(analyze_predictions.is_pass({"score": 0.5}))
        self.assertFalse(analyze_predictions.is_pass({"score": 0.49}))
        self.assertFalse(analyze_predictions.is_pass({"score": 0}))
        self.assertFalse(analyze_predictions.is_pass({}))  # missing
        self.assertFalse(analyze_predictions.is_pass({"score": "garbage"}))

    def test_fail_classification(self):
        self.assertTrue(analyze_predictions.is_fail({"score": 0}))
        self.assertTrue(analyze_predictions.is_fail({"score": 0.3}))
        self.assertFalse(analyze_predictions.is_fail({"score": 0.6}))
        self.assertFalse(analyze_predictions.is_fail({"score": 1}))
        # Missing or unparseable scores are NOT counted as failed:
        self.assertFalse(analyze_predictions.is_fail({}))
        self.assertFalse(analyze_predictions.is_fail({"score": "garbage"}))


# ---------------------------------------------------------------------------
# quant_config_validator: synthesize a tiny valid + tiny broken variant
# and exercise the checkers.
# ---------------------------------------------------------------------------
class TestQuantConfigValidator(unittest.TestCase):
    def _write_valid_variant(self, root: Path) -> None:
        (root / "prepare_env.sh").write_text(
            'export SGLANG_SERVER_ARGS="--disable-radix-cache '
            '--attention-backend minicpm_flashinfer '
            '--quantization gptq_marlin --dtype float16"\n'
            "# fp16 sed-patch\n"
            "sed -i 's/torch\\.bfloat16/torch.float16/g' x.py\n"
            "sed -i 's/\"bfloat16\"/\"float16\"/g' x.py\n"
        )
        (root / "prepare_model.sh").write_text(
            'BUNDLED_CALIB="${SCRIPT_DIR}/perf_public_set.jsonl"\n'
            'QUANT_TIMEOUT_MIN="${QUANT_TIMEOUT_MIN:-90}"\n'
            "EXTRA_ARGS+=(--no-offload-disk)\n"
            "CALIB_WINDOW_MODE=\"${CALIB_WINDOW_MODE:-tail}\"\n"
            "DISABLE_CHAT_TEMPLATE=\"${DISABLE_CHAT_TEMPLATE:-0}\"\n"
        )
        (root / "quantize_gptqmodel_w4a16.py").write_text(
            "def fix_qzeros_for_marlin():\n"
            "    pass\n"
            "def register_minicpm_sala_with_gptqmodel():\n"
            "    pass\n"
            'def tokenize_calibration():\n'
            '    tokenizer.truncation_side = "left"\n'
            "fix_qzeros_for_marlin()\n"
            "register_minicpm_sala_with_gptqmodel()\n"
            "# --no-chat-template support\n"
            "# multi-adaptive support via _slice_windows_for_prompt\n"
        )
        (root / "perf_public_set.jsonl").write_text("")
        # bundled sglang with sed targets
        sub = root / "sglang" / "python" / "sglang" / "srt" / "layers" / "attention"
        sub.mkdir(parents=True)
        (sub / "minicpm_backend.py").write_text(
            "import torch\n"
            "x = torch.bfloat16  # patch me\n"
            "y = \"bfloat16\"     # patch me\n"
        )
        (sub / "minicpm_sparse_utils.py").write_text("dtype = torch.bfloat16\n")
        (root / "flash_attn-test.whl").write_text("")  # marker file

    def test_valid_variant_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_valid_variant(root)
            n_pass, n_total, _ = quant_config_validator.run_checks(root)
            self.assertEqual(n_pass, n_total, f"expected all checks to pass, got {n_pass}/{n_total}")

    def test_missing_qzeros_fix_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_valid_variant(root)
            # Sabotage: remove the call site
            qf = root / "quantize_gptqmodel_w4a16.py"
            text = qf.read_text().replace("fix_qzeros_for_marlin()\n", "")
            qf.write_text(text)
            n_pass, n_total, _ = quant_config_validator.run_checks(root)
            self.assertLess(n_pass, n_total)

    def test_sed_targets_missing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_valid_variant(root)
            # Sabotage: remove the sed target file
            (root / "sglang" / "python" / "sglang" / "srt" / "layers" / "attention"
             / "minicpm_backend.py").unlink()
            n_pass, n_total, _ = quant_config_validator.run_checks(root)
            self.assertLess(n_pass, n_total)

    def test_sed_zero_matches_fails(self):
        """If sglang sources contain no bfloat16 lines, sed will silently
        no-op — this is the suspected v21 platform failure mode."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_valid_variant(root)
            # Sabotage: blank the sed-target files so 0 lines match
            sub = root / "sglang" / "python" / "sglang" / "srt" / "layers" / "attention"
            (sub / "minicpm_backend.py").write_text("# nothing to patch here\n")
            (sub / "minicpm_sparse_utils.py").write_text("# nothing to patch here\n")
            n_pass, n_total, _ = quant_config_validator.run_checks(root)
            self.assertLess(n_pass, n_total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
