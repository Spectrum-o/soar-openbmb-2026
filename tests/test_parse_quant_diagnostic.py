#!/usr/bin/env python3
"""Unit tests for tools/parse_quant_diagnostic.py.

Run from repo root:
    python3 -m unittest tests.test_parse_quant_diagnostic -v

Mirrors tests/test_calibration_tools.py: unittest + tempfile, stdlib only.
Covers:
  - New 2026-05-21 diagnostic block format (full happy path)
  - Backward-compat with legacy [qzeros-fix] total format
  - FATAL detection -> raised path captured
  - POST-CHECK parsing -> is_good flag for both 0x88 (good) and 0x77 (bug)
  - Sample qzeros parsing (the inner list-literal escapes correctly)
  - Diff mode: version drift surfaces; output-dir layout drift surfaces;
    heuristic verdict fires on qzeros-patched count mismatch
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import parse_quant_diagnostic as pqd  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic log fragments
# ---------------------------------------------------------------------------
NEW_FORMAT_LOG = """\
[prepare_env] start 2026-05-21 09:00:00
[prepare_env] submission dir: /root/autodl-tmp/sub
[prepare_env] gptqmodel 7.0.0 already installed; skipping PyPI download
[prepare_env] patched minicpm_backend.py for fp16 quantized run
[prepare_env] patched minicpm_sparse_utils.py for fp16 quantized run
[versions] python=3.10.12
[versions] torch=2.9.1
[versions] transformers=4.57.1
[versions] gptqmodel=7.0.0
[versions] flash-attn=2.8.3
[versions] sglang=0.5.6
[prepare_env] SGLANG_SERVER_ARGS=--disable-radix-cache --quantization gptq_marlin --dtype float16
[prepare_env] done
[prepare_model] DIAGNOSTIC: script dir contents
[prepare_model] DIAGNOSTIC: input dir
[prepare_model] DIAGNOSTIC: calib jsonl path=/root/autodl-tmp/sub/perf_public_set.jsonl
[prepare_model] DIAGNOSTIC: calib jsonl size=23925601 bytes
[prepare_model] DIAGNOSTIC: calib jsonl row count=150
[prepare_model] DIAGNOSTIC: calib jsonl first row (first 200 chars):
{"task": "niah", "question": "find the magic number...", "gold": "abc123"}
[prepare_model] quantize timeout: 90 min
... python3 quantize_gptqmodel_w4a16.py --input X --output Y --num-calib 256 --max-calib-len 8192 --calib-window-mode tail --no-offload-disk
[calib] applying chat template to calibration prompts
[quantize] starting GPTQModel W4A16 group_size=128 samples=256 max_len=8192
[quantize] saving to /root/autodl-fs/zyn/models/sub-quantized
[qzeros-fix] output_dir inventory (/root/autodl-fs/zyn/models/sub-quantized):
[qzeros-fix]   config.json  (2,468 bytes)
[qzeros-fix]   model-00001-of-00003.safetensors  (4,983,672,832 bytes)
[qzeros-fix]   model-00002-of-00003.safetensors  (3,213,440,128 bytes)
[qzeros-fix]   model-00003-of-00003.safetensors  (1,124,560,896 bytes)
[qzeros-fix] scanning 3 .safetensors weight file(s)
[qzeros-fix] model-00002-of-00003.safetensors: patched 71 qzeros tensors
[qzeros-fix] model-00003-of-00003.safetensors: patched 25 qzeros tensors
[qzeros-fix] SUMMARY:
[qzeros-fix]   qzeros tensors seen:        96
[qzeros-fix]   patched (had 0x77777777):   96
[qzeros-fix]   already good (0x88888888):  0
[qzeros-fix]   unknown dtype skipped:      0
[qzeros-fix]   unknown values not patched: 0
[qzeros-fix]   sample: model-00002-of-00003.safetensors::layers.0.mlp.gate_proj.qzeros dtype=torch.int32 unique[:6]=[2004318071] hex=['0x77777777']
[qzeros-fix]   sample: model-00002-of-00003.safetensors::layers.0.mlp.up_proj.qzeros dtype=torch.int32 unique[:6]=[2004318071] hex=['0x77777777']
[qzeros-fix] POST-CHECK model-00002-of-00003.safetensors::layers.0.mlp.gate_proj.qzeros first_val=-2004318072 (0x88888888)
[qzeros-fix] OK
[prepare_model] DIAGNOSTIC: output dir contents after quantize
-rw-r--r-- 1 root root      2468 May 21 09:30 config.json
-rw-r--r-- 1 root root 4983672832 May 21 09:30 model-00001-of-00003.safetensors
-rw-r--r-- 1 root root 3213440128 May 21 09:30 model-00002-of-00003.safetensors
[quantize] done.
"""

LEGACY_FORMAT_LOG = """\
[versions] gptqmodel=7.0.0
[versions] transformers=4.57.1
[quantize] saving to /root/autodl-fs/zyn/models/sub-quantized
[qzeros-fix] rewrote 71 qzeros tensors in model-00002-of-00003.safetensors
[qzeros-fix] rewrote 25 qzeros tensors in model-00003-of-00003.safetensors
[qzeros-fix] total 96 qzeros tensors patched 0x77777777 -> 0x88888888
[quantize] done.
"""

FATAL_FORMAT_LOG = """\
[versions] gptqmodel=7.0.5
[versions] transformers=4.57.1
[quantize] saving to /tmp/out
[qzeros-fix] output_dir inventory (/tmp/out):
[qzeros-fix]   model.safetensors  (6,536,000,000 bytes)
[qzeros-fix] scanning 1 .safetensors weight file(s)
[qzeros-fix] SUMMARY:
[qzeros-fix]   qzeros tensors seen:        0
[qzeros-fix]   patched (had 0x77777777):   0
[qzeros-fix]   already good (0x88888888):  0
[qzeros-fix]   unknown dtype skipped:      0
[qzeros-fix]   unknown values not patched: 0
[qzeros-fix] FATAL: zero .qzeros tensors found in any shard.
"""

PLATFORM_BAD_QZEROS_LOG = """\
[prepare_env] forcing gptqmodel==7.0.0 (currently: 7.0.5)
[versions] python=3.10.12
[versions] torch=2.9.1
[versions] transformers=4.57.1
[versions] gptqmodel=7.0.0
[prepare_model] DIAGNOSTIC: calib jsonl path=/sub/perf_public_set.jsonl
[prepare_model] DIAGNOSTIC: calib jsonl size=23925601 bytes
[prepare_model] DIAGNOSTIC: calib jsonl row count=150
[quantize] saving to /tmp/out
[qzeros-fix] output_dir inventory (/tmp/out):
[qzeros-fix]   model.safetensors  (6,536,000,000 bytes)
[qzeros-fix] scanning 1 .safetensors weight file(s)
[qzeros-fix] model.safetensors: patched 96 qzeros tensors
[qzeros-fix] SUMMARY:
[qzeros-fix]   qzeros tensors seen:        96
[qzeros-fix]   patched (had 0x77777777):   96
[qzeros-fix]   already good (0x88888888):  0
[qzeros-fix]   unknown dtype skipped:      0
[qzeros-fix]   unknown values not patched: 0
[qzeros-fix] POST-CHECK model.safetensors::layers.0.mlp.gate_proj.qzeros first_val=-2004318072 (0x88888888)
[qzeros-fix] OK
[quantize] done.
"""


def _write(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


class TestParseNewFormat(unittest.TestCase):
    def setUp(self):
        self.p = pqd.parse_log(_write(NEW_FORMAT_LOG))

    def test_versions_extracted(self):
        self.assertEqual(self.p.versions["gptqmodel"], "7.0.0")
        self.assertEqual(self.p.versions["transformers"], "4.57.1")
        self.assertEqual(self.p.versions["torch"], "2.9.1")

    def test_install_skipped(self):
        self.assertTrue(self.p.gptqmodel_install.skipped)
        self.assertIsNone(self.p.gptqmodel_install.forced_version)

    def test_sed_patches(self):
        files = [s.file_name for s in self.p.sed_patches]
        self.assertIn("minicpm_backend.py", files)
        self.assertIn("minicpm_sparse_utils.py", files)

    def test_sglang_args(self):
        self.assertIsNotNone(self.p.sglang_server_args)
        self.assertIn("gptq_marlin", self.p.sglang_server_args)

    def test_calib(self):
        c = self.p.calib
        self.assertEqual(c.row_count, 150)
        self.assertEqual(c.size_bytes, 23925601)
        self.assertEqual(c.num_calib, 256)
        self.assertEqual(c.max_calib_len, 8192)
        self.assertEqual(c.window_mode, "tail")
        self.assertIn("magic number", c.first_row_preview or "")

    def test_quant_save(self):
        self.assertEqual(self.p.quant_save_path,
                         "/root/autodl-fs/zyn/models/sub-quantized")
        self.assertTrue(self.p.quant_completed)

    def test_qzeros_fix_counts(self):
        q = self.p.qzeros_fix
        self.assertEqual(q.scanned_file_count, 3)
        self.assertEqual(q.total_seen, 96)
        self.assertEqual(q.total_patched, 96)
        self.assertEqual(q.total_already_good, 0)
        self.assertTrue(q.ended_ok)
        self.assertIsNone(q.fatal_message)

    def test_qzeros_per_shard(self):
        m = dict(self.p.qzeros_fix.per_shard_patched)
        self.assertEqual(m["model-00002-of-00003.safetensors"], 71)
        self.assertEqual(m["model-00003-of-00003.safetensors"], 25)

    def test_qzeros_inventory(self):
        names = [n for n, _ in self.p.qzeros_fix.inventory_files]
        self.assertIn("model-00001-of-00003.safetensors", names)
        self.assertEqual(len(names), 4)

    def test_qzeros_samples(self):
        s = self.p.qzeros_fix.samples
        self.assertEqual(len(s), 2)
        self.assertEqual(s[0].dtype, "torch.int32")
        self.assertEqual(s[0].unique_hex, ["0x77777777"])
        self.assertEqual(s[0].unique_decimal, [2004318071])

    def test_post_check_good(self):
        pc = self.p.qzeros_fix.post_check
        self.assertIsNotNone(pc)
        self.assertTrue(pc.is_good)
        self.assertEqual(pc.first_val_hex, "0x88888888")


class TestLegacyFormat(unittest.TestCase):
    def test_legacy_per_shard_and_total(self):
        p = pqd.parse_log(_write(LEGACY_FORMAT_LOG))
        self.assertEqual(p.qzeros_fix.total_patched, 96)
        self.assertTrue(p.qzeros_fix.ended_ok)
        m = dict(p.qzeros_fix.per_shard_patched)
        self.assertEqual(m["model-00002-of-00003.safetensors"], 71)
        self.assertEqual(m["model-00003-of-00003.safetensors"], 25)
        self.assertTrue(p.quant_completed)


class TestFatalCapture(unittest.TestCase):
    def test_fatal(self):
        p = pqd.parse_log(_write(FATAL_FORMAT_LOG))
        q = p.qzeros_fix
        self.assertEqual(q.total_seen, 0)
        self.assertEqual(q.total_patched, 0)
        self.assertIsNotNone(q.fatal_message)
        self.assertIn("zero .qzeros tensors", q.fatal_message)
        self.assertGreaterEqual(len(p.fatals), 1)
        self.assertFalse(q.ended_ok)


class TestPostCheckBugDetection(unittest.TestCase):
    def test_bug_first_val(self):
        bug_line = (
            "[qzeros-fix] POST-CHECK m.safetensors::layers.0.mlp.gate_proj.qzeros "
            "first_val=2004318071 (0x77777777)\n"
        )
        p = pqd.parse_log(_write(bug_line))
        self.assertIsNotNone(p.qzeros_fix.post_check)
        self.assertFalse(p.qzeros_fix.post_check.is_good)
        self.assertEqual(p.qzeros_fix.post_check.first_val_hex, "0x77777777")


class TestDiffMode(unittest.TestCase):
    def setUp(self):
        self.good = pqd.parse_log(_write(NEW_FORMAT_LOG))
        self.bad = pqd.parse_log(_write(FATAL_FORMAT_LOG))
        self.platform_singlefile = pqd.parse_log(_write(PLATFORM_BAD_QZEROS_LOG))

    def test_diff_surfaces_version_drift(self):
        # FATAL_FORMAT_LOG has gptqmodel=7.0.5; NEW_FORMAT_LOG has 7.0.0
        diff = pqd._versions_diff(self.good.versions, self.bad.versions)
        names = [k for k, _, _ in diff]
        self.assertIn("gptqmodel", names)

    def test_diff_surfaces_qzeros_drift(self):
        rows = pqd._qzeros_diff_summary(self.good.qzeros_fix, self.bad.qzeros_fix)
        fields = {n for n, _, _ in rows}
        self.assertIn("total_seen", fields)
        self.assertIn("total_patched", fields)
        self.assertIn("ended_ok", fields)
        self.assertIn("fatal_message", fields)

    def test_render_diff_mentions_fatal(self):
        report = pqd.render_diff_report(self.good, self.bad)
        self.assertIn("FATAL", report)
        self.assertIn("Verdict heuristic", report)

    def test_render_diff_platform_singlefile_vs_local_sharded(self):
        # When platform writes single-file and patches but local sharded
        # also patches, the heuristic should note the safetensors file
        # count divergence.
        report = pqd.render_diff_report(self.good, self.platform_singlefile)
        self.assertIn("safetensors file count", report)

    def test_one_line_summary_stable(self):
        s = pqd.one_line_summary(self.good)
        self.assertIn("gptqmodel=7.0.0", s)
        self.assertIn("fix=OK", s)
        self.assertIn("quant_done=True", s)


class TestSampleListLiteralParsing(unittest.TestCase):
    def test_parse_simple(self):
        self.assertEqual(pqd._parse_list_literal("[1, 2, 3]"), [1, 2, 3])

    def test_parse_hex_strings(self):
        self.assertEqual(
            pqd._parse_list_literal("['0x77777777', '0x88888888']"),
            ["0x77777777", "0x88888888"],
        )

    def test_parse_garbage(self):
        # Should not crash on malformed input — returns [].
        self.assertEqual(pqd._parse_list_literal("not a list"), [])


if __name__ == "__main__":
    unittest.main()
