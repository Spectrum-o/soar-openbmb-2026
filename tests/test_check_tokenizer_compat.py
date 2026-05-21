#!/usr/bin/env python3
"""Unit tests for tools/check_tokenizer_compat.py.

Run from repo root:
    python3 -m unittest tests.test_check_tokenizer_compat -v

Mirrors tests/test_parse_quant_diagnostic.py: unittest + tempfile,
stdlib-only. Covers the H4-related failure detection paths:
  - byte-identical artifact vs base -> no drift
  - file size / md5 divergence -> flagged
  - file present only in artifact (GPTQModel-synthesized) -> flagged
  - chat_template inline-only -> safe
  - chat_template sidecar without inline -> flagged (H4 mechanism)
  - chat_template in BOTH but disagree -> flagged (H4 mechanism)
  - chat_template in BOTH and agree -> safe
  - vocab size drift -> flagged
  - tokenizer_config.json chat_template as list-form (legacy) -> handled
  - exit codes match drift state
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import check_tokenizer_compat as ctc  # noqa: E402


def _make_dir(tmp: Path, name: str) -> Path:
    d = tmp / name
    d.mkdir()
    return d


def _write_tokenizer_json(p: Path, vocab_size: int = 32000) -> None:
    """Write a minimal but parseable tokenizer.json."""
    p.write_text(json.dumps({
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": {f"tok_{i}": i for i in range(vocab_size)},
        },
    }))


def _write_config(p: Path, chat_template: str | list | None) -> None:
    cfg: dict = {"model_type": "llama"}
    if chat_template is not None:
        cfg["chat_template"] = chat_template
    p.write_text(json.dumps(cfg))


class TestNoDrift(unittest.TestCase):
    """When artifact files match base byte-for-byte, no drift."""

    def test_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            base = _make_dir(tmp_p, "base")
            art = _make_dir(tmp_p, "artifact")
            _write_tokenizer_json(base / "tokenizer.json", 32000)
            _write_tokenizer_json(art / "tokenizer.json", 32000)
            _write_config(base / "tokenizer_config.json", "INLINE")
            _write_config(art / "tokenizer_config.json", "INLINE")

            files = ctc.scan(base, art)
            _, drift = ctc.render_report(base, art, files)
            self.assertFalse(drift, "byte-identical should produce no drift")


class TestFileLevelDiff(unittest.TestCase):
    def test_size_divergence_flags_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            base = _make_dir(tmp_p, "base")
            art = _make_dir(tmp_p, "artifact")
            _write_tokenizer_json(base / "tokenizer.json", 32000)
            _write_tokenizer_json(art / "tokenizer.json", 32100)  # different vocab size
            _write_config(base / "tokenizer_config.json", "X")
            _write_config(art / "tokenizer_config.json", "X")
            files = ctc.scan(base, art)
            report, drift = ctc.render_report(base, art, files)
            self.assertTrue(drift)
            self.assertIn("DIFFERS", report)

    def test_artifact_only_file_flags_drift(self):
        # H4 fingerprint: chat_template.jinja appears in artifact, not base
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            base = _make_dir(tmp_p, "base")
            art = _make_dir(tmp_p, "artifact")
            _write_tokenizer_json(base / "tokenizer.json")
            _write_tokenizer_json(art / "tokenizer.json")
            _write_config(base / "tokenizer_config.json", "INLINE-only")
            _write_config(art / "tokenizer_config.json", "INLINE-only")
            (art / "chat_template.jinja").write_text("SIDECAR")  # only in artifact
            files = ctc.scan(base, art)
            report, drift = ctc.render_report(base, art, files)
            self.assertTrue(drift)
            self.assertIn("only in artifact", report)

    def test_base_only_file_flags_drift(self):
        # If artifact is MISSING a file the base has, that's also drift
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            base = _make_dir(tmp_p, "base")
            art = _make_dir(tmp_p, "artifact")
            _write_tokenizer_json(base / "tokenizer.json")
            _write_tokenizer_json(art / "tokenizer.json")
            _write_config(base / "tokenizer_config.json", "X")
            # Note: artifact does NOT get tokenizer_config.json
            files = ctc.scan(base, art)
            report, drift = ctc.render_report(base, art, files)
            self.assertTrue(drift)
            self.assertIn("only in base", report)


class TestChatTemplateDuplication(unittest.TestCase):
    def _setup(self, tmp: Path, base_inline, art_inline, art_sidecar):
        base = _make_dir(tmp, "base")
        art = _make_dir(tmp, "artifact")
        _write_tokenizer_json(base / "tokenizer.json")
        _write_tokenizer_json(art / "tokenizer.json")
        _write_config(base / "tokenizer_config.json", base_inline)
        _write_config(art / "tokenizer_config.json", art_inline)
        if art_sidecar is not None:
            (art / "chat_template.jinja").write_text(art_sidecar)
        return base, art

    def test_inline_only_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, art = self._setup(Path(tmp), "INLINE", "INLINE", None)
            files = ctc.scan(base, art)
            report, _ = ctc.render_report(base, art, files)
            self.assertIn("Safe", report)

    def test_inline_and_sidecar_agree_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, art = self._setup(Path(tmp), "TPL", "TPL", "TPL")
            files = ctc.scan(base, art)
            report, _ = ctc.render_report(base, art, files)
            self.assertIn("Likely safe across versions", report)

    def test_inline_and_sidecar_disagree_flagged_drift(self):
        # H4 mechanism: both present but disagree -> transformers
        # version determines which one is read
        with tempfile.TemporaryDirectory() as tmp:
            base, art = self._setup(Path(tmp), "BASE", "INLINE_VERSION", "SIDECAR_VERSION")
            files = ctc.scan(base, art)
            report, drift = ctc.render_report(base, art, files)
            self.assertTrue(drift)
            self.assertIn("BOTH inline and sidecar but they DISAGREE", report)

    def test_sidecar_only_no_inline_flagged_drift(self):
        # Worst case: artifact has no inline template at all, only sidecar.
        # transformers<4.45 sees empty template -> garbage on platform.
        with tempfile.TemporaryDirectory() as tmp:
            base = _make_dir(Path(tmp), "base")
            art = _make_dir(Path(tmp), "artifact")
            _write_tokenizer_json(base / "tokenizer.json")
            _write_tokenizer_json(art / "tokenizer.json")
            _write_config(base / "tokenizer_config.json", "BASE")
            # Artifact's tokenizer_config has NO chat_template field
            _write_config(art / "tokenizer_config.json", None)
            (art / "chat_template.jinja").write_text("SIDECAR_ONLY")
            files = ctc.scan(base, art)
            report, drift = ctc.render_report(base, art, files)
            self.assertTrue(drift)
            self.assertIn("NO inline", report)
            self.assertIn("H4 failure mechanism confirmed", report)

    def test_neither_inline_nor_sidecar_flagged_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _make_dir(Path(tmp), "base")
            art = _make_dir(Path(tmp), "artifact")
            _write_tokenizer_json(base / "tokenizer.json")
            _write_tokenizer_json(art / "tokenizer.json")
            _write_config(base / "tokenizer_config.json", None)
            _write_config(art / "tokenizer_config.json", None)
            files = ctc.scan(base, art)
            report, drift = ctc.render_report(base, art, files)
            self.assertTrue(drift)
            self.assertIn("NEITHER inline nor sidecar", report)


class TestVocabDrift(unittest.TestCase):
    def test_vocab_size_diff_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _make_dir(Path(tmp), "base")
            art = _make_dir(Path(tmp), "artifact")
            _write_tokenizer_json(base / "tokenizer.json", 32000)
            _write_tokenizer_json(art / "tokenizer.json", 32128)
            _write_config(base / "tokenizer_config.json", "X")
            _write_config(art / "tokenizer_config.json", "X")
            files = ctc.scan(base, art)
            report, drift = ctc.render_report(base, art, files)
            self.assertTrue(drift)
            self.assertIn("vocab size diverged by 128", report)


class TestLegacyListChatTemplate(unittest.TestCase):
    """HF historical chat_template-as-list-of-dicts form should not crash."""

    def test_list_form_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _make_dir(Path(tmp), "base")
            _write_tokenizer_json(base / "tokenizer.json")
            (base / "tokenizer_config.json").write_text(json.dumps({
                "chat_template": [
                    {"name": "default", "template": "DEFAULT_TPL"},
                    {"name": "tool_use", "template": "TOOLUSE_TPL"},
                ]
            }))
            out = ctc.inspect_chat_template_inline(base / "tokenizer_config.json")
            self.assertIsNotNone(out)
            self.assertIn("DEFAULT_TPL", out)
            self.assertIn("TOOLUSE_TPL", out)


class TestHelperFunctions(unittest.TestCase):
    def test_md5_file_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("hello")
            h1 = ctc.md5_file(p)
            h2 = ctc.md5_file(p)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 32)

    def test_sidecar_absent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertIsNone(ctc.inspect_chat_template_sidecar(d))

    def test_vocab_size_from_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                ctc.vocab_size_from_tokenizer_json(Path(tmp) / "nope.json")
            )


class TestExitCodeViaMain(unittest.TestCase):
    def _invoke_main(self, base: Path, artifact: Path) -> int:
        """Patch sys.argv + call ctc.main, restoring afterwards."""
        old_argv = sys.argv
        old_stdout = sys.stdout
        try:
            sys.argv = ["check_tokenizer_compat.py",
                        "--base", str(base), "--artifact", str(artifact)]
            sys.stdout = open(os.devnull, "w")
            return ctc.main()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            sys.argv = old_argv

    def test_exit_0_when_no_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            base = _make_dir(tmp_p, "base")
            art = _make_dir(tmp_p, "artifact")
            _write_tokenizer_json(base / "tokenizer.json")
            _write_tokenizer_json(art / "tokenizer.json")
            _write_config(base / "tokenizer_config.json", "T")
            _write_config(art / "tokenizer_config.json", "T")
            rc = self._invoke_main(base, art)
            self.assertEqual(rc, 0)

    def test_exit_1_when_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            base = _make_dir(tmp_p, "base")
            art = _make_dir(tmp_p, "artifact")
            _write_tokenizer_json(base / "tokenizer.json", 32000)
            _write_tokenizer_json(art / "tokenizer.json", 33000)  # vocab drift
            _write_config(base / "tokenizer_config.json", "X")
            _write_config(art / "tokenizer_config.json", "X")
            rc = self._invoke_main(base, art)
            self.assertEqual(rc, 1)

    def test_exit_2_when_base_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = _make_dir(Path(tmp), "art")
            rc = self._invoke_main(Path(tmp) / "nonexistent", art)
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
