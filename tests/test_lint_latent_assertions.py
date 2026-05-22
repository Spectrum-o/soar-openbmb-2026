"""Tests for tools/lint_latent_assertions.py.

The lint exists to catch the v23 platform-failure bug class. The single
most important test: if we synthesize the exact v23 anti-pattern in a
temporary file, the lint flags it. If we synthesize the post-fix version,
the lint stays silent.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lint_latent_assertions import _strip_comments_and_docstrings, lint_file  # noqa: E402


# Synthetic reproductions of the v23 anti-pattern. Kept inline (vs reading
# from the real history) so the tests are self-contained and don't depend
# on the commit graph being intact.

V23_BUG_SHAPE = '''
import sys
from pathlib import Path

def fix_qzeros_for_marlin(output_dir):
    shards = sorted(Path(output_dir).glob("*.safetensors"))
    total_qzeros_patched = 99
    verified = False
    # The v23 bug: blindly inspects shards[0], which contains lm_head + embed
    # for MLP-only artifacts — no qzeros at all.
    for key in shards[0:].__iter__():  # pretend this was iterating the wrong thing
        pass
    bad = shards[0]
    with open(bad) as f:
        for line in f:
            if "qzero" in line:
                verified = True
    if not verified and total_qzeros_patched > 0:
        raise RuntimeError("FATAL: qzeros patched but POST-CHECK failed")
'''


V23_POST_FIX_SHAPE = '''
import sys
from pathlib import Path

def fix_qzeros_for_marlin(output_dir):
    # Post-fix: iterate the LIST OF SHARDS WE ACTUALLY PATCHED, not a fixed
    # index. per_shard_patched is built up earlier in the same function.
    per_shard_patched = []
    verified = False
    for shard in per_shard_patched:
        with open(shard) as f:
            for line in f:
                if "qzero" in line:
                    verified = True
                    break
    if not verified and sum(1 for _ in per_shard_patched) > 0:
        raise RuntimeError("FATAL: qzeros patched but POST-CHECK failed")
'''


CLEAN_INDEX_NO_ASSERTION = '''
# shards[0] used here, but NOT near any sanity check — just logging.
import sys
shards = list(range(5))
print(f"first shard: {shards[0]}")
print(f"second:      {shards[1]}")
'''


SLICE_IS_NOT_INDEXING = '''
def f(shards):
    # shards[0:] is iteration, not a hard-coded index lookup. Should NOT trigger.
    verified = False
    for s in shards[0:]:
        if check(s):
            verified = True
    if not verified:
        raise RuntimeError("FATAL: nothing verified")
'''


COMMENT_MENTION_IS_OK = '''
def f(shards):
    # The v23 bug was `shards[0]` indexing — but this is just a comment.
    return len(shards)
'''


class TestLintFile(unittest.TestCase):
    def _write_and_lint(self, src: str) -> list:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tf:
            tf.write(src)
            path = Path(tf.name)
        try:
            return lint_file(path)
        finally:
            path.unlink()

    def test_v23_bug_shape_is_flagged(self):
        findings = self._write_and_lint(V23_BUG_SHAPE)
        self.assertTrue(findings, "v23 bug shape MUST be flagged")
        # Specifically, we want `shards[0]` (the indexing, not the slice) called out.
        flagged = [f.match for f in findings]
        self.assertIn("shards[0]", flagged)

    def test_post_fix_shape_is_clean(self):
        findings = self._write_and_lint(V23_POST_FIX_SHAPE)
        self.assertEqual(
            findings, [],
            f"post-fix shape should NOT be flagged; got: {[f.match for f in findings]}",
        )

    def test_indexing_without_critical_assertion_is_clean(self):
        # Code that uses shards[0] for logging — not a critical sanity check
        # — should not fire. The lint is calibrated for the v23 bug shape
        # (indexing-in-window-of-FATAL), not for all shard indexing.
        findings = self._write_and_lint(CLEAN_INDEX_NO_ASSERTION)
        self.assertEqual(findings, [])

    def test_slice_is_not_flagged(self):
        # `shards[0:]` is iteration. Distinguishing slice from index is
        # the whole point of the negative-lookahead in the regex.
        findings = self._write_and_lint(SLICE_IS_NOT_INDEXING)
        self.assertEqual(
            findings, [],
            f"slice expression must not be flagged as indexing; got: {[f.match for f in findings]}",
        )

    def test_comment_mention_does_not_trigger(self):
        # The lint strips comments. A line of prose mentioning shards[0]
        # shouldn't fire.
        findings = self._write_and_lint(COMMENT_MENTION_IS_OK)
        self.assertEqual(findings, [])

    def test_other_shard_like_names_also_flagged(self):
        src = '''
def f(safetensor_files):
    verified = False
    bad = safetensor_files[0]
    if not verified:
        raise RuntimeError("FATAL")
'''
        findings = self._write_and_lint(src)
        self.assertTrue(findings)
        self.assertTrue(any("safetensor_files[0]" in f.match for f in findings))


class TestStripCommentsAndDocstrings(unittest.TestCase):
    def test_strips_line_comment(self):
        out = _strip_comments_and_docstrings("x = 1  # comment\n")
        self.assertEqual(out, [(1, "x = 1  ")])

    def test_strips_triple_quoted_docstring(self):
        src = '"""hello\nworld"""\nx = 1\n'
        out = _strip_comments_and_docstrings(src)
        # Line 3 should be intact.
        text_for_line_3 = next(t for ln, t in out if ln == 3)
        self.assertIn("x = 1", text_for_line_3)
        # Docstring body should be blanked.
        text_for_line_2 = next(t for ln, t in out if ln == 2)
        self.assertEqual(text_for_line_2, "")


if __name__ == "__main__":
    unittest.main()
