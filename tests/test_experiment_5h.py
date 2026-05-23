"""Tests for scripts/experiment_5h/pick_winners.py and parse_results.py.

Why these tests matter
----------------------
The 5h pipeline that ran 2026-05-23 01:36:22 silently lost 5.5 hours of
results because:

1. `pick_winners.py` ignored its --session arg, picking winners from any
   session in the CSV → Phase E would have stacked wrong knobs.
2. `parse_results.py::parse_aggregate` took the LAST regex match of
   `acc=N`, which on a SOAR-Toolkit eval log can be a per-sample value
   rather than the aggregate.
3. `commit_push.sh` used `if git push | tail` whose exit code was tail's,
   so non-fast-forward rejections were silently reported as success.

These tests cover bugs 1 and 2. Bug 3 is shell, tested separately by
running `bash -n` over the file and a manual divergence simulation.
"""
from __future__ import annotations

import csv
import io
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PICK_WINNERS = REPO / "scripts" / "experiment_5h" / "pick_winners.py"
PARSE_RESULTS = REPO / "scripts" / "experiment_5h" / "parse_results.py"


def run_pick_winners(csv_text: str, session: str, threshold: float = 2.0,
                    baseline_exp: str = "A_J0_baseline") -> tuple[str, str, int]:
    """Run pick_winners.py against an in-memory CSV. Returns (stdout, stderr, rc)."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        tf.write(csv_text)
        csv_path = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, str(PICK_WINNERS),
             "--csv", csv_path,
             "--session", session,
             "--baseline-exp", baseline_exp,
             "--threshold", str(threshold)],
            capture_output=True, text=True, timeout=30,
        )
        return proc.stdout, proc.stderr, proc.returncode
    finally:
        Path(csv_path).unlink(missing_ok=True)


class TestPickWinnersSessionFilter(unittest.TestCase):
    """The bug: pick_winners.py ignored --session. With this fix it must
    REFUSE to pick winners from a different session, even if the exp name
    matches.
    """

    HEADER = "timestamp,exp,desc,acc,acc_ori,niah,cwe,fwe,qa,mcq,duration_s,log_path,predictions_path\n"

    def test_picks_winner_from_correct_session(self):
        csv_text = self.HEADER + textwrap.dedent("""\
            2026-05-23 01:00,A_J0_baseline,,,46.00,,,,,,,scripts/logs/5h_NOW/A_J0_baseline.log,
            2026-05-23 01:35,D1_sym_false,,,50.00,,,,,,,scripts/logs/5h_NOW/D1_sym_false.log,
        """)
        out, err, rc = run_pick_winners(csv_text, session="NOW", threshold=2.0)
        self.assertEqual(rc, 0)
        self.assertIn("GPTQ_SYM=False", out)

    def test_refuses_winner_from_OTHER_session(self):
        """Critical: a D1 row from a DIFFERENT session must NOT be picked."""
        csv_text = self.HEADER + textwrap.dedent("""\
            2026-05-23 00:45,A_J0_baseline,,,46.00,,,,,,,scripts/logs/5h_EARLIER/A_J0_baseline.log,
            2026-05-23 00:50,D1_sym_false,,,99.00,,,,,,,scripts/logs/5h_EARLIER/D1_sym_false.log,
        """)
        # Asking for session NOW — even though D1 has acc 99 in EARLIER, it must NOT match
        out, err, rc = run_pick_winners(csv_text, session="NOW", threshold=2.0)
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "",
            f"expected no winners (other session is invisible), got: {out!r}")
        self.assertIn("no rows for session NOW", err)

    def test_baseline_missing_in_session_means_no_winners(self):
        csv_text = self.HEADER + textwrap.dedent("""\
            2026-05-23 00:45,A_J0_baseline,,,46.00,,,,,,,scripts/logs/5h_EARLIER/A_J0_baseline.log,
            2026-05-23 01:35,D1_sym_false,,,50.00,,,,,,,scripts/logs/5h_NOW/D1_sym_false.log,
        """)
        out, err, rc = run_pick_winners(csv_text, session="NOW", threshold=2.0)
        self.assertEqual(rc, 0)
        # NOW has D1 but no baseline — without baseline, can't decide winners
        self.assertEqual(out.strip(), "")
        self.assertIn("no baseline", err)

    def test_multiple_winners_in_session(self):
        csv_text = self.HEADER + textwrap.dedent("""\
            2026-05-23 01:00,A_J0_baseline,,,46.00,,,,,,,scripts/logs/5h_NOW/A_J0_baseline.log,
            2026-05-23 01:35,D1_sym_false,,,50.00,,,,,,,scripts/logs/5h_NOW/D1_sym_false.log,
            2026-05-23 01:55,D2_damp_01,,,49.00,,,,,,,scripts/logs/5h_NOW/D2_damp_01.log,
            2026-05-23 02:20,D3_calib_512,,,47.00,,,,,,,scripts/logs/5h_NOW/D3_calib_512.log,
        """)
        out, err, rc = run_pick_winners(csv_text, session="NOW", threshold=2.0)
        self.assertEqual(rc, 0)
        # D1 (+4pp) qualifies, D2 (+3pp) qualifies, D3 (+1pp) does not
        self.assertIn("GPTQ_SYM=False", out)
        self.assertIn("GPTQ_DAMPENING_FRAC=0.1", out)
        self.assertNotIn("NUM_CALIB=512", out)

    def test_empty_csv(self):
        out, err, rc = run_pick_winners(self.HEADER, session="NOW")
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")


# ---------------------------------------------------------------------------
# parse_aggregate priority tests
# ---------------------------------------------------------------------------
def import_parse_aggregate():
    # Module-level import — adds the parent path so we can import the module.
    sys.path.insert(0, str(REPO / "scripts" / "experiment_5h"))
    import importlib
    mod = importlib.import_module("parse_results")
    return mod.parse_aggregate


class TestParseAggregatePriority(unittest.TestCase):
    """The bug: parse_aggregate used re.findall(...)[-1], which on a log
    that prints per-sample acc=N N times would grab the LAST sample's acc
    instead of the summary aggregate. The fix prefers JSON-style > line-
    anchored > inline, and filters values to a plausible [0, 100] range.
    """

    def setUp(self):
        parse = import_parse_aggregate()
        self.parse = parse

    def _make_log(self, text: str) -> Path:
        tf = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        tf.write(text)
        tf.close()
        self.addCleanup(lambda: Path(tf.name).unlink(missing_ok=True))
        return Path(tf.name)

    def test_per_sample_acc_does_not_shadow_aggregate(self):
        log = self._make_log(textwrap.dedent("""\
            ... per-sample logs ...
            sample 1: acc=1.0  prediction=ABC
            sample 2: acc=0.0  prediction=XYZ
            sample 3: acc=1.0  prediction=DEF
            ... summary ...
            acc_ori = 46.89
            acc     = 58.61
            duration_s = 3120
        """))
        out = self.parse(log)
        # Aggregate must win, even though there are TWO per-sample matches that
        # happen to fall in [0,1] range and would convert to 100.
        self.assertEqual(out["acc_ori"], "46.89")
        self.assertEqual(out["acc"], "58.61")

    def test_json_style_wins(self):
        log = self._make_log(textwrap.dedent("""\
            ... evaluating ...
            acc=99 (sample 0)
            acc=99 (sample 1)
            ... final ...
            {"acc": 58.61, "acc_ori": 46.89}
        """))
        out = self.parse(log)
        self.assertEqual(out["acc"], "58.61")
        self.assertEqual(out["acc_ori"], "46.89")

    def test_fraction_is_converted_to_percent(self):
        log = self._make_log("acc_ori: 0.4689\nacc: 0.5861\n")
        out = self.parse(log)
        self.assertEqual(out["acc_ori"], "46.89")
        self.assertEqual(out["acc"], "58.61")

    def test_implausible_values_rejected(self):
        # Some logs print huge token counts like "acc=65546". Must be filtered.
        log = self._make_log(textwrap.dedent("""\
            output_tokens=65546 acc=65546
            output_tokens=42 acc=42
            acc_ori = 46.89
            acc = 58.61
        """))
        out = self.parse(log)
        self.assertEqual(out["acc_ori"], "46.89")
        self.assertEqual(out["acc"], "58.61")

    def test_missing_metrics(self):
        log = self._make_log("just a normal log with no metrics here\n")
        out = self.parse(log)
        self.assertEqual(out["acc"], "")
        self.assertEqual(out["acc_ori"], "")

    def test_only_average_score_fallback(self):
        log = self._make_log("Average Score: 63.33%\n")
        out = self.parse(log)
        self.assertEqual(out["acc"], "63.33")


if __name__ == "__main__":
    unittest.main()
