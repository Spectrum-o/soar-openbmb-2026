"""Tests for the GPTQ_DESC_ACT / GPTQ_STATIC_GROUPS env knob plumbing.

These knobs were added 2026-05-23 on branch quant/preflight-hardening. The
historical "Marlin doesn't support desc_act=True" comment was wrong — see
python/sglang/srt/layers/quantization/gptq.py:238 (desc_act is only forced
to False at group_size=-1, which we never use).

What we test:
  * Default stays False — adding these knobs must NOT change in-flight
    experiment behavior. Production runs without setting the env var
    must produce identical artifacts.
  * "True" / "true" / "1" / "yes" all resolve to True (consistent with
    the existing GPTQ_SYM convention).
  * Other strings resolve to False (no surprises).
  * The new knob is wired into BOTH make_quant_config (the GPTQModel
    constructor side) AND write_sglang_compatible_quant_config (the
    config.json side) — these must agree, or SGLang loader and the
    quant artifact will diverge (cf. v14 KeyError).

Tested via source-text inspection rather than importing the script
itself, because the script eagerly imports gptqmodel/transformers which
aren't in the dev mirror's pip env.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUANT_SCRIPT = REPO / "submission_gptqmodel_calib_w4a16" / "quantize_gptqmodel_w4a16.py"


class TestDescActEnvKnob(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = QUANT_SCRIPT.read_text(encoding="utf-8")

    def test_quant_script_exists(self):
        self.assertTrue(
            QUANT_SCRIPT.exists(),
            f"expected quant script at {QUANT_SCRIPT}",
        )

    def test_make_quant_config_reads_desc_act_env(self):
        # The function body must reference GPTQ_DESC_ACT — without this,
        # the user has no way to flip the knob.
        # Find the make_quant_config function body specifically.
        m = re.search(
            r"def make_quant_config\(.*?\n(.*?)\ndef ",
            self.src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(m, "could not locate make_quant_config body")
        body = m.group(1)
        self.assertIn(
            "GPTQ_DESC_ACT", body,
            "make_quant_config must read GPTQ_DESC_ACT env var",
        )

    def test_make_quant_config_default_is_false(self):
        # Default behavior must be unchanged. The pattern we look for:
        # os.environ.get("GPTQ_DESC_ACT", "False")
        self.assertRegex(
            self.src,
            r"os\.environ\.get\(\s*['\"]GPTQ_DESC_ACT['\"]\s*,\s*['\"]False['\"]\s*\)",
            "GPTQ_DESC_ACT must default to 'False' to preserve in-flight "
            "experiment behavior. If you intentionally flipped the default, "
            "update this test AND coordinate with the running 5h experiments.",
        )

    def test_sglang_quant_config_also_reads_desc_act(self):
        # write_sglang_compatible_quant_config must also read the same env
        # var. If quant-time and load-time disagree on desc_act, you get a
        # v14-class KeyError at SGLang weight loading.
        m = re.search(
            r"def write_sglang_compatible_quant_config\(.*?\n(.*?)\ndef ",
            self.src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(m, "could not locate write_sglang_compatible_quant_config body")
        body = m.group(1)
        self.assertIn(
            "GPTQ_DESC_ACT", body,
            "write_sglang_compatible_quant_config must read GPTQ_DESC_ACT "
            "to keep the quantize_config.json in sync with what GPTQModel "
            "actually produced",
        )

    def test_both_sides_use_identical_truth_string_rules(self):
        # Both call sites must reference GPTQ_DESC_ACT and parse the same
        # set of truth strings. They may differ syntactically (one-liner
        # vs split across two lines) but the (true, 1, yes) tuple should
        # appear in both code regions.
        for func_name in ("make_quant_config", "write_sglang_compatible_quant_config"):
            m = re.search(
                rf"def {func_name}\(.*?\n(.*?)\ndef ",
                self.src,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(m, f"could not locate {func_name} body")
            body = m.group(1)
            self.assertIn(
                "GPTQ_DESC_ACT", body,
                f"{func_name} must reference GPTQ_DESC_ACT — otherwise "
                f"quant-time and load-time desc_act will silently diverge",
            )
            # The truth-tuple must be consistent. We accept either of the
            # two flavors currently in use.
            truth_tuple_found = (
                re.search(r"in\s*\(\s*['\"]true['\"]\s*,\s*['\"]1['\"]\s*,\s*['\"]yes['\"]\s*\)", body)
                is not None
            )
            self.assertTrue(
                truth_tuple_found,
                f"{func_name} must parse GPTQ_DESC_ACT with the canonical "
                f"truth tuple ('true', '1', 'yes') — keeps parsing in sync "
                f"across both call sites and matches GPTQ_SYM convention",
            )

    def test_static_groups_default_is_false(self):
        # static_groups is the companion knob; same default-False contract.
        self.assertRegex(
            self.src,
            r"os\.environ\.get\(\s*['\"]GPTQ_STATIC_GROUPS['\"]\s*,\s*['\"]False['\"]\s*\)",
        )

    def test_docstring_corrects_historical_marlin_claim(self):
        # We want a future reader to NOT re-introduce the "Marlin doesn't
        # support desc_act=True" myth. Make sure the docstring explicitly
        # corrects it.
        m = re.search(
            r"def make_quant_config\(.*?\"\"\"(.*?)\"\"\"",
            self.src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(m, "make_quant_config docstring not found")
        doc = m.group(1)
        self.assertIn("desc_act", doc.lower())
        # Either explicit correction OR explicit citation of the SGLang
        # path that handles it. We're loose on phrasing.
        cited = ("act-order" in doc.lower() or
                 "act_order" in doc.lower() or
                 "process_weights_after_loading" in doc or
                 "supports desc_act" in doc.lower())
        self.assertTrue(
            cited,
            "make_quant_config docstring should note that SGLang Marlin "
            "supports desc_act, citing the relevant path. Otherwise future "
            "readers will treat desc_act=False as a hard constraint.",
        )


if __name__ == "__main__":
    unittest.main()
