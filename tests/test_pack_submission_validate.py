#!/usr/bin/env python3
"""Unit tests for tools/pack_submission.validate_variant.

Run from repo root:
    python3 -m unittest tests.test_pack_submission_validate -v

Mirrors tests/test_calibration_tools.py style: unittest + tempfile, no
external deps. Each test builds a minimal fake variant dir in tempfile,
mutates one fix (or omits it), then asserts validate_variant flags it.

The tests are designed to FAIL if any of the 2026-05-21..22 fix canaries
get accidentally removed from a future quantize_gptqmodel_w4a16.py
refactor — which is exactly the kind of silent regression that
re-introduced the platform=0 bug class.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import pack_submission  # noqa: E402


# Reference content fragments: a variant that PASSES every check (used as
# the baseline; each test mutates one piece and re-runs validate_variant).
GOOD_QUANTIZE_SCRIPT = '''"""Stub for testing."""
def fix_qzeros_for_marlin(output_dir):
    """ HARDENED: ... POST-CHECK ... """
    pass

def copy_runtime_assets(input_dir, output_dir):
    """CRITICAL 2026-05-22: this function now OVERWRITES files GPTQModel
    already wrote during model.save(). Previously had `if not target.exists()`
    guard. Removed.
    """
    pass

def tokenize_calibration(tokenizer, prompts, max_len):
    tokenizer.truncation_side = "left"
    return []
'''

GOOD_PREPARE_ENV = '''#!/usr/bin/env bash
GPTQMODEL_PIN="${GPTQMODEL_PIN:-7.0.0}"
export SGLANG_SERVER_ARGS="--disable-radix-cache --chunked-prefill-size 8192 --quantization gptq_marlin --dtype float16"
'''

GOOD_PREPARE_MODEL = '''#!/usr/bin/env bash
echo "[prepare_model] DIAGNOSTIC: script dir contents"
ls -la $SCRIPT_DIR
echo "[prepare_model] DIAGNOSTIC: calib jsonl size"
'''


def _make_variant(parent: Path, *, quant_script: str | None = GOOD_QUANTIZE_SCRIPT,
                  prepare_env: str | None = GOOD_PREPARE_ENV,
                  prepare_model: str | None = GOOD_PREPARE_MODEL,
                  with_calib: bool = True,
                  with_bundled_sglang: bool = True) -> Path:
    """Build a synthetic variant dir. Pass None for any file to omit it.

    `with_bundled_sglang` controls whether to create a synthetic
    sglang/python/sglang/srt/{configs,layers}/ subtree with both
    defensive patches present (kwargs.pop in minicpm.py, early-return
    in torchao_utils.py). Set to False to test the missing-bundle
    failure branches.
    """
    d = parent / "fake_variant"
    d.mkdir()
    if quant_script is not None:
        (d / "quantize_gptqmodel_w4a16.py").write_text(quant_script)
    if prepare_env is not None:
        (d / "prepare_env.sh").write_text(prepare_env)
    if prepare_model is not None:
        (d / "prepare_model.sh").write_text(prepare_model)
    if with_calib:
        (d / "perf_public_set.jsonl").write_text('{"task":"x"}\n')
    if with_bundled_sglang and quant_script is not None:
        # Create minimal bundled sglang with the two defensive patches' canaries.
        bundled = d / "sglang" / "python" / "sglang" / "srt"
        (bundled / "configs").mkdir(parents=True)
        (bundled / "layers" / "attention").mkdir(parents=True)
        (bundled / "configs" / "minicpm.py").write_text(
            '"""minicpm.py stub with kwargs.pop patch canary"""\n'
            'class Cfg:\n'
            '    def __init__(self, **kwargs):\n'
            '        for k in ("has_sparse_attention",):\n'
            '            kwargs.pop(k, None)\n'
        )
        (bundled / "layers" / "torchao_utils.py").write_text(
            '"""torchao_utils.py stub with early-return canary"""\n'
            'def apply(torchao_config):\n'
            '    if torchao_config == "" or torchao_config is None:\n'
            '        return\n'
        )
        # Minimal attention stub files for sed-patch targets
        (bundled / "layers" / "attention" / "minicpm_backend.py").write_text(
            '# stub for sed-patch target\n'
        )
        (bundled / "layers" / "attention" / "minicpm_sparse_utils.py").write_text(
            '# stub for sed-patch target\n'
        )
    return d


def _has_problem_containing(problems: list[str], needle: str) -> bool:
    return any(needle in p for p in problems)


class TestValidateVariantGood(unittest.TestCase):
    """The "everything correct" baseline should produce no problems."""

    def test_baseline_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(Path(tmp))
            problems = pack_submission.validate_variant(variant)
            self.assertEqual(problems, [],
                             f"unexpected problems on good variant: {problems}")


class TestValidateVariantQuantScript(unittest.TestCase):
    """Check the 4 GPTQ-script canaries fire when removed."""

    def test_missing_fix_qzeros_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                quant_script='''"""no fix"""
def copy_runtime_assets(): """CRITICAL 2026-05-22: this function now OVERWRITES"""
def tokenize_calibration(tokenizer): tokenizer.truncation_side = "left"
''',
            )
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, "fix_qzeros_for_marlin"),
                            f"expected qzeros fix problem, got: {problems}")

    def test_missing_post_check_canary(self):
        # qzeros function present but not hardened (no POST-CHECK marker)
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                quant_script='''
def fix_qzeros_for_marlin(): pass
def copy_runtime_assets(): """CRITICAL 2026-05-22: this function now OVERWRITES"""
def tokenize_calibration(tokenizer): tokenizer.truncation_side = "left"
''',
            )
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, "POST-CHECK"),
                            f"expected POST-CHECK problem, got: {problems}")

    def test_missing_h4_canary(self):
        # buggy copy_runtime_assets: no CRITICAL 2026-05-22 marker
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                quant_script='''
def fix_qzeros_for_marlin(): """POST-CHECK"""
def copy_runtime_assets(input_dir, output_dir):
    """ old version without H4 marker """
    for p in input_dir.iterdir():
        if not (output_dir / p.name).exists():
            shutil.copy2(p, output_dir / p.name)
def tokenize_calibration(tokenizer): tokenizer.truncation_side = "left"
''',
            )
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, "copy_runtime_assets fix canary"),
                            f"expected H4 canary problem, got: {problems}")

    def test_missing_truncation_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                quant_script='''
def fix_qzeros_for_marlin(): """POST-CHECK"""
def copy_runtime_assets(): """CRITICAL 2026-05-22: this function now OVERWRITES"""
def tokenize_calibration(): pass
''',
            )
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, 'truncation_side="left"'),
                            f"expected truncation_side problem, got: {problems}")


class TestValidateVariantPrepareEnv(unittest.TestCase):
    def test_missing_prepare_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(Path(tmp), prepare_env=None)
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, "missing prepare_env.sh"))

    def test_missing_sglang_server_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                prepare_env='#!/usr/bin/env bash\nGPTQMODEL_PIN=7.0.0\n',
            )
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, "SGLANG_SERVER_ARGS"))

    def test_missing_gptqmodel_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                prepare_env='#!/usr/bin/env bash\nexport SGLANG_SERVER_ARGS="--quantization gptq_marlin --chunked-prefill-size 8192"\n',
            )
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, "gptqmodel not pinned"))

    def test_chunked_prefill_65k_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                prepare_env='''#!/usr/bin/env bash
GPTQMODEL_PIN="7.0.0"
export SGLANG_SERVER_ARGS="--chunked-prefill-size 65536 --max-prefill-tokens 65536 --mem-fraction-static 0.80 --quantization gptq_marlin"
''',
            )
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(
                _has_problem_containing(problems, "chunked-prefill-size 65536"),
                f"expected chunked-prefill 65K poisoning problem, got: {problems}"
            )

    def test_chunked_prefill_8k_passes(self):
        # Same as good baseline but make explicit: 8192 is the correct value for v23
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(Path(tmp))
            problems = pack_submission.validate_variant(variant)
            self.assertFalse(
                _has_problem_containing(problems, "chunked-prefill-size 65536"),
                f"8k should not trip the 65k check; got: {problems}"
            )

    def test_chunked_prefill_65k_whitelisted_for_v24_variants(self):
        # v24+ perf forks are explicitly allowed to ship chunked-prefill 65K
        # since they layer it on top of v23's confirmed-correctness baseline.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            for name in ("fake_v24", "fake_v25", "fake_perf", "fake_chunk65k"):
                d = tmp_p / name
                d.mkdir()
                (d / "quantize_gptqmodel_w4a16.py").write_text(GOOD_QUANTIZE_SCRIPT)
                (d / "prepare_env.sh").write_text(
                    '#!/usr/bin/env bash\n'
                    'GPTQMODEL_PIN="${GPTQMODEL_PIN:-7.0.0}"\n'
                    'export SGLANG_SERVER_ARGS="--chunked-prefill-size 65536 '
                    '--max-prefill-tokens 65536 --mem-fraction-static 0.80 '
                    '--quantization gptq_marlin"\n'
                )
                (d / "prepare_model.sh").write_text(GOOD_PREPARE_MODEL)
                (d / "perf_public_set.jsonl").write_text('{"task":"x"}\n')
                problems = pack_submission.validate_variant(d)
                self.assertFalse(
                    _has_problem_containing(problems, "chunked-prefill-size 65536"),
                    f"variant '{name}' should be whitelisted for 65K; got: {problems}"
                )

    def test_chunked_prefill_65k_flagged_for_v23_variant_names(self):
        # Variant names not matching the v24+ pattern still trip the rule.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            for name in ("submission_gptqmodel_calib_w4a16", "fake_baseline",
                         "experimental_attempt"):
                d = tmp_p / name
                d.mkdir()
                (d / "quantize_gptqmodel_w4a16.py").write_text(GOOD_QUANTIZE_SCRIPT)
                (d / "prepare_env.sh").write_text(
                    '#!/usr/bin/env bash\n'
                    'GPTQMODEL_PIN="${GPTQMODEL_PIN:-7.0.0}"\n'
                    'export SGLANG_SERVER_ARGS="--chunked-prefill-size 65536 '
                    '--quantization gptq_marlin"\n'
                )
                (d / "prepare_model.sh").write_text(GOOD_PREPARE_MODEL)
                (d / "perf_public_set.jsonl").write_text('{"task":"x"}\n')
                problems = pack_submission.validate_variant(d)
                self.assertTrue(
                    _has_problem_containing(problems, "chunked-prefill-size 65536"),
                    f"variant '{name}' should still trip 65K rule; got: {problems}"
                )


class TestValidateVariantPrepareModel(unittest.TestCase):
    def test_missing_prepare_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(Path(tmp), prepare_model=None)
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, "missing prepare_model.sh"))

    def test_missing_diagnostic_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                prepare_model='#!/usr/bin/env bash\necho "no diagnostic prints here"\n',
            )
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, "DIAGNOSTIC:"))


class TestValidateVariantBF16Mode(unittest.TestCase):
    """No quantize_*.py file → BF16 / identity path. Only env/model required."""

    def test_bf16_minimal_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                quant_script=None,
                prepare_env='#!/usr/bin/env bash\nexport SGLANG_SERVER_ARGS="--dense-as-sparse"\n',
                with_calib=False,
            )
            problems = pack_submission.validate_variant(variant)
            self.assertEqual(problems, [],
                             f"unexpected problems for BF16 variant: {problems}")

    def test_bf16_missing_prepare_env_still_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                quant_script=None,
                prepare_env=None,
                with_calib=False,
            )
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(_has_problem_containing(problems, "missing prepare_env.sh"))


class TestValidateVariantBundledSglang(unittest.TestCase):
    """Verify the bundled-sglang preflight checks (added 2026-05-22 after
    v23 was packed without sglang/ and the platform crashed)."""

    def test_missing_bundled_sglang_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(Path(tmp), with_bundled_sglang=False)
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(
                _has_problem_containing(problems, "sglang/python/ missing"),
                f"missing bundled sglang should be flagged; got: {problems}"
            )

    def test_missing_minicpm_kwargs_pop_patch_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(Path(tmp), with_bundled_sglang=True)
            # Corrupt the canary: rewrite minicpm.py without the patch
            (variant / "sglang" / "python" / "sglang" / "srt" / "configs"
             / "minicpm.py").write_text('# no patch here\n')
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(
                _has_problem_containing(problems, "kwargs.pop defensive patch"),
                f"missing kwargs.pop should be flagged; got: {problems}"
            )

    def test_missing_torchao_early_return_patch_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(Path(tmp), with_bundled_sglang=True)
            (variant / "sglang" / "python" / "sglang" / "srt" / "layers"
             / "torchao_utils.py").write_text('# no patch here\n')
            problems = pack_submission.validate_variant(variant)
            self.assertTrue(
                _has_problem_containing(problems, "early-return patch"),
                f"missing torchao early-return should be flagged; got: {problems}"
            )

    def test_bf16_variant_does_not_require_bundled_sglang(self):
        # BF16 path has no quantize script -> sglang bundle check skipped
        with tempfile.TemporaryDirectory() as tmp:
            variant = _make_variant(
                Path(tmp),
                quant_script=None,
                prepare_env='#!/usr/bin/env bash\nexport SGLANG_SERVER_ARGS="--dense-as-sparse"\n',
                with_calib=False,
                with_bundled_sglang=False,
            )
            problems = pack_submission.validate_variant(variant)
            self.assertFalse(
                _has_problem_containing(problems, "sglang/python/ missing"),
                f"BF16 variant should not require bundled sglang; got: {problems}"
            )


class TestValidateVariantOnLiveSources(unittest.TestCase):
    """Run validate_variant on the live submission_*/ dirs in this repo.
    Both should currently pass (assuming the 2026-05-22 fixes are in place).
    Functions as a sentinel: if a future refactor silently regresses one
    of the fixes, this test will catch it before the next submission pack.
    """

    def test_v21_source_clean(self):
        v21 = REPO_ROOT / "submission_gptqmodel_calib_w4a16"
        if not v21.is_dir():
            self.skipTest("v21 variant dir not present locally")
        problems = pack_submission.validate_variant(v21)
        self.assertEqual(problems, [],
                         f"v21 variant should be clean post-fixes; got: {problems}")

    def test_v22_source_clean(self):
        v22 = REPO_ROOT / "submission_gptq_v17_minconfig"
        if not v22.is_dir():
            self.skipTest("v22 variant dir not present locally")
        problems = pack_submission.validate_variant(v22)
        self.assertEqual(problems, [],
                         f"v22 variant should be clean post-fixes; got: {problems}")


if __name__ == "__main__":
    unittest.main()
