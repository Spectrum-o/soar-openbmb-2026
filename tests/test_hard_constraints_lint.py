"""Focused tests for hard_constraints_lint preflight behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import hard_constraints_lint  # noqa: E402


class TestHardConstraintsFlashAttn(unittest.TestCase):
    def _variant_with_broken_flash_wheel(self, root: Path) -> Path:
        variant = root / "variant"
        variant.mkdir()
        (variant / "prepare_env.sh").write_text(
            '#!/usr/bin/env bash\n'
            'export SGLANG_SERVER_ARGS="--quantization gptq_marlin"\n'
        )
        (variant / "flash_attn-test.whl").symlink_to("missing.whl")
        return variant

    def test_broken_flash_attn_wheel_symlink_fails_for_gptq(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = self._variant_with_broken_flash_wheel(Path(tmp))

            result = hard_constraints_lint._check_flash_attn_installed(
                variant, hard_constraints_lint.Mode.GPTQ
            )

            self.assertEqual(result.status, hard_constraints_lint.Status.FAIL)
            self.assertIn("symlink is broken", result.message)

    def test_broken_flash_attn_wheel_symlink_fails_before_bf16_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = self._variant_with_broken_flash_wheel(Path(tmp))

            result = hard_constraints_lint._check_flash_attn_installed(
                variant, hard_constraints_lint.Mode.BF16_SYMLINK
            )

            self.assertEqual(result.status, hard_constraints_lint.Status.FAIL)
            self.assertIn("symlink is broken", result.message)


if __name__ == "__main__":
    unittest.main()
