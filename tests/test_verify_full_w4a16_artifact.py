from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools import verify_full_w4a16_artifact as verifier


TARGET_MODULES = verifier.TARGET_MODULES
GPTQ_SUFFIXES = verifier.GPTQ_SUFFIXES


def _write_artifact(root: Path, *, skip: set[str] | None = None) -> None:
    skip = skip or set()
    tensors = {}
    for layer in range(2):
        for module in TARGET_MODULES:
            base = f"model.layers.{layer}.{module}"
            if base in skip:
                tensors[f"{base}.weight"] = torch.zeros((1, 1), dtype=torch.bfloat16)
            else:
                for suffix in GPTQ_SUFFIXES:
                    tensors[f"{base}.{suffix}"] = torch.zeros((1, 1), dtype=torch.int32)
    save_file(tensors, root / "model.safetensors")


def _write_quant_config(root: Path, *, dynamic: dict | None = None) -> None:
    (root / "quantize_config.json").write_text(
        json.dumps(
            {
                "bits": 4,
                "group_size": 64,
                "quant_method": "gptq",
                "sym": True,
                "dynamic": dynamic or {
                    "-:.*o_gate$": {},
                    "-:.*z_proj$": {},
                    "-:.*o_norm$": {},
                    "-:.*q_norm$": {},
                    "-:.*k_norm$": {},
                },
            }
        ),
        encoding="utf-8",
    )


class TestVerifyFullW4A16Artifact(unittest.TestCase):
    def _run(self, root: Path, *args: str) -> int:
        ns = verifier.main.__globals__["argparse"].Namespace(
            artifact=str(root),
            mixed_skip_layers="",
            mixed_skip_modules="all",
            group_size=64,
        )
        for i, arg in enumerate(args):
            if arg == "--mixed-skip-layers":
                ns.mixed_skip_layers = args[i + 1]
            if arg == "--mixed-skip-modules":
                ns.mixed_skip_modules = args[i + 1]
        return verifier.verify(ns)

    def test_uniform_artifact_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            _write_quant_config(root)
            self.assertEqual(self._run(root), 0)

    def test_mixed_skip_down_passes_with_matching_dynamic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skip = {
                "model.layers.1.mlp.down_proj",
            }
            _write_artifact(root, skip=skip)
            _write_quant_config(
                root,
                dynamic={
                    "-:.*o_gate$": {},
                    "-:.*z_proj$": {},
                    "-:.*o_norm$": {},
                    "-:.*q_norm$": {},
                    "-:.*k_norm$": {},
                    "-:.*\\.layers\\.(1)\\.mlp\\.down_proj$": {},
                },
            )
            self.assertEqual(
                self._run(
                    root,
                    "--mixed-skip-layers",
                    "1",
                    "--mixed-skip-modules",
                    "down",
                ),
                0,
            )

    def test_bf16_weight_without_expected_skip_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root, skip={"model.layers.1.mlp.down_proj"})
            _write_quant_config(root)
            self.assertEqual(self._run(root), 1)

    def test_missing_dynamic_for_expected_skip_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root, skip={"model.layers.1.mlp.down_proj"})
            _write_quant_config(root)
            self.assertEqual(
                self._run(
                    root,
                    "--mixed-skip-layers",
                    "1",
                    "--mixed-skip-modules",
                    "down",
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
