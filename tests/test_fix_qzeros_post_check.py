#!/usr/bin/env python3
"""Tests for fix_qzeros_for_marlin POST-CHECK on real shard distributions.

The 2026-05-22 v23 platform submission FAILED because POST-CHECK iterated
shards[0] blindly, but for MLP-only artifacts shard 0 has embedding + lm_head
with NO .qzeros tensors. This test reproduces the EXACT failure scenario
LOCALLY and asserts the fix (commit 551826514) actually works.

Run:
    python3 -m unittest tests.test_fix_qzeros_post_check -v
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _import_quantize_module(submission_dir: str):
    """Import quantize_gptqmodel_w4a16.py from a submission dir.

    Stubs `transformers` and `transformers.integrations.hub_kernels` at
    import time so the module-level `_stub_transformers_for_gptqmodel_7()`
    call doesn't fail when transformers isn't installed locally.
    """
    import builtins
    real_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        if name == "transformers":
            class _StubTransformers:
                class PretrainedConfig:
                    pass
                PreTrainedConfig = PretrainedConfig
            return _StubTransformers()
        if name.startswith("transformers.integrations"):
            class _StubHK:
                _gptqmodel_local_causal_conv1d_kernel = True
            return _StubHK()
        return real_import(name, *args, **kwargs)

    builtins.__import__ = stub_import
    try:
        spec = importlib.util.spec_from_file_location(
            f"quant_{submission_dir.replace('/', '_')}",
            REPO_ROOT / submission_dir / "quantize_gptqmodel_w4a16.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        builtins.__import__ = real_import


def _build_v23_style_artifact(out_dir: Path, n_layers_per_shard: tuple = (0, 2, 1)):
    """Create a mock W4A16-MLP-only artifact with the same shard distribution
    that caused the v23 platform bug.

    Args:
        out_dir: where to write the safetensors shards
        n_layers_per_shard: (shard1, shard2, shard3) — number of MLP layer
            triples per shard. shard1=0 means no qzeros in shard 1
            (embeddings only), which is the bug-triggering layout.
    """
    import torch
    from safetensors.torch import save_file

    BAD = 0x77777777  # signed int32: 2004318071

    # Shard 1: embedding + lm_head + (optionally) early MLP layers
    s1 = {
        "model.embed_tokens.weight": torch.zeros(100, 64, dtype=torch.float16),
        "lm_head.weight": torch.zeros(100, 64, dtype=torch.float16),
        "model.norm.weight": torch.ones(64, dtype=torch.float16),
    }
    for i in range(n_layers_per_shard[0]):
        for p in ("gate_proj", "up_proj", "down_proj"):
            s1[f"model.layers.{i}.mlp.{p}.qzeros"] = torch.full((4, 8), BAD, dtype=torch.int32)
    save_file(s1, str(out_dir / "model-00001-of-00003.safetensors"))

    # Shard 2: middle layers' MLP qzeros
    s2 = {}
    start = n_layers_per_shard[0]
    for i in range(start, start + n_layers_per_shard[1]):
        for p in ("gate_proj", "up_proj", "down_proj"):
            s2[f"model.layers.{i}.mlp.{p}.qzeros"] = torch.full((4, 8), BAD, dtype=torch.int32)
    save_file(s2, str(out_dir / "model-00002-of-00003.safetensors"))

    # Shard 3: tail layers' MLP qzeros
    s3 = {}
    start = n_layers_per_shard[0] + n_layers_per_shard[1]
    for i in range(start, start + n_layers_per_shard[2]):
        for p in ("gate_proj", "up_proj", "down_proj"):
            s3[f"model.layers.{i}.mlp.{p}.qzeros"] = torch.full((4, 8), BAD, dtype=torch.int32)
    save_file(s3, str(out_dir / "model-00003-of-00003.safetensors"))


def _all_qzeros_good(out_dir: Path, expected: int = -2004318072) -> bool:
    """Reload every shard, check every .qzeros tensor has expected bias."""
    import torch
    from safetensors.torch import load_file

    for shard in sorted(out_dir.glob("model-*.safetensors")):
        state = load_file(str(shard))
        for k, v in state.items():
            if not k.endswith(".qzeros"):
                continue
            if v.dtype == torch.uint32:
                v = v.view(torch.int32)
            first = int(v.flatten()[0].item())
            if first != expected:
                return False
    return True


class TestFixQzerosV23ShardLayout(unittest.TestCase):
    """The exact failure scenario that bricked v23 on the platform 2026-05-22."""

    def setUp(self):
        try:
            import torch  # noqa
            import safetensors  # noqa
        except ImportError:
            self.skipTest("torch + safetensors not installed")

    def test_v21_v23_mlp_only_shard_distribution_passes(self):
        """Shard 1 has NO qzeros (just embedding+lm_head), shards 2+3 have
        all the MLP qzeros. Before fix 551826514, POST-CHECK iterated
        shards[0] and never set verified=True → spurious FATAL."""
        mod = _import_quantize_module("submission_gptqmodel_calib_w4a16")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "artifact"
            out.mkdir()
            _build_v23_style_artifact(out, n_layers_per_shard=(0, 2, 1))
            # Should NOT raise
            try:
                mod.fix_qzeros_for_marlin(out)
            except RuntimeError as e:
                self.fail(f"fix_qzeros raised on v23-style layout: {e}")
            # And all qzeros should now be 0x88888888
            self.assertTrue(_all_qzeros_good(out),
                            "qzeros not all patched to 0x88888888 after fix")

    def test_v23_layout_v22_source_also_passes(self):
        """Same scenario via the v22 source's quantize script (which got
        the same POST-CHECK fix in commit 551826514)."""
        mod = _import_quantize_module("submission_gptq_v17_minconfig")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "artifact"
            out.mkdir()
            _build_v23_style_artifact(out, n_layers_per_shard=(0, 2, 1))
            try:
                mod.fix_qzeros_for_marlin(out)
            except RuntimeError as e:
                self.fail(f"fix_qzeros (v22 source) raised on v23-style layout: {e}")
            self.assertTrue(_all_qzeros_good(out))

    def test_v24_no_dtype_key_also_passes(self):
        """The v24_no_dtype_key variant has its own copy of the quantize
        script (since it's the only file that differs from v23). Verify
        its POST-CHECK is also fixed."""
        mod = _import_quantize_module("submission_gptqmodel_calib_w4a16_v24_no_dtype_key")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "artifact"
            out.mkdir()
            _build_v23_style_artifact(out, n_layers_per_shard=(0, 2, 1))
            try:
                mod.fix_qzeros_for_marlin(out)
            except RuntimeError as e:
                self.fail(f"fix_qzeros (v24_no_dtype_key) raised: {e}")
            self.assertTrue(_all_qzeros_good(out))

    def test_qzeros_in_shard_zero_also_passes(self):
        """If a future quant config DOES put qzeros in shard 0 (e.g. full-attn
        with smaller embedding), the fix should still work. shard 0 has 1
        layer's qzeros + the embedding."""
        mod = _import_quantize_module("submission_gptqmodel_calib_w4a16")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "artifact"
            out.mkdir()
            _build_v23_style_artifact(out, n_layers_per_shard=(1, 1, 1))
            try:
                mod.fix_qzeros_for_marlin(out)
            except RuntimeError as e:
                self.fail(f"fix_qzeros raised on shard-0-has-qzeros layout: {e}")
            self.assertTrue(_all_qzeros_good(out))

    def test_all_qzeros_already_good_passes(self):
        """If qzeros are ALREADY 0x88888888 (rerun on a previously-patched
        artifact), per_shard_patched will be empty. POST-CHECK should fall
        back to iterating all shards and still find a qzeros to verify."""
        import torch
        from safetensors.torch import save_file
        GOOD = -2004318072

        mod = _import_quantize_module("submission_gptqmodel_calib_w4a16")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "artifact"
            out.mkdir()
            save_file({
                "model.embed_tokens.weight": torch.zeros(100, 64, dtype=torch.float16),
            }, str(out / "model-00001-of-00003.safetensors"))
            save_file({
                "model.layers.0.mlp.gate_proj.qzeros": torch.full((4, 8), GOOD, dtype=torch.int32),
                "model.layers.0.mlp.up_proj.qzeros": torch.full((4, 8), GOOD, dtype=torch.int32),
            }, str(out / "model-00002-of-00003.safetensors"))
            save_file({
                "model.layers.5.mlp.down_proj.qzeros": torch.full((4, 8), GOOD, dtype=torch.int32),
            }, str(out / "model-00003-of-00003.safetensors"))
            try:
                mod.fix_qzeros_for_marlin(out)
            except RuntimeError as e:
                self.fail(f"fix_qzeros raised on already-good artifact: {e}")
            self.assertTrue(_all_qzeros_good(out))

    def test_bits8_qzeros_are_patched_to_0x80808080(self):
        """v24_bits8 uses uint8b128, so symmetric qzeros should be 128.

        GPTQModel's off-by-one pattern generalizes from 4-bit 7->8 to
        8-bit 127->128. The previous fix only recognized 0x77777777 and
        would FATAL on an 8-bit diagnostic artifact.
        """
        import torch
        from safetensors.torch import save_file

        BAD8 = 0x7F7F7F7F
        GOOD8 = -2139062144  # 0x80808080 as signed int32

        mod = _import_quantize_module("submission_gptqmodel_calib_w4a16")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "artifact"
            out.mkdir()
            save_file({
                "model.embed_tokens.weight": torch.zeros(100, 64, dtype=torch.float16),
            }, str(out / "model-00001-of-00003.safetensors"))
            save_file({
                "model.layers.0.mlp.gate_proj.qzeros": torch.full((4, 8), BAD8, dtype=torch.int32),
                "model.layers.0.mlp.up_proj.qzeros": torch.full((4, 8), BAD8, dtype=torch.int32),
                "model.layers.0.mlp.down_proj.qzeros": torch.full((4, 8), BAD8, dtype=torch.int32),
            }, str(out / "model-00002-of-00003.safetensors"))
            try:
                mod.fix_qzeros_for_marlin(out)
            except RuntimeError as e:
                self.fail(f"fix_qzeros raised on bits8 qzeros layout: {e}")
            self.assertTrue(_all_qzeros_good(out, expected=GOOD8),
                            "bits8 qzeros not patched to 0x80808080")


if __name__ == "__main__":
    unittest.main()
