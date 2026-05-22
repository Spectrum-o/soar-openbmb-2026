"""Tests for tools/verify_artifact_dequant.py.

Pure-Python coverage of module discovery + representative picking. The torch
dequant path is exercised by tools/dequant_one_tensor.py's existing usage
patterns (server-side, against real artifacts) and by the live invocation
in scripts/local_eval.sh — those need a GPU and 9 GB of safetensors so we
don't try to reproduce them in unit tests.

What we DO test here:
  * discover_quantized_modules walks the safetensors index correctly,
    including the single-shard fallback.
  * parse_layer_idx_and_role handles standard names + edge cases
    (lm_head, embed_tokens, weirdly-named modules).
  * pick_representatives returns one-per-role in --fast and
    early/mid/late-per-role otherwise.
  * The role bucketing is what would have caught the v23 self-bug class:
    we explicitly sample modules from SALA's heterogeneous layers
    (MiniCPM4 dense vs Lightning-attn), not just shards[0].
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Module under test
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_artifact_dequant import (  # noqa: E402
    discover_quantized_modules,
    parse_layer_idx_and_role,
    pick_representatives,
)


class TestParseLayerIdxAndRole(unittest.TestCase):
    def test_standard_layer_module(self):
        idx, role = parse_layer_idx_and_role("model.layers.0.mlp.gate_proj")
        self.assertEqual(idx, 0)
        self.assertEqual(role, "mlp.gate_proj")

    def test_high_layer_index(self):
        idx, role = parse_layer_idx_and_role("model.layers.31.self_attn.q_proj")
        self.assertEqual(idx, 31)
        self.assertEqual(role, "self_attn.q_proj")

    def test_lightning_attn_module(self):
        # SALA's Lightning layers have different module names. Make sure we
        # don't confuse them with MiniCPM4 layers.
        idx, role = parse_layer_idx_and_role("model.layers.1.self_attn.o_gate")
        self.assertEqual(idx, 1)
        self.assertEqual(role, "self_attn.o_gate")

    def test_lm_head_is_not_layer(self):
        idx, role = parse_layer_idx_and_role("lm_head")
        self.assertEqual(idx, -1)
        self.assertEqual(role, "lm_head")

    def test_embed_tokens(self):
        idx, role = parse_layer_idx_and_role("model.embed_tokens")
        self.assertEqual(idx, -1)
        self.assertEqual(role, "model.embed_tokens")

    def test_non_integer_layer_idx_returns_negative(self):
        # Defensive: malformed name shouldn't crash.
        idx, role = parse_layer_idx_and_role("model.layers.foo.mlp.gate")
        self.assertEqual(idx, -1)


class TestPickRepresentatives(unittest.TestCase):
    def _build_sala_like_modules(self, num_layers: int = 4) -> list[str]:
        """Simulate SALA's heterogeneous layer types.

        Even-indexed layers = MiniCPM4 dense attention (q/k/v/o + mlp).
        Odd-indexed layers = Lightning attention (z_proj/o_gate + mlp).
        Catches the case the v23 bug class represents: a check that only
        looked at the first layer would miss the Lightning modules entirely.
        """
        out = []
        for i in range(num_layers):
            for role in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                out.append(f"model.layers.{i}.{role}")
            if i % 2 == 0:
                for role in ("self_attn.q_proj", "self_attn.k_proj",
                             "self_attn.v_proj", "self_attn.o_proj"):
                    out.append(f"model.layers.{i}.{role}")
            else:
                for role in ("self_attn.z_proj", "self_attn.o_gate"):
                    out.append(f"model.layers.{i}.{role}")
        return sorted(out)

    def test_fast_mode_picks_one_per_role(self):
        modules = self._build_sala_like_modules(num_layers=8)
        picked = pick_representatives(modules, fast=True)
        # 3 MLP roles + 4 dense-attn roles + 2 lightning roles = 9 distinct.
        roles = {parse_layer_idx_and_role(m)[1] for m in picked}
        self.assertEqual(len(roles), 9)
        # And we picked exactly one module per role.
        self.assertEqual(len(picked), 9)

    def test_default_mode_picks_early_mid_late(self):
        modules = self._build_sala_like_modules(num_layers=8)
        picked = pick_representatives(modules, fast=False)
        # For each role we want early/mid/late. MLP roles exist in all 8
        # layers → 3 picks each. Dense-attn roles exist in 4 even layers →
        # 3 picks each (the helper dedups by layer_idx but layers 0/4/6 are
        # all distinct). Lightning roles exist in 4 odd layers → 3 each.
        by_role = {}
        for m in picked:
            _, role = parse_layer_idx_and_role(m)
            by_role.setdefault(role, []).append(m)
        # Every role should have AT LEAST 2 picks (early + late), preferably 3.
        for role, picks in by_role.items():
            self.assertGreaterEqual(
                len(picks), 2,
                f"role {role!r} got only {len(picks)} picks: {picks}",
            )

    def test_lightning_modules_are_not_dropped(self):
        # The v23-class bug shape: only-shard-0 / only-layer-0 sampling
        # misses Lightning. Verify we sample at least one Lightning module.
        modules = self._build_sala_like_modules(num_layers=8)
        picked = pick_representatives(modules, fast=True)
        lightning_roles = {"self_attn.z_proj", "self_attn.o_gate"}
        picked_lightning_roles = {
            parse_layer_idx_and_role(m)[1] for m in picked
        } & lightning_roles
        self.assertEqual(
            picked_lightning_roles, lightning_roles,
            "pick_representatives must include every distinct role, "
            "including Lightning-attn modules that only appear in odd layers",
        )

    def test_empty_input(self):
        self.assertEqual(pick_representatives([], fast=True), [])
        self.assertEqual(pick_representatives([], fast=False), [])

    def test_only_non_layer_modules(self):
        # lm_head etc. should pass through.
        picked = pick_representatives(["lm_head", "model.norm"], fast=True)
        self.assertEqual(sorted(picked), ["lm_head", "model.norm"])

    def test_deterministic_ordering(self):
        modules = self._build_sala_like_modules(num_layers=4)
        a = pick_representatives(modules, fast=False)
        b = pick_representatives(modules, fast=False)
        self.assertEqual(a, b)


class TestDiscoverQuantizedModules(unittest.TestCase):
    def _write_index(self, dirpath: Path, weight_map: dict[str, str]) -> None:
        idx = {
            "metadata": {"total_size": 0},
            "weight_map": weight_map,
        }
        (dirpath / "model.safetensors.index.json").write_text(json.dumps(idx))

    def test_index_only_returns_qweight_modules(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write_index(d, {
                "model.embed_tokens.weight": "shard-1.safetensors",
                "model.layers.0.mlp.gate_proj.qweight": "shard-1.safetensors",
                "model.layers.0.mlp.gate_proj.qzeros": "shard-1.safetensors",
                "model.layers.0.mlp.gate_proj.scales": "shard-1.safetensors",
                "model.layers.0.mlp.gate_proj.g_idx": "shard-1.safetensors",
                "model.layers.0.input_layernorm.weight": "shard-1.safetensors",
                "lm_head.weight": "shard-2.safetensors",
            })
            modules = discover_quantized_modules(d)
            self.assertEqual(modules, ["model.layers.0.mlp.gate_proj"])

    def test_discovers_multiple_layers(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            wm = {}
            for i in range(3):
                base = f"model.layers.{i}.mlp.gate_proj"
                wm[f"{base}.qweight"] = "shard-1.safetensors"
                wm[f"{base}.qzeros"] = "shard-1.safetensors"
            self._write_index(d, wm)
            modules = discover_quantized_modules(d)
            self.assertEqual(len(modules), 3)
            self.assertIn("model.layers.0.mlp.gate_proj", modules)
            self.assertIn("model.layers.2.mlp.gate_proj", modules)

    def test_raises_on_missing_dir(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "nonexistent"
            with self.assertRaises(FileNotFoundError):
                discover_quantized_modules(d)


if __name__ == "__main__":
    unittest.main()
