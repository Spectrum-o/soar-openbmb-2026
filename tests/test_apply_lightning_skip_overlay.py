#!/usr/bin/env python3
"""Unit tests for tools/apply_lightning_skip_overlay.py — no GPU.

Tests the index / config manipulation logic plus a tiny safetensors rewrite
fixture. The risky parts of the tool that benefit from automated testing are:
  - find_lightning_layer_indices() — config parsing
  - remove_quantized_lightning_tensors_from_index() — index surgery
  - rewrite_safetensors_without_tensors() — physical shard cleanup for SGLang
  - update_quantize_config_dynamic() — config patch

Validates them with synthetic test fixtures: tmp config.json,
model.safetensors.index.json, quantize_config.json, and tiny safetensors files.

Run:
    python3 -m unittest tests.test_apply_lightning_skip_overlay
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

# Import the module under test
import apply_lightning_skip_overlay as overlay  # noqa: E402


# Realistic SALA-like mixer_types: 8 dense + 24 lightning, interleaved
SALA_LIKE_MIXER_TYPES = [
    "minicpm4" if i % 4 == 0 else "lightning-attn"
    for i in range(32)
]


def write_config_json(path: Path, mixer_types: list[str], num_hidden_layers: int | None = None) -> None:
    cfg = {
        "model_type": "minicpm_sala",
        "mixer_types": mixer_types,
        "num_hidden_layers": num_hidden_layers or len(mixer_types),
        "auto_map": {"AutoConfig": "configuration_minicpm_sala.MiniCPMSALAConfig"},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f)


def write_safetensors_index(
    path: Path, layer_count: int, has_attn: bool = True, fmt: str = "gptq_marlin"
) -> dict:
    """Create a fake model.safetensors.index.json matching a quantized SALA artifact."""
    weight_map: dict[str, str] = {
        "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
        "model.norm.weight": "model-00003-of-00003.safetensors",
        "lm_head.weight": "model-00003-of-00003.safetensors",
    }

    suffixes = (
        (".qweight", ".qzeros", ".scales")
        if fmt == "gptq_marlin"
        else (".weight_packed", ".weight_scale", ".weight_zero_point")
    )

    for i in range(layer_count):
        shard = f"model-{(i // 12) + 1:05d}-of-00003.safetensors"
        # MLP quantized tensors (the things to be replaced for lightning layers)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suf in suffixes:
                weight_map[f"model.layers.{i}.mlp.{proj}{suf}"] = shard
        if has_attn:
            # attention BF16 (unquantized — already in MLP-only setup)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                weight_map[f"model.layers.{i}.self_attn.{proj}.weight"] = shard
        # layer norms
        weight_map[f"model.layers.{i}.input_layernorm.weight"] = shard
        weight_map[f"model.layers.{i}.post_attention_layernorm.weight"] = shard

    idx = {"metadata": {"total_size": 5_000_000_000}, "weight_map": weight_map}
    with path.open("w", encoding="utf-8") as f:
        json.dump(idx, f)
    return idx


def write_quantize_config(path: Path, fmt: str = "gptq_marlin") -> None:
    if fmt == "gptq_marlin":
        cfg = {
            "bits": 4,
            "group_size": 128,
            "quant_method": "gptq",
            "desc_act": False,
            "sym": True,
            "lm_head": False,
            "dynamic": {
                "-:.*self_attn.*": True,
            },
        }
    else:
        cfg = {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "group_size": 128, "symmetric": False},
                }
            },
            "format": "pack-quantized",
            "ignore": ["lm_head"],
            "quant_method": "compressed-tensors",
            "dynamic": {
                "-:.*self_attn.*": True,
            },
        }
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f)


class TestFindLightningLayers(unittest.TestCase):
    def test_typical_sala_mixer_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            write_config_json(cfg_path, SALA_LIKE_MIXER_TYPES)
            indices, total = overlay.find_lightning_layer_indices(cfg_path)
            # Every layer not at a multiple of 4 should be lightning
            expected = [i for i in range(32) if i % 4 != 0]
            self.assertEqual(indices, expected)
            self.assertEqual(total, 32)

    def test_all_lightning(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            write_config_json(cfg_path, ["lightning-attn"] * 8)
            indices, total = overlay.find_lightning_layer_indices(cfg_path)
            self.assertEqual(indices, list(range(8)))
            self.assertEqual(total, 8)

    def test_no_lightning(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            write_config_json(cfg_path, ["minicpm4"] * 8)
            indices, total = overlay.find_lightning_layer_indices(cfg_path)
            self.assertEqual(indices, [])
            self.assertEqual(total, 8)

    def test_missing_mixer_types_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            # config without mixer_types
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump({"model_type": "minicpm_sala", "num_hidden_layers": 8}, f)
            with self.assertRaises(RuntimeError):
                overlay.find_lightning_layer_indices(cfg_path)

    def test_missing_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                overlay.find_lightning_layer_indices(Path(tmp) / "config.json")

    def test_lightning_marker_variants(self):
        # Should recognize all of these as lightning
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            mixer_types = ["minicpm4", "lightning-attn", "Lightning", "GATED_DELTA", "linear_attn"]
            write_config_json(cfg_path, mixer_types)
            indices, _ = overlay.find_lightning_layer_indices(cfg_path)
            self.assertEqual(indices, [1, 2, 3, 4])


class TestSelectorParsing(unittest.TestCase):
    def test_last_n_lightning_selector(self):
        selected = overlay.parse_layer_selector(
            "last4-lightning", lightning_indices=[1, 2, 3, 5, 7, 11], total_layers=12
        )
        self.assertEqual(selected, [3, 5, 7, 11])

    def test_explicit_layers_and_ranges(self):
        selected = overlay.parse_layer_selector(
            "23,27,29-31", lightning_indices=[], total_layers=32
        )
        self.assertEqual(selected, [23, 27, 29, 30, 31])

    def test_all_selector(self):
        selected = overlay.parse_layer_selector(
            "all", lightning_indices=[1, 2], total_layers=4
        )
        self.assertEqual(selected, [0, 1, 2, 3])

    def test_rejects_out_of_range_layer(self):
        with self.assertRaises(ValueError):
            overlay.parse_layer_selector("31-32", lightning_indices=[], total_layers=32)

    def test_module_group_aliases(self):
        self.assertEqual(overlay.parse_module_groups("attention,down"), ("attn", "down_proj"))


class TestRemoveQuantizedLightningFromIndex(unittest.TestCase):
    def test_gptq_marlin_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            idx_path = qdir / "model.safetensors.index.json"
            original_idx = write_safetensors_index(idx_path, layer_count=8, fmt="gptq_marlin")
            lightning_indices = [1, 3, 5, 7]

            removed = overlay.remove_quantized_lightning_tensors_from_index(qdir, lightning_indices)

            # 4 layers × 3 projs × 3 suffixes (qweight, qzeros, scales) = 36 tensors
            self.assertEqual(len(removed), 4 * 3 * 3)
            # Verify the named tensors are NO LONGER in the index
            with idx_path.open() as f:
                new_idx = json.load(f)
            for layer in lightning_indices:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    self.assertNotIn(f"model.layers.{layer}.mlp.{proj}.qweight", new_idx["weight_map"])
                    self.assertNotIn(f"model.layers.{layer}.mlp.{proj}.qzeros", new_idx["weight_map"])
                    self.assertNotIn(f"model.layers.{layer}.mlp.{proj}.scales", new_idx["weight_map"])
            # Dense layers (0, 2, 4, 6) should still be in the index
            for layer in (0, 2, 4, 6):
                self.assertIn(f"model.layers.{layer}.mlp.gate_proj.qweight", new_idx["weight_map"])
            # Attention untouched
            self.assertIn("model.layers.0.self_attn.q_proj.weight", new_idx["weight_map"])
            self.assertIn("model.layers.7.self_attn.o_proj.weight", new_idx["weight_map"])

    def test_compressed_tensors_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            idx_path = qdir / "model.safetensors.index.json"
            write_safetensors_index(idx_path, layer_count=4, fmt="compressed_tensors")
            lightning_indices = [0, 2]

            removed = overlay.remove_quantized_lightning_tensors_from_index(qdir, lightning_indices)

            # 2 layers × 3 projs × 3 ct suffixes = 18 tensors
            self.assertEqual(len(removed), 2 * 3 * 3)
            with idx_path.open() as f:
                new_idx = json.load(f)
            for layer in lightning_indices:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    self.assertNotIn(
                        f"model.layers.{layer}.mlp.{proj}.weight_packed",
                        new_idx["weight_map"],
                    )

    def test_empty_lightning_indices_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            idx_path = qdir / "model.safetensors.index.json"
            original_idx = write_safetensors_index(idx_path, layer_count=4)
            n_before = len(original_idx["weight_map"])

            removed = overlay.remove_quantized_lightning_tensors_from_index(qdir, [])
            self.assertEqual(len(removed), 0)

            with idx_path.open() as f:
                new_idx = json.load(f)
            self.assertEqual(len(new_idx["weight_map"]), n_before)

    def test_missing_index_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                overlay.remove_quantized_lightning_tensors_from_index(Path(tmp), [1, 2])


class TestRemoveSelectedModulesFromIndex(unittest.TestCase):
    def test_removes_attention_quant_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            idx_path = qdir / "model.safetensors.index.json"
            write_safetensors_index(idx_path, layer_count=4, has_attn=False)
            with idx_path.open() as f:
                idx = json.load(f)
            # Simulate full-W4A16 attention quant tensors.
            for layer in range(4):
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    for suf in (".qweight", ".qzeros", ".scales"):
                        idx["weight_map"][
                            f"model.layers.{layer}.self_attn.{proj}{suf}"
                        ] = "model-00001-of-00003.safetensors"
            with idx_path.open("w", encoding="utf-8") as f:
                json.dump(idx, f)

            removed = overlay.remove_quantized_tensors_from_index(
                qdir, [1, 3], ("attn",)
            )

            self.assertEqual(len(removed), 2 * 4 * 3)
            with idx_path.open() as f:
                new_idx = json.load(f)
            for layer in (1, 3):
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    self.assertNotIn(
                        f"model.layers.{layer}.self_attn.{proj}.qweight",
                        new_idx["weight_map"],
                    )
            self.assertIn("model.layers.0.self_attn.q_proj.qweight", new_idx["weight_map"])


class TestRewriteSafetensorsWithoutTensors(unittest.TestCase):
    def test_physically_removes_orphan_quant_tensors(self):
        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError as exc:
            self.skipTest(f"torch/safetensors unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            shard = qdir / "model-00001-of-00001.safetensors"
            removed_name = "model.layers.1.mlp.down_proj.qweight"
            keep_quant_name = "model.layers.0.mlp.down_proj.qweight"
            keep_dense_name = "model.layers.1.input_layernorm.weight"
            save_file(
                {
                    removed_name: torch.ones((2, 2), dtype=torch.int32),
                    "model.layers.1.mlp.down_proj.qzeros": torch.ones(
                        (2, 2), dtype=torch.int32
                    ),
                    keep_quant_name: torch.zeros((2, 2), dtype=torch.int32),
                    keep_dense_name: torch.zeros((2,), dtype=torch.float32),
                },
                str(shard),
                metadata={"format": "pt"},
            )

            removed = {removed_name, "model.layers.1.mlp.down_proj.qzeros"}
            by_shard = overlay.rewrite_safetensors_without_tensors(qdir, removed)

            self.assertEqual(by_shard, {"model-00001-of-00001.safetensors": 2})
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                keys = set(f.keys())
                self.assertNotIn(removed_name, keys)
                self.assertNotIn("model.layers.1.mlp.down_proj.qzeros", keys)
                self.assertIn(keep_quant_name, keys)
                self.assertIn(keep_dense_name, keys)
                self.assertEqual(f.metadata(), {"format": "pt"})

    def test_rewrite_is_idempotent_when_physical_keys_are_already_absent(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError as exc:
            self.skipTest(f"torch/safetensors unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            save_file(
                {"model.layers.0.input_layernorm.weight": torch.zeros((2,))},
                str(qdir / "model-00001-of-00001.safetensors"),
            )

            by_shard = overlay.rewrite_safetensors_without_tensors(
                qdir, {"model.layers.1.mlp.down_proj.qweight"}
            )

            self.assertEqual(by_shard, {})

    def test_rewrite_can_remove_selected_keys_absent_from_index(self):
        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError as exc:
            self.skipTest(f"torch/safetensors unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            shard = qdir / "model-00001-of-00001.safetensors"
            stale_name = "model.layers.2.self_attn.q_proj.qweight"
            keep_name = "model.layers.1.self_attn.q_proj.qweight"
            save_file(
                {
                    stale_name: torch.ones((2, 2), dtype=torch.int32),
                    keep_name: torch.zeros((2, 2), dtype=torch.int32),
                },
                str(shard),
                metadata={"format": "pt"},
            )

            names = overlay.quantized_tensor_names_for_modules([2], ("attn",))
            by_shard = overlay.rewrite_safetensors_without_tensors(qdir, names)

            self.assertEqual(by_shard, {"model-00001-of-00001.safetensors": 1})
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                keys = set(f.keys())
            self.assertNotIn(stale_name, keys)
            self.assertIn(keep_name, keys)


class TestUpdateQuantizeConfigDynamic(unittest.TestCase):
    def test_adds_per_layer_skip_rules_to_quantize_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            qcfg_path = qdir / "quantize_config.json"
            write_quantize_config(qcfg_path)

            overlay.update_quantize_config_dynamic(qdir, [1, 3, 5])

            with qcfg_path.open() as f:
                cfg = json.load(f)

            # 3 layers × 2 SGLang-side fused Linears (gate_up_proj + down_proj)
            # = 6 new skip rules. Pre-2026-05-26 this was 9 (gate/up/down)
            # but the gate/up rules never matched SGLang's fused Linear name.
            dyn = cfg["dynamic"]
            new_rules_count = sum(
                1 for k in dyn
                if "mlp" in k and "model.layers." in k
            )
            self.assertEqual(new_rules_count, 6)
            # Verify the actual rule shapes
            for layer in (1, 3, 5):
                self.assertIn(f"-:model.layers.{layer}.mlp.gate_up_proj$", dyn)
                self.assertIn(f"-:model.layers.{layer}.mlp.down_proj$", dyn)
            # Original self_attn skip rule preserved
            self.assertIn("-:.*self_attn.*", dyn)

    def test_adds_to_config_json_quantization_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            # Write a config.json with quantization_config block
            cfg_path = qdir / "config.json"
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "model_type": "minicpm_sala",
                    "quantization_config": {
                        "bits": 4,
                        "dynamic": {},
                    }
                }, f)

            overlay.update_quantize_config_dynamic(qdir, [2, 4])

            with cfg_path.open() as f:
                cfg = json.load(f)
            qc = cfg["quantization_config"]
            self.assertIn("dynamic", qc)
            new_rules_count = sum(
                1 for k in qc["dynamic"]
                if "mlp" in k and "model.layers." in k
            )
            self.assertEqual(new_rules_count, 4)  # 2 layers × 2 fused Linears (gate_up_proj + down_proj)

    def test_dynamic_rules_match_sglang_fused_linear_prefix(self):
        """Regression guard for 2026-05-26 bug.

        SGLang's MiniCPM implementation creates a single MergedColumnParallelLinear
        named `gate_up_proj` (minicpm.py:62), NOT separate `gate_proj` + `up_proj`.
        Dynamic rule lookup uses re.match() against the SGLang-side prefix at
        Linear construction time (utils.py:257). A rule like
        `-:model.layers.0.mlp.gate_proj$` will NEVER match `gate_up_proj`,
        so SGLang would fall back to gptq_marlin and crash trying to load
        the .qweight tensors that overlay had already removed.
        """
        import re
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            qcfg_path = qdir / "quantize_config.json"
            write_quantize_config(qcfg_path)
            overlay.update_quantize_config_dynamic(qdir, [0, 1])
            with qcfg_path.open() as f:
                cfg = json.load(f)
            dyn = cfg["dynamic"]
            # Simulate SGLang's Linear prefix and rule matching path
            for layer in (0, 1):
                gate_up_prefix = f"model.layers.{layer}.mlp.gate_up_proj"
                down_prefix = f"model.layers.{layer}.mlp.down_proj"
                # At least one rule must match each SGLang prefix
                matched_gate_up = any(
                    pat.startswith("-:") and re.match(pat[2:], gate_up_prefix)
                    for pat in dyn.keys()
                )
                matched_down = any(
                    pat.startswith("-:") and re.match(pat[2:], down_prefix)
                    for pat in dyn.keys()
                )
                self.assertTrue(
                    matched_gate_up,
                    f"no dynamic skip rule matches SGLang fused prefix {gate_up_prefix!r}; "
                    f"rules were {list(dyn.keys())}",
                )
                self.assertTrue(
                    matched_down,
                    f"no dynamic skip rule matches SGLang prefix {down_prefix!r}; "
                    f"rules were {list(dyn.keys())}",
                )

    def test_attention_dynamic_rules_match_sglang_fused_linear_prefix(self):
        """Selective full-W4A16 recovery must skip SGLang's fused qkv_proj.

        The checkpoint stores BF16 q_proj/k_proj/v_proj separately, but
        SGLang constructs one QKVParallelLinear named qkv_proj. The dynamic
        rule therefore has to match qkv_proj, not the HF names.
        """
        import re
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            qcfg_path = qdir / "quantize_config.json"
            with qcfg_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "bits": 4,
                    "group_size": 128,
                    "quant_method": "gptq",
                    "desc_act": False,
                    "sym": True,
                    "lm_head": False,
                    "dynamic": {},
                }, f)
            overlay.update_quantize_config_dynamic_for_modules(
                qdir, [23, 27], ("attn",)
            )
            with qcfg_path.open() as f:
                cfg = json.load(f)
            dyn = cfg["dynamic"]
            for layer in (23, 27):
                self.assertIn(f"-:model.layers.{layer}.self_attn.qkv_proj$", dyn)
                self.assertIn(f"-:model.layers.{layer}.self_attn.o_proj$", dyn)
                qkv_prefix = f"model.layers.{layer}.self_attn.qkv_proj"
                o_prefix = f"model.layers.{layer}.self_attn.o_proj"
                self.assertTrue(
                    any(pat.startswith("-:") and re.match(pat[2:], qkv_prefix) for pat in dyn),
                    f"no dynamic skip rule matches {qkv_prefix!r}; rules were {list(dyn.keys())}",
                )
                self.assertTrue(
                    any(pat.startswith("-:") and re.match(pat[2:], o_prefix) for pat in dyn),
                    f"no dynamic skip rule matches {o_prefix!r}; rules were {list(dyn.keys())}",
                )

    def test_o_proj_module_group_only_skips_o_proj(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            qcfg_path = qdir / "quantize_config.json"
            write_quantize_config(qcfg_path)

            overlay.update_quantize_config_dynamic_for_modules(
                qdir, [23], ("o_proj",)
            )

            with qcfg_path.open() as f:
                cfg = json.load(f)
            dyn = cfg["dynamic"]
            self.assertIn("-:model.layers.23.self_attn.o_proj$", dyn)
            self.assertNotIn("-:model.layers.23.self_attn.qkv_proj$", dyn)

            stale_names = overlay.quantized_tensor_names_for_modules([23], ("o_proj",))
            self.assertIn("model.layers.23.self_attn.o_proj.qweight", stale_names)
            self.assertNotIn("model.layers.23.self_attn.q_proj.qweight", stale_names)

    def test_sequential_qkv_then_o_proj_rules_accumulate(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            qcfg_path = qdir / "quantize_config.json"
            write_quantize_config(qcfg_path)

            overlay.update_quantize_config_dynamic_for_modules(
                qdir, [21, 22, 23, 24], ("qkv",)
            )
            overlay.update_quantize_config_dynamic_for_modules(
                qdir, [23, 24], ("o_proj",)
            )

            with qcfg_path.open() as f:
                cfg = json.load(f)
            dyn = cfg["dynamic"]
            for layer in (21, 22, 23, 24):
                self.assertIn(f"-:model.layers.{layer}.self_attn.qkv_proj$", dyn)
            for layer in (23, 24):
                self.assertIn(f"-:model.layers.{layer}.self_attn.o_proj$", dyn)
            for layer in (21, 22):
                self.assertNotIn(f"-:model.layers.{layer}.self_attn.o_proj$", dyn)

    def test_no_files_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Neither quantize_config.json nor config.json exists
            # Should not raise
            overlay.update_quantize_config_dynamic(Path(tmp), [1])


class TestAlignmentInvariant(unittest.TestCase):
    """The CRITICAL invariant: after applying overlay, the set of layer indices
    referenced in quantize_config.json's dynamic field MUST equal the set we
    removed from model.safetensors.index.json.

    Misalignment is the v14-style KeyError pitfall.
    """

    def test_indices_match_between_index_and_dynamic(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp)
            write_safetensors_index(qdir / "model.safetensors.index.json", layer_count=8)
            write_quantize_config(qdir / "quantize_config.json")

            lightning_indices = [0, 2, 4]

            removed = overlay.remove_quantized_lightning_tensors_from_index(qdir, lightning_indices)
            overlay.update_quantize_config_dynamic(qdir, lightning_indices)

            # Extract layer indices from removed set
            removed_indices: set[int] = set()
            for name in removed:
                import re
                m = re.search(r"model\.layers\.(\d+)\.mlp\.", name)
                if m:
                    removed_indices.add(int(m.group(1)))

            # Extract layer indices from quantize_config.json's dynamic
            with (qdir / "quantize_config.json").open() as f:
                cfg = json.load(f)
            dyn_indices: set[int] = set()
            for rule in cfg.get("dynamic", {}):
                import re
                m = re.search(r"model\.layers\.(\d+)\.mlp", rule)
                if m:
                    dyn_indices.add(int(m.group(1)))

            self.assertEqual(removed_indices, dyn_indices,
                             "Alignment broken: index removes layers X but dynamic skips layers Y")


class TestSelectiveAttentionEndToEnd(unittest.TestCase):
    def test_cli_applies_selective_attention_overlay(self):
        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError as exc:
            self.skipTest(f"torch/safetensors unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bf16_dir = root / "bf16"
            qdir = root / "quant"
            bf16_dir.mkdir()
            qdir.mkdir()

            write_config_json(
                bf16_dir / "config.json",
                ["minicpm4", "lightning-attn", "lightning-attn", "lightning-attn"],
            )
            q_config = {
                "model_type": "minicpm_sala",
                "quantization_config": {
                    "bits": 4,
                    "group_size": 128,
                    "quant_method": "gptq",
                    "desc_act": False,
                    "sym": True,
                    "dynamic": {},
                },
            }
            with (qdir / "config.json").open("w", encoding="utf-8") as f:
                json.dump(q_config, f)
            write_quantize_config(qdir / "quantize_config.json")

            bf16_tensors = {}
            bf16_weight_map = {}
            for layer in (1, 2, 3):
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    name = f"model.layers.{layer}.self_attn.{proj}.weight"
                    bf16_tensors[name] = torch.full((2, 2), layer, dtype=torch.bfloat16)
                    bf16_weight_map[name] = "model-00001-of-00001.safetensors"
            save_file(
                bf16_tensors,
                str(bf16_dir / "model-00001-of-00001.safetensors"),
                metadata={"format": "pt"},
            )
            with (bf16_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
                json.dump({"metadata": {}, "weight_map": bf16_weight_map}, f)

            quant_tensors = {
                "model.layers.2.self_attn.q_proj.qweight": torch.ones((2, 2), dtype=torch.int32),
                "model.layers.2.self_attn.q_proj.qzeros": torch.ones((2, 2), dtype=torch.int32),
                "model.layers.2.self_attn.q_proj.scales": torch.ones((2, 2), dtype=torch.bfloat16),
                "model.layers.2.self_attn.o_proj.qweight": torch.ones((2, 2), dtype=torch.int32),
                "model.layers.2.self_attn.o_proj.qzeros": torch.ones((2, 2), dtype=torch.int32),
                "model.layers.2.self_attn.o_proj.scales": torch.ones((2, 2), dtype=torch.bfloat16),
                "model.layers.1.self_attn.q_proj.qweight": torch.zeros((2, 2), dtype=torch.int32),
                "model.layers.0.input_layernorm.weight": torch.zeros((2,), dtype=torch.bfloat16),
            }
            save_file(
                quant_tensors,
                str(qdir / "model-00001-of-00001.safetensors"),
                metadata={"format": "pt"},
            )
            q_weight_map = {
                name: "model-00001-of-00001.safetensors"
                for name in quant_tensors
            }
            with (qdir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
                json.dump({"metadata": {}, "weight_map": q_weight_map}, f)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "apply_lightning_skip_overlay.py",
                    "--quantized-dir", str(qdir),
                    "--bf16-dir", str(bf16_dir),
                    "--layers", "2",
                    "--modules", "attn",
                    "--overlay-name", "selective.safetensors",
                ]
                self.assertEqual(overlay.main(), 0)
            finally:
                sys.argv = old_argv

            with (qdir / "model.safetensors.index.json").open() as f:
                idx = json.load(f)
            weight_map = idx["weight_map"]
            self.assertEqual(
                weight_map["model.layers.2.self_attn.q_proj.weight"],
                "selective.safetensors",
            )
            self.assertEqual(
                weight_map["model.layers.2.self_attn.o_proj.weight"],
                "selective.safetensors",
            )
            self.assertNotIn("model.layers.2.self_attn.q_proj.qweight", weight_map)
            self.assertIn("model.layers.1.self_attn.q_proj.qweight", weight_map)

            with safe_open(str(qdir / "model-00001-of-00001.safetensors"), framework="pt", device="cpu") as f:
                keys = set(f.keys())
            self.assertNotIn("model.layers.2.self_attn.q_proj.qweight", keys)
            self.assertNotIn("model.layers.2.self_attn.o_proj.qweight", keys)
            self.assertIn("model.layers.1.self_attn.q_proj.qweight", keys)

            with safe_open(str(qdir / "selective.safetensors"), framework="pt", device="cpu") as f:
                overlay_keys = set(f.keys())
            self.assertIn("model.layers.2.self_attn.q_proj.weight", overlay_keys)
            self.assertIn("model.layers.2.self_attn.o_proj.weight", overlay_keys)

            with (qdir / "quantize_config.json").open() as f:
                qcfg = json.load(f)
            dyn = qcfg["dynamic"]
            self.assertIn("-:model.layers.2.self_attn.qkv_proj$", dyn)
            self.assertIn("-:model.layers.2.self_attn.o_proj$", dyn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
