#!/usr/bin/env python3
"""Structural tests for ``python/sglang/srt/models/minicpm_sala_eagle3.py``.

CPU-only: parses the module via AST and verifies the class hierarchy,
EntryClass exposure, and the EAGLE3 invariants that we know from
llama_eagle3.py / cnets.py upstream. We deliberately do NOT instantiate
the model (that needs CUDA + a full SGLang runtime); the file's runtime
correctness is validated end-to-end during Phase 3/4 on AutoDL.

Run:
    python3 -m unittest tests.test_minicpm_sala_eagle3
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "models"
    / "minicpm_sala_eagle3.py"
)


def _parse():
    return ast.parse(SRC_PATH.read_text())


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in {SRC_PATH.name}")


def _method_names(cls: ast.ClassDef) -> set[str]:
    return {
        n.name for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _bases(cls: ast.ClassDef) -> list[str]:
    out = []
    for b in cls.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


class TestMiniCPMSALAEagle3Module(unittest.TestCase):
    """Module-level structural assertions."""

    def setUp(self):
        self.tree = _parse()
        self.text = SRC_PATH.read_text()

    def test_file_exists_and_parses(self):
        self.assertTrue(SRC_PATH.exists(), f"missing: {SRC_PATH}")
        # ast.parse already ran in setUp; if we got here syntax is OK.

    def test_entry_class_exposed(self):
        # The registry harvests `EntryClass` from each module — see
        # python/sglang/srt/models/registry.py:109. If this assignment
        # is missing or malformed, the architecture won't be discovered.
        found = False
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "EntryClass":
                        found = True
                        self.assertIsInstance(node.value, ast.List)
                        names = [
                            e.id
                            for e in node.value.elts
                            if isinstance(e, ast.Name)
                        ]
                        self.assertEqual(names, ["MiniCPMSALAEagle3ForCausalLM"])
        self.assertTrue(found, "module is missing `EntryClass = [...]`")

    def test_required_classes_defined(self):
        for name in (
            "MiniCPMSALAEagle3DecoderLayer",
            "MiniCPMSALAEagle3Model",
            "MiniCPMSALAEagle3ForCausalLM",
        ):
            _find_class(self.tree, name)

    def test_causal_lm_inherits_sala(self):
        # The runtime makes type-level decisions on SALA via
        # isinstance(model, MiniCPMSALAForCausalLM). Keeping the EAGLE3
        # entry class as a subclass means draft-model dispatch lights up
        # the same code paths as the target.
        cls = _find_class(self.tree, "MiniCPMSALAEagle3ForCausalLM")
        self.assertIn(
            "MiniCPMSALAForCausalLM",
            _bases(cls),
            "EAGLE3 entry class must subclass MiniCPMSALAForCausalLM",
        )

    def test_causal_lm_has_eagle3_api(self):
        cls = _find_class(self.tree, "MiniCPMSALAEagle3ForCausalLM")
        methods = _method_names(cls)
        for m in ("__init__", "forward", "load_weights", "get_hot_token_id"):
            self.assertIn(m, methods, f"missing method: {m}")

    def test_decoder_layer_concat_2h(self):
        """qkv_proj must be sized for concat(embeds, hidden) = 2 * hidden_size.

        This is the EAGLE3 signature that distinguishes the draft midlayer
        from a vanilla transformer block — without it, the trained head
        weights wouldn't load.
        """
        cls_src = self.text.split("class MiniCPMSALAEagle3DecoderLayer")[1]
        cls_src = cls_src.split("\nclass ")[0]
        # Look for the QKVParallelLinear override with `2 * self.hidden_size`.
        self.assertRegex(
            cls_src,
            r"self\.self_attn\.qkv_proj\s*=\s*QKVParallelLinear\(\s*\n\s*2\s*\*\s*self\.hidden_size",
            "decoder layer must override qkv_proj to accept 2*hidden_size input",
        )

    def test_model_fc_three_times_input(self):
        """fc projection must be ``hidden_size_in * 3 → hidden_size``.

        EAGLE3 fuses three intermediate hidden-state layers from the
        target. The factor 3 is the load-bearing channel count for the
        trained draft-head weights. See ``llama_eagle3.py:135-139``.
        """
        cls_src = self.text.split("class MiniCPMSALAEagle3Model")[1]
        cls_src = cls_src.split("\nclass ")[0]
        self.assertIn("self.hidden_size_in * 3", cls_src)
        self.assertIn("torch.nn.Linear", cls_src)

    def test_model_returns_aux_list(self):
        """Model.forward must return ``(hidden_to_logits, [hidden_to_aux])``.

        The draft worker / multi-layer eagle worker depends on this
        2-tuple shape (logits-ready tensor + a list of auxiliary hidden
        states). Returning just a tensor or a wrong-shaped list silently
        breaks the verify path.
        """
        cls = _find_class(self.tree, "MiniCPMSALAEagle3Model")
        forwards = [
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        ]
        self.assertEqual(len(forwards), 1)
        # Find at least one `return` that yields a 2-tuple with a list as
        # second element.
        seen_tuple_with_list = False
        for node in ast.walk(forwards[0]):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                if len(node.value.elts) == 2 and isinstance(
                    node.value.elts[1], ast.List
                ):
                    seen_tuple_with_list = True
                    break
        self.assertTrue(
            seen_tuple_with_list,
            "Model.forward must return (hidden, [aux_hidden, ...])",
        )

    def test_uses_dense_attention_not_lightning(self):
        """Draft head must NOT IMPORT Lightning Attention building blocks.

        Design decision (see ``experiments/EAGLE3_SALA_DESIGN.md`` §1.5):
        the draft is dense-only so it stays fast and small. Importing
        ``MiniCPMLightningMixer`` here would be a code smell; the target
        model has its own lightning layers and the Phase 1 SimpleGLA
        tree-verify kernel handles correctness there.

        We check actual imports (AST) rather than raw text so module
        docstrings can still describe the design choice in prose.
        """
        forbidden = {"MiniCPMLightningMixer", "SimpleGLAAttnBackend"}
        imported = set()
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
        leaked = forbidden & imported
        self.assertFalse(
            leaked,
            f"draft head must not import {forbidden}, but imports {leaked}",
        )
        # Also verify no class reference (e.g. `MiniCPMLightningMixer(...)`).
        names_used = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names_used.add(node.id)
            if isinstance(node, ast.Attribute):
                names_used.add(node.attr)
        leaked_refs = forbidden & names_used
        self.assertFalse(
            leaked_refs,
            f"draft head must not reference {forbidden}, but references {leaked_refs}",
        )

    def test_enforces_single_midlayer(self):
        """num_hidden_layers must be checked == 1 (EAGLE3 single-layer)."""
        self.assertRegex(
            self.text,
            r"self\.config\.num_hidden_layers\s*!=\s*1",
            "EAGLE3 entry class must enforce single-layer config",
        )

    def test_handles_hot_token_id_remap(self):
        """``d2t`` / ``t2d`` parameter names trigger draft-vocab remapping.

        Reduced ``draft_vocab_size`` is a key knob for the 2GB cap; the
        loader must recognize the ``d2t``/``t2d`` tensors that ship with
        a hot-token-reduced draft head.
        """
        self.assertIn('"d2t" in name', self.text)
        self.assertIn('"t2d" in name', self.text)


if __name__ == "__main__":
    unittest.main()
