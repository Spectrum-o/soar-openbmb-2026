#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SAFE_RMS_OPFUSION_OVERLAYS = [
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
        "/overlay/minicpm.py"
    ),
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
        "_calib300_multi120/overlay/minicpm.py"
    ),
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
        "_calib300_damp01_fp8kv_hp224_midcalib_tail4k_offload/overlay/minicpm.py"
    ),
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_denseqkv_rmsopfusion"
        "_calib300_damp01_fp8kv_hp224_midcalib_tail4k_offload/overlay/minicpm.py"
    ),
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
        "_calib300_damp01_fp8kv_hp224_postq_midcalib_tail4k_offload/overlay/minicpm.py"
    ),
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
        "_calib300_damp01_fp8kv_hp224_postq_perhead_midcalib_tail4k_bridge/overlay/minicpm.py"
    ),
]

FP8KV_OVERLAYS = [
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
        "_calib300_damp01_fp8kv_hp224_midcalib_tail4k_offload/overlay/minicpm.py"
    ),
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_denseqkv_rmsopfusion"
        "_calib300_damp01_fp8kv_hp224_midcalib_tail4k_offload/overlay/minicpm.py"
    ),
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
        "_calib300_damp01_fp8kv_hp224_postq_midcalib_tail4k_offload/overlay/minicpm.py"
    ),
    Path(
        "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
        "_calib300_damp01_fp8kv_hp224_postq_perhead_midcalib_tail4k_bridge/overlay/minicpm.py"
    ),
]

BLOCKED_QKNORM_FUSION_DIR = ROOT / (
    "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
    "_qknormfusion_calib300_multi120"
)


def read_overlay(rel_path: Path) -> str:
    path = ROOT / rel_path
    if not path.is_file():
        raise AssertionError(f"expected overlay does not exist: {rel_path}")
    return path.read_text()


class TestMiniCPMOpfusionSafety(unittest.TestCase):
    def test_rope_fp32_upcast_is_preserved(self):
        for rel_path in SAFE_RMS_OPFUSION_OVERLAYS:
            with self.subTest(overlay=str(rel_path)):
                text = read_overlay(rel_path)
                self.assertEqual(text.count("orig_dtype = q.dtype"), 2)
                self.assertEqual(text.count("q, k = q.float(), k.float()"), 2)
                self.assertEqual(text.count("q, k = q.to(orig_dtype), k.to(orig_dtype)"), 2)

    def test_qknorm_jit_fusion_is_not_enabled_in_safe_overlays(self):
        for rel_path in SAFE_RMS_OPFUSION_OVERLAYS:
            with self.subTest(overlay=str(rel_path)):
                text = read_overlay(rel_path)
                self.assertNotIn("apply_qk_norm", text)
                self.assertIn("q = self.q_norm(q.reshape(-1, self.head_dim))", text)
                self.assertIn("k = self.k_norm(k.reshape(-1, self.head_dim))", text)

    def test_rms_residual_delay_path_is_the_only_safe_fusion(self):
        for rel_path in SAFE_RMS_OPFUSION_OVERLAYS:
            with self.subTest(overlay=str(rel_path)):
                text = read_overlay(rel_path)
                self.assertIn(
                    "scale = self.config.scale_depth / math.sqrt("
                    "self.config.num_hidden_layers)",
                    text,
                )
                self.assertIn(
                    "hidden_states, residual = self.input_layernorm("
                    "hidden_states, residual)",
                    text,
                )
                self.assertIn(
                    "hidden_states, residual = self.post_attention_layernorm("
                    "hidden_states, residual)",
                    text,
                )
                self.assertIn("hidden_states, _ = self.norm(hidden_states, residual)", text)

    def test_fp8kv_overlays_keep_scale_name_remap(self):
        for rel_path in FP8KV_OVERLAYS:
            with self.subTest(overlay=str(rel_path)):
                text = read_overlay(rel_path)
                self.assertIn("maybe_remap_kv_scale_name", text)
                self.assertIn('if "scale" in name and name not in params_dict:', text)
                self.assertIn("name = maybe_remap_kv_scale_name(name, params_dict)", text)

    def test_no_unproven_qknorm_fusion_submission_dir(self):
        self.assertFalse(
            BLOCKED_QKNORM_FUSION_DIR.exists(),
            "QKNorm JIT fusion is only approximate in upstream tests and must not "
            "exist as a submit-ready package until full MiniCPM equivalence is proven.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
