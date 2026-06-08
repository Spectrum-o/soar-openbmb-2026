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
QKNORM_PROBE = ROOT / "tools/probe_minicpm_qknorm_fusion.py"
ROPE_PROBE = ROOT / "tools/probe_minicpm_rope_fusion.py"
GUARDED_ROPE_FUSION_DIR = ROOT / (
    "submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion"
    "_calib300_multi120_ropefusion_guarded"
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

    def test_qknorm_probe_defaults_to_strict_equivalence(self):
        text = QKNORM_PROBE.read_text()
        self.assertIn('--max-abs", type=float, default=0.0', text)
        self.assertIn('"--model-config"', text)
        self.assertIn("load_minicpm_lightning_shape", text)
        self.assertIn("not row.bitwise", text)
        self.assertIn("current MiniCPM two-RMSNorm path", text)

    def test_minicpm_mlp_gate_up_silu_fusion_is_already_present(self):
        text = (ROOT / "python/sglang/srt/models/minicpm.py").read_text()
        self.assertIn("self.gate_up_proj = MergedColumnParallelLinear", text)
        self.assertIn("self.act_fn = SiluAndMul()", text)
        self.assertIn("gate_up, _ = self.gate_up_proj(x)", text)
        self.assertIn("x = self.act_fn(gate_up)", text)

    def test_rope_probe_requires_actual_minicpm_config_and_bitwise_match(self):
        text = ROPE_PROBE.read_text()
        self.assertIn('"--model-config"', text)
        self.assertIn("required=True", text)
        self.assertIn("check_model_identity", text)
        self.assertIn("mixer_types length", text)
        self.assertIn("FAST_CUDA_ROPE_HEAD_DIMS", text)
        self.assertIn('--max-abs", type=float, default=0.0', text)
        self.assertIn("torch.equal(q_ref, q_fused)", text)
        self.assertIn("torch.equal(k_ref, k_fused)", text)

    def test_guarded_rope_fusion_package_is_probe_gated(self):
        prepare_model = (GUARDED_ROPE_FUSION_DIR / "prepare_model.sh").read_text()
        overlay = (GUARDED_ROPE_FUSION_DIR / "overlay/minicpm.py").read_text()
        readme = (GUARDED_ROPE_FUSION_DIR / "README_SUBMISSION.md").read_text()

        self.assertIn("probe_minicpm_rope_fusion.py", prepare_model)
        self.assertIn('--model-config "${INPUT_DIR}/config.json"', prepare_model)
        self.assertIn("--dtype bfloat16", prepare_model)
        self.assertNotIn("--allow-no-cuda", prepare_model)
        self.assertIn("ROPE_FUSION_PROBE=0 is not allowed", prepare_model)
        self.assertEqual(overlay.count("orig_dtype = q.dtype"), 0)
        self.assertEqual(overlay.count("q, k = q.float(), k.float()"), 0)
        self.assertEqual(overlay.count("q, k = q.to(orig_dtype), k.to(orig_dtype)"), 0)
        self.assertGreaterEqual(overlay.count("q, k = self.rotary_emb(positions, q, k)"), 2)
        self.assertIn("bitwise equality", readme)
        self.assertIn("actual MiniCPM-SALA config", readme)


if __name__ == "__main__":
    unittest.main(verbosity=2)
