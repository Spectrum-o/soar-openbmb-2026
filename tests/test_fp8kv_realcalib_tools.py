#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_fp8kv_realcalib_package as audit_pkg  # noqa: E402
import decide_fp8kv_next as decide_fp8kv  # noqa: E402


def write_minimal_fp8kv_package(
    path: Path,
    *,
    include_question: bool = True,
    postq: bool = False,
    perhead: bool = False,
    bool_dynamic: bool = False,
    gptq_calib_defaults: tuple[int, int, str] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "kv_calibrate.py").write_text(
        """
prompt = sample.get("question")
tokenizer.truncation_side = args.truncation_side
ap.add_argument("--truncation-side", choices=("left", "right"), default="left")
if ran_forwards == 0:
    raise RuntimeError("no calibration sample was executed")
unvisited = [lid for lid, s in stats.items() if s["n_samples"] == 0]
tensors[f"model.layers.{lid}.self_attn.attn.k_scale"] = torch.tensor([k_scale])
""",
        encoding="utf-8",
    )
    (path / "prepare_env.sh").write_text(
        'export SGLANG_SERVER_ARGS="--kv-cache-dtype fp8_e4m3 --attention-backend minicpm_flashinfer"\n',
        encoding="utf-8",
    )
    gptq_defaults_text = ""
    if gptq_calib_defaults is not None:
        num_calib, max_calib_len, window_mode = gptq_calib_defaults
        gptq_defaults_text = (
            f'NUM_CALIB="${{NUM_CALIB:-{num_calib}}}"\n'
            f'MAX_CALIB_LEN="${{MAX_CALIB_LEN:-{max_calib_len}}}"\n'
            f'CALIB_WINDOW_MODE="${{CALIB_WINDOW_MODE:-{window_mode}}}"\n'
        )
    (path / "prepare_model.sh").write_text(
        gptq_defaults_text
        +
        'KV_CALIB_NUM_SAMPLES="${KV_CALIB_NUM_SAMPLES:-2}"\n'
        'KV_CALIB_TRUNCATION_SIDE="${KV_CALIB_TRUNCATION_SIDE:-left}"\n'
        'KV_CALIB_SAFE_MAX="${KV_CALIB_SAFE_MAX:-224}"\n'
        'timeout "${QUANT_TIMEOUT_MIN}m" python3 "${SCRIPT_DIR}/quantize_gptqmodel_w4a16.py"\n'
        'echo "[prepare_model] step 2: applying selective BF16 overlay on full-W4A16 artifact"\n'
        + ('KV_CALIB_EMIT_PER_HEAD="${KV_CALIB_EMIT_PER_HEAD:-1}"\n' if perhead else "")
        + (
            'echo "[prepare_model] step 3: post-GPTQ fp8 KV scale calibration"\n'
            '--post-quant-kv-only\n'
            '--post-quant-kv-truncation-side "${KV_CALIB_TRUNCATION_SIDE}"\n'
            + ("--post-quant-kv-emit-per-head-scales\n" if perhead else "")
            if postq
            else 'echo "[prepare_model] step 3: fp8 KV scale calibration"\n'
            '--truncation-side "${KV_CALIB_TRUNCATION_SIDE}"\n'
        ),
        encoding="utf-8",
    )
    if postq:
        (path / "quantize_gptqmodel_w4a16.py").write_text(
            """
question = row.get("question")
tokenizer.truncation_side = args.post_quant_kv_truncation_side
if ran_forwards == 0:
    raise RuntimeError("post-GPTQ KV calibration ran zero successful forwards")
unvisited = [lid for lid, s in stats.items() if s["n_samples"] == 0]
tensors[f"model.layers.{lid}.self_attn.attn.k_scale"] = torch.tensor([k_scale])
report = {"source": "post_gptq_qzeros_fixed"}
def _build_scale_tensor_values():
    return {"output_tensor_granularity": "per_head_when_available", "k_tensor_granularity": "per_head"}
if args.post_quant_kv_emit_per_head_scales:
    pass
def _infer_remote_autoconfig_value(input_dir, output_config):
    return "configuration_minicpm_sala.MiniCPMSALAConfig"
def _temporarily_restore_remote_autoconfig_for_postq_reload(output_dir, input_dir):
    auto_map = {}
    auto_map["AutoConfig"] = _infer_remote_autoconfig_value(input_dir, {})
    return "original config"
def _restore_postq_reload_config(output_dir, original_text):
    pass
def _augment_dynamic_skip_aliases_for_reload(output_dir):
    fused_to_hf = {
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
    }
    print("quantize_config.json quantization_config", fused_to_hf)
_augment_dynamic_skip_aliases_for_reload(output_dir)
original_config_text = _temporarily_restore_remote_autoconfig_for_postq_reload(output_dir, input_dir)
try:
    qmodel = GPTQModel.load(output_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)
finally:
    _restore_postq_reload_config(output_dir, original_config_text)
def _stub_transformers_for_gptqmodel_7():
    PreTrainedConfig
    _class_to_module
    _objects
""" + (
                'dynamic = {"-:.*o_gate$": True}\n'
                'def sglang_dynamic_skip_rules_for_layer():\n'
                '    rules: dict[str, bool] = {}\n'
                '    rules["-:model.layers.23.self_attn.qkv_proj$"] = True\n'
                if bool_dynamic else ""
            ) + """
""",
            encoding="utf-8",
        )
    if perhead:
        kv_dir = path / "sglang/python/sglang/srt/layers/quantization"
        kv_dir.mkdir(parents=True, exist_ok=True)
        (kv_dir / "kv_scale_utils.py").write_text(
            "divide_kv_cache_by_scale_\nmultiply_kv_cache_by_scale_\n",
            encoding="utf-8",
        )
        (kv_dir / "kv_cache.py").write_text(
            "k_scale_per_head\nmake_kv_cache_scale_loader\nfinalize_kv_cache_scales\n",
            encoding="utf-8",
        )
        mem_dir = path / "sglang/python/sglang/srt/mem_cache"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "memory_pool.py").write_text(
            "divide_kv_cache_by_scale_\nOptional[Union[float, torch.Tensor]]\nnum_heads=self.head_num\n",
            encoding="utf-8",
        )
        attn_dir = path / "sglang/python/sglang/srt/layers/attention"
        attn_dir.mkdir(parents=True, exist_ok=True)
        (attn_dir / "minicpm_backend.py").write_text(
            "_scale_query_for_per_head_fp8_kv\nscale_flat_query_for_per_head_k\n"
            "q=q_reshaped\nq=q_reshaped_by_head_group\nscale_grouped_output_for_per_head_v\n"
            "multiply_kv_cache_by_scale_\nk_scale_per_head\nv_scale_per_head\n"
            "num_heads=layer.tp_k_head_num\nnum_heads=layer.tp_v_head_num\n",
            encoding="utf-8",
        )
        (attn_dir / "minicpm_attention_kernels.py").write_text(
            'scale_kwargs["k_scale"] = 1.0\n',
            encoding="utf-8",
        )
    rows = []
    for i, task in enumerate(("cwe", "qa")):
        row = {"index": i, "task": task}
        if include_question:
            row["question"] = f"question {i}"
        rows.append(json.dumps(row))
    (path / "perf_public_set.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


class TestAuditFp8kvRealcalibPackage(unittest.TestCase):
    def test_infer_expected_from_names(self):
        self.assertEqual(
            audit_pkg.infer_expected(Path("soar_fp8kv_REALCALIB_HP224_x.tar.gz")),
            ("fp8_e4m3", "224"),
        )
        self.assertEqual(
            audit_pkg.infer_expected(Path("soar_fp8kv_REALCALIB_MAX448_x.tar.gz")),
            ("fp8_e4m3", "448"),
        )
        self.assertEqual(
            audit_pkg.infer_expected(Path("soar_fp8kv_REALCALIB_E5M2_x.tar.gz")),
            ("fp8_e5m2", "224"),
        )

    def test_audit_accepts_realcalib_hp224_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_REALCALIB_HP224_fixture"
            write_minimal_fp8kv_package(package)
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertTrue(ok, out.getvalue())
        self.assertIn("PASS reads SOAR question field", out.getvalue())
        self.assertIn("PASS first 2 calibration rows have question", out.getvalue())

    def test_audit_rejects_missing_question_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_REALCALIB_HP224_fixture"
            write_minimal_fp8kv_package(package, include_question=False)
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertFalse(ok)
        self.assertIn("FAIL first 2 calibration rows have question", out.getvalue())

    def test_audit_accepts_postq_hp224_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_POSTQ_HP224_fixture"
            write_minimal_fp8kv_package(package, postq=True)
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertTrue(ok, out.getvalue())
        self.assertIn("PASS prepare_model uses post-GPTQ KV calibration", out.getvalue())
        self.assertIn("PASS prepare_model passes post-quant truncation-side left", out.getvalue())
        self.assertIn("PASS postq reload uses remote AutoConfig then restores SGLang config", out.getvalue())
        self.assertIn("PASS postq reload config window wraps GPTQModel.load and tokenizer fallback", out.getvalue())
        self.assertIn("PASS postq reload adds HF-name skip aliases before config snapshot", out.getvalue())
        self.assertIn("PASS postq does not mix MiniCPMHybridConfig with remote HF model class", out.getvalue())

    def test_audit_rejects_postq_bool_dynamic_skip_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_POSTQ_HP224_fixture"
            write_minimal_fp8kv_package(package, postq=True, bool_dynamic=True)
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertFalse(ok)
        self.assertIn(
            "FAIL postq dynamic skip values are GPTQModel-dict compatible",
            out.getvalue(),
        )

    def test_audit_rejects_postq_legacy_hybrid_config_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_POSTQ_HP224_fixture"
            write_minimal_fp8kv_package(package, postq=True)
            quantize = package / "quantize_gptqmodel_w4a16.py"
            quantize.write_text(
                quantize.read_text(encoding="utf-8")
                + '\ndef register_minicpm_sala_hf_config():\n'
                + '    AutoConfig.register("minicpm_sala", MiniCPMHybridConfig)\n'
                + "register_minicpm_sala_hf_config()\n",
                encoding="utf-8",
            )
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertFalse(ok)
        self.assertIn(
            "FAIL postq does not mix MiniCPMHybridConfig with remote HF model class",
            out.getvalue(),
        )

    def test_audit_rejects_postq_restore_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_POSTQ_HP224_fixture"
            write_minimal_fp8kv_package(package, postq=True)
            quantize = package / "quantize_gptqmodel_w4a16.py"
            text = quantize.read_text(encoding="utf-8").replace(
                "try:\n    qmodel = GPTQModel.load(output_dir, trust_remote_code=True)\n    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)\nfinally:\n    _restore_postq_reload_config(output_dir, original_config_text)",
                "_restore_postq_reload_config(output_dir, original_config_text)\ntry:\n    qmodel = GPTQModel.load(output_dir, trust_remote_code=True)\n    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)\nfinally:\n    pass",
            )
            quantize.write_text(text, encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertFalse(ok)
        self.assertIn(
            "FAIL postq reload config window wraps GPTQModel.load and tokenizer fallback",
            out.getvalue(),
        )

    def test_audit_rejects_postq_missing_hf_name_skip_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_POSTQ_HP224_fixture"
            write_minimal_fp8kv_package(package, postq=True)
            quantize = package / "quantize_gptqmodel_w4a16.py"
            text = quantize.read_text(encoding="utf-8").replace(
                'def _augment_dynamic_skip_aliases_for_reload(output_dir):\n'
                '    fused_to_hf = {\n'
                '        "qkv_proj": ("q_proj", "k_proj", "v_proj"),\n'
                '        "gate_up_proj": ("gate_proj", "up_proj"),\n'
                '    }\n'
                '    print("quantize_config.json quantization_config", fused_to_hf)\n'
                '_augment_dynamic_skip_aliases_for_reload(output_dir)\n',
                "",
            )
            quantize.write_text(text, encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertFalse(ok)
        self.assertIn(
            "FAIL postq reload adds HF-name skip aliases before config snapshot",
            out.getvalue(),
        )

    def test_audit_rejects_tarball_with_critical_symlink_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "soar_fp8kv_POSTQ_HP224_fixture"
            write_minimal_fp8kv_package(package, postq=True)
            (package / "sglang_target").mkdir()
            (package / "sglang").symlink_to("sglang_target")
            tar_path = root / "soar_fp8kv_POSTQ_HP224_fixture.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tf:
                for item in package.iterdir():
                    tf.add(item, arcname=item.name, recursive=True)

            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(tar_path)

        self.assertFalse(ok)
        self.assertIn("FAIL tarball has no critical symlink members", out.getvalue())
        self.assertIn("sglang", out.getvalue())

    def test_audit_accepts_postq_perhead_hp224_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_POSTQ_PERHEAD_HP224_fixture"
            write_minimal_fp8kv_package(package, postq=True, perhead=True)
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertTrue(ok, out.getvalue())
        self.assertIn("PASS per-head scale emission enabled", out.getvalue())
        self.assertIn("PASS calibration can write per-head scale tensors", out.getvalue())
        self.assertIn("PASS MiniCPM per-head Q/output bridge packaged", out.getvalue())

    def test_audit_rejects_multi8k300_name_without_matching_gptq_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_REALCALIB_MULTI8K300_HP224_fixture"
            write_minimal_fp8kv_package(package)
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertFalse(ok)
        self.assertIn("FAIL GPTQ calibration is 300 x 8K multi-adaptive", out.getvalue())

    def test_audit_accepts_multi8k300_name_with_matching_gptq_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_REALCALIB_MULTI8K300_HP224_fixture"
            write_minimal_fp8kv_package(
                package,
                gptq_calib_defaults=(300, 8192, "multi-adaptive"),
            )
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertTrue(ok, out.getvalue())
        self.assertIn("PASS GPTQ calibration is 300 x 8K multi-adaptive", out.getvalue())

    def test_audit_rejects_kv_calibration_before_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "soar_fp8kv_REALCALIB_HP224_fixture"
            write_minimal_fp8kv_package(package)
            prepare_model = package / "prepare_model.sh"
            text = prepare_model.read_text(encoding="utf-8")
            text = text.replace(
                'echo "[prepare_model] step 2: applying selective BF16 overlay on full-W4A16 artifact"\n'
                'echo "[prepare_model] step 3: fp8 KV scale calibration"\n',
                'echo "[prepare_model] step 3: fp8 KV scale calibration"\n'
                'echo "[prepare_model] step 2: applying selective BF16 overlay on full-W4A16 artifact"\n',
            )
            prepare_model.write_text(text, encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                ok = audit_pkg.audit(package)
        self.assertFalse(ok)
        self.assertIn("FAIL KV scale calibration runs after GPTQ quantize and overlay", out.getvalue())


class TestDecideFp8kvNext(unittest.TestCase):
    def test_extract_score_json_from_platform_text(self):
        score = decide_fp8kv.extract_json_object(
            """
            Score
            {
              "acc_ori": 77.24,
              "final_score": 0.0,
              "benchmark_duration": {"S1": 649.48}
            }
            """
        )
        self.assertIsNotNone(score)
        self.assertEqual(score["acc_ori"], 77.24)
        self.assertEqual(score["benchmark_duration"]["S1"], 649.48)

    def test_hp224_low_accuracy_routes_to_denseqkv_by_default(self):
        text = '{"acc_ori": 77.24, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("hp224", text, score)
        self.assertIn("decision: submit LOWMEM DENSEQKV HP224 diagnostic next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["denseqkv_lowmem"].tarball, result)

    def test_hp224_lowmem_low_accuracy_routes_to_midcalib_hp224(self):
        text = "[INFERENCING] SGLang 服务已就绪\nScore\n{\"acc_ori\": 77.24, \"final_score\": 0.0}"
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("hp224_lowmem", text, score)
        self.assertIn("decision: submit MIDCALIB TAIL4K300 HP224 next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["hp224_midcalib"].tarball, result)
        self.assertNotIn(decide_fp8kv.PACKAGES["e5m2_lowmem"].tarball, result)

    def test_hp224_range_symptom_routes_to_e5m2(self):
        text = '{"acc_ori": 79.1, "final_score": 0.0, "note": "e4m3 range-limited saturation"}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("hp224", text, score)
        self.assertIn("decision: submit LOWMEM E5M2 range backup next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["e5m2_lowmem"].tarball, result)

    def test_startup_failure_stops_dtype_ab(self):
        text = "[FAILED] 任务失败: SGLang 服务启动失败\nKeyError: model.layers.0.self_attn.k_scale"
        result = decide_fp8kv.decide("hp224", text, None)
        self.assertIn("decision: STOP FP8KV queue", result)
        self.assertNotIn(".tar.gz", result)

    def test_prepare_oom_routes_to_lowmem_tail4k_hp224(self):
        text = (
            "[FAILED] 任务失败: prepare_model.sh 执行失败\n"
            "torch.OutOfMemoryError: CUDA out of memory\n"
            "[prepare_model] quantize exited 1"
        )
        result = decide_fp8kv.decide("hp224", text, None)
        self.assertIn("oom_classification: mixed_no_gpu_diag", result)
        self.assertIn("decision: submit MIDCALIB TAIL4K300 HP224 next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["hp224_midcalib"].tarball, result)

    def test_prepare_oom_with_dirty_gpu_diag_resubmits_same_package(self):
        text = """
        [prepare_model] GPU DIAG (before-gptq-quantize): nvidia-smi summary
        0, NVIDIA H100, 85000, 84000, 1000, 0
        [prepare_model] GPU DIAG (before-gptq-quantize): processes
        |    0   N/A  N/A     1234      C   python                       83000MiB |
        [prepare_model] GPU DIAG (before-gptq-quantize): torch mem_get_info
        torch.cuda.mem_get_info free_gib=0.97 total_gib=83.05
        torch.OutOfMemoryError: CUDA out of memory
        [prepare_model] quantize exited 1
        """
        result = decide_fp8kv.decide("hp224_midcalib", text, None)
        self.assertIn("oom_classification: platform_dirty_gpu", result)
        self.assertIn("decision: resubmit the same diagnostic package", result)
        self.assertIn(decide_fp8kv.PACKAGES["hp224_midcalib"].tarball, result)

    def test_prepare_oom_with_clean_gpu_diag_shrinks_midcalib(self):
        text = """
        [prepare_model] GPU DIAG (before-gptq-quantize): nvidia-smi summary
        0, NVIDIA H100, 85000, 0, 85000, 0
        [prepare_model] GPU DIAG (before-gptq-quantize): processes
        |  No running processes found                                                 |
        [prepare_model] GPU DIAG (before-gptq-quantize): torch mem_get_info
        torch.cuda.mem_get_info free_gib=83.01 total_gib=83.05
        [prepare_model] GPU DIAG (after-gptq-quantize-failure): nvidia-smi summary
        0, NVIDIA H100, 85000, 84800, 200, 0
        torch.OutOfMemoryError: CUDA out of memory
        [prepare_model] quantize exited 1
        """
        result = decide_fp8kv.decide("hp224_midcalib", text, None)
        self.assertIn("oom_classification: package_peak_memory", result)
        self.assertIn("decision: STOP and build an even smaller calibrated package", result)
        self.assertNotIn("resubmit the same diagnostic package", result)

    def test_prepare_oom_on_lowmem_hp224_stops_precision_variants(self):
        text = (
            "[FAILED] 任务失败: prepare_model.sh 执行失败\n"
            "torch.OutOfMemoryError: CUDA out of memory\n"
            "[prepare_model] quantize exited 1"
        )
        result = decide_fp8kv.decide("hp224_lowmem", text, None)
        self.assertIn("decision: STOP precision queue", result)
        self.assertIn("NUM_CALIB=75", result)
        self.assertNotIn(".tar.gz", result)

    def test_infer_225432_as_legacy_e5m2(self):
        args = type("Args", (), {"last": None})()
        text = "soar_fp8kv_REALCALIB_E5M2_20260530_225432.tar.gz"
        self.assertEqual(decide_fp8kv.infer_last(args, text), "legacy_e5m2")

    def test_infer_095129_as_legacy_perhead(self):
        args = type("Args", (), {"last": None})()
        text = "soar_fp8kv_POSTQ_PERHEAD_HP224_BRIDGE_20260531_095129.tar.gz"
        self.assertEqual(decide_fp8kv.infer_last(args, text), "legacy_perhead_095129")

    def test_legacy_e5m2_low_accuracy_routes_to_hp224_offload(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("legacy_e5m2", text, score)
        self.assertIn("decision: submit OOM-safe TAIL HP224 next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["hp224_midcalib"].tarball, result)

    def test_infer_tail4k300_denseqkv_as_midcalib(self):
        args = type("Args", (), {"last": None})()
        text = "soar_fp8kv_DENSEQKV_TAIL4K300_HP224_DIAG_OFFLOAD_FIXED_20260531_214103.tar.gz"
        self.assertEqual(decide_fp8kv.infer_last(args, text), "denseqkv_midcalib")

    def test_infer_multi8k300_hp224_as_diag(self):
        args = type("Args", (), {"last": None})()
        text = "soar_fp8kv_REALCALIB_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260531_225344.tar.gz"
        self.assertEqual(decide_fp8kv.infer_last(args, text), "hp224_multi8k_diag")

    def test_infer_multi8k300_denseqkv_as_diag(self):
        args = type("Args", (), {"last": None})()
        text = "soar_fp8kv_DENSEQKV_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260601_101730.tar.gz"
        self.assertEqual(decide_fp8kv.infer_last(args, text), "denseqkv_multi8k_diag")

    def test_infer_tail4k300_postq_as_midcalib(self):
        args = type("Args", (), {"last": None})()
        text = "soar_fp8kv_POSTQ_TAIL4K300_HP224_DIAG_OFFLOAD_FIXED_20260531_215412.tar.gz"
        self.assertEqual(decide_fp8kv.infer_last(args, text), "postq_midcalib")

    def test_infer_tail4k300_perhead_as_midcalib(self):
        args = type("Args", (), {"last": None})()
        text = "soar_fp8kv_POSTQ_PERHEAD_TAIL4K300_HP224_BRIDGE_DIAG_OFFLOAD_FIXED_20260531_221735.tar.gz"
        self.assertEqual(decide_fp8kv.infer_last(args, text), "perhead_midcalib")

    def test_legacy_perhead_095129_routes_to_fixed_hp224(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("legacy_perhead_095129", text, score)
        self.assertIn("decision: submit MIDCALIB fixed HP224 next.", result)
        self.assertIn("not a PERHEAD accuracy signal", result)
        self.assertIn(decide_fp8kv.PACKAGES["hp224_midcalib"].tarball, result)

    def test_offload_kwarg_failure_routes_to_fixed_hp224(self):
        text = (
            "[FAILED] 任务失败: prepare_model.sh 执行失败\n"
            "TypeError: MiniCPMSALAForCausalLM.__init__() got an unexpected "
            "keyword argument 'offload_to_disk'"
        )
        result = decide_fp8kv.decide("legacy_perhead_095129", text, None)
        self.assertIn("decision: ignore this old offload package result", result)
        self.assertIn(decide_fp8kv.PACKAGES["hp224_midcalib"].tarball, result)

    def test_passing_accuracy_keeps_current_result(self):
        text = '{"acc_ori": 80.12, "final_score": 22.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("hp224", text, score)
        self.assertIn("decision: KEEP this FP8KV result", result)

    def test_max448_low_accuracy_routes_to_e5m2(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("max448", text, score)
        self.assertIn("decision: submit LOWMEM E5M2 range backup next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["e5m2_lowmem"].tarball, result)

    def test_cli_current_and_list_are_available(self):
        current_out = StringIO()
        with patch.object(sys, "argv", ["decide_fp8kv_next.py", "--current"]):
            with redirect_stdout(current_out):
                self.assertEqual(decide_fp8kv.main(), 0)
        self.assertIn("build a new POSTQ_MULTI8K HP224 package", current_out.getvalue())
        self.assertIn("Do not submit the old 4K POSTQ/PERHEAD", current_out.getvalue())

        list_out = StringIO()
        with patch.object(sys, "argv", ["decide_fp8kv_next.py", "--list"]):
            with redirect_stdout(list_out):
                self.assertEqual(decide_fp8kv.main(), 0)
        self.assertIn("[hp224_midcalib]", list_out.getvalue())
        self.assertIn("[hp224_multi8k_diag]", list_out.getvalue())
        self.assertIn(decide_fp8kv.PACKAGES["hp224_multi8k_diag"].tarball, list_out.getvalue())
        self.assertIn("[denseqkv_multi8k_diag]", list_out.getvalue())
        self.assertIn(decide_fp8kv.PACKAGES["denseqkv_multi8k_diag"].tarball, list_out.getvalue())
        self.assertIn("[denseqkv_midcalib]", list_out.getvalue())
        self.assertIn(decide_fp8kv.PACKAGES["denseqkv_midcalib"].tarball, list_out.getvalue())
        self.assertIn("[postq_midcalib]", list_out.getvalue())
        self.assertIn(decide_fp8kv.PACKAGES["postq_midcalib"].tarball, list_out.getvalue())
        self.assertIn("[perhead_midcalib]", list_out.getvalue())
        self.assertIn(decide_fp8kv.PACKAGES["perhead_midcalib"].tarball, list_out.getvalue())
        self.assertIn(decide_fp8kv.PACKAGES["denseqkv_lowmem"].tarball, list_out.getvalue())

    def test_hp224_midcalib_low_accuracy_routes_to_denseqkv_midcalib(self):
        text = '{"acc_ori": 77.7, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("hp224_midcalib", text, score)
        self.assertIn("decision: submit MULTI8K300 HP224 diagnostic next", result)
        self.assertIn(decide_fp8kv.PACKAGES["hp224_multi8k_diag"].tarball, result)
        self.assertNotIn(decide_fp8kv.PACKAGES["denseqkv_lowmem"].tarball, result)

    def test_hp224_multi8k_low_accuracy_routes_to_denseqkv_multi8k(self):
        text = '{"acc_ori": 77.7, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("hp224_multi8k_diag", text, score)
        self.assertIn("decision: submit MULTI8K300 DENSEQKV HP224 next", result)
        self.assertIn(decide_fp8kv.PACKAGES["denseqkv_multi8k_diag"].tarball, result)
        self.assertNotIn(decide_fp8kv.PACKAGES["denseqkv_midcalib"].tarball, result)
        self.assertNotIn(decide_fp8kv.PACKAGES["denseqkv_lowmem"].tarball, result)

    def test_hp224_multi8k_prepare_oom_does_not_blame_kv_calibration(self):
        text = (
            "[prepare_model] GPU DIAG (before-gptq-quantize): nvidia-smi summary\n"
            "0, NVIDIA H100, 85000, 0, 85000, 0\n"
            "torch.OutOfMemoryError: CUDA out of memory\n"
            "[prepare_model] quantize exited 1"
        )
        result = decide_fp8kv.decide("hp224_multi8k_diag", text, None)
        self.assertIn("300 x 8K multi-adaptive GPTQ replay", result)
        self.assertIn("not KV-scale calibration", result)

    def test_denseqkv_midcalib_low_accuracy_requests_comparable_postq_build(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("denseqkv_midcalib", text, score)
        self.assertIn("decision: submit MIDCALIB POSTQ HP224 package", result)
        self.assertIn("300 x 4K calibration strength", result)
        self.assertIn(decide_fp8kv.PACKAGES["postq_midcalib"].tarball, result)
        self.assertNotIn(decide_fp8kv.PACKAGES["postq_lowmem"].tarball, result)

    def test_denseqkv_multi8k_low_accuracy_requests_postq_multi8k_build(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("denseqkv_multi8k_diag", text, score)
        self.assertIn("decision: STOP and build a POSTQ_MULTI8K HP224 package", result)
        self.assertIn("300 x 8K multi-adaptive GPTQ coverage", result)
        self.assertNotIn(decide_fp8kv.PACKAGES["postq_midcalib"].tarball, result)
        self.assertNotIn(decide_fp8kv.PACKAGES["postq_lowmem"].tarball, result)

    def test_postq_midcalib_low_accuracy_routes_to_comparable_perhead(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("postq_midcalib", text, score)
        self.assertIn("decision: submit MIDCALIB PERHEAD HP224 BRIDGE next", result)
        self.assertIn("per-head runtime scaling at comparable calibration strength", result)
        self.assertIn(decide_fp8kv.PACKAGES["perhead_midcalib"].tarball, result)
        self.assertNotIn(decide_fp8kv.PACKAGES["perhead_lowmem"].tarball, result)

    def test_denseqkv_low_accuracy_routes_to_postq(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("denseqkv_lowmem", text, score)
        self.assertIn("decision: submit LOWMEM POSTQ HP224 diagnostic next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["postq_lowmem"].tarball, result)

    def test_denseqkv_range_symptom_routes_to_e5m2(self):
        text = '{"acc_ori": 78.5, "note": "range-limited outlier saturation"}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("denseqkv_lowmem", text, score)
        self.assertIn("decision: submit LOWMEM E5M2 range backup next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["e5m2_lowmem"].tarball, result)

    def test_e5m2_low_accuracy_stops_fp8kv_queue(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("e5m2", text, score)
        self.assertIn("decision: STOP FP8KV queue", result)

    def test_postq_low_accuracy_stops_fp8kv_queue(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("postq_lowmem", text, score)
        self.assertIn("decision: submit LOWMEM PERHEAD HP224 BRIDGE next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["perhead_lowmem"].tarball, result)

    def test_postq_range_symptom_routes_to_e5m2(self):
        text = '{"acc_ori": 78.5, "note": "range-limited outlier saturation"}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("postq_lowmem", text, score)
        self.assertIn("decision: submit LOWMEM E5M2 range backup next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["e5m2_lowmem"].tarball, result)

    def test_perhead_low_accuracy_stops_fp8kv_queue(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("perhead", text, score)
        self.assertIn("decision: STOP FP8KV queue", result)

    def test_perhead_midcalib_low_accuracy_stops_fp8kv_queue(self):
        text = '{"acc_ori": 78.5, "final_score": 0.0}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("perhead_midcalib", text, score)
        self.assertIn("decision: STOP FP8KV queue", result)

    def test_perhead_range_symptom_routes_to_e5m2(self):
        text = '{"acc_ori": 78.5, "note": "range-limited outlier saturation"}'
        score = decide_fp8kv.extract_json_object(text)
        result = decide_fp8kv.decide("perhead_lowmem", text, score)
        self.assertIn("decision: submit LOWMEM E5M2 range backup next.", result)
        self.assertIn(decide_fp8kv.PACKAGES["e5m2_lowmem"].tarball, result)


if __name__ == "__main__":
    unittest.main()
