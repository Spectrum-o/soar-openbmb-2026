"""Static checks for the full-W4A16 experiment package.

These tests protect the current experiment contract:
  - full-W4A16 defaults to group-size 64 for the acc-recovery pass
  - prepare_env.sh is a real file, not the historical symlink to MLP-only
  - serving stays on the verified BF16/chunk32k runtime stack
  - --dry-run remains lightweight and does not import transformers at module load
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
VARIANT = REPO / "submission_gptqmodel_full_w4a16"
QUANT_SCRIPT = VARIANT / "quantize_gptqmodel_w4a16.py"
PREPARE_MODEL = VARIANT / "prepare_model.sh"
PREPARE_ENV = VARIANT / "prepare_env.sh"
WAIT_SCRIPT = REPO / "scripts" / "run_full_w4a16_platform_acc_when_idle.sh"
NOW_SCRIPT = REPO / "scripts" / "run_full_w4a16_platform_acc_now.sh"
DECIDE_SCRIPT = REPO / "scripts" / "full_w4a16_decide_after_eval.py"
SHARDED_SCRIPT = REPO / "scripts" / "local_eval_sharded.sh"
CHECKPOINT_SCRIPT = REPO / "scripts" / "checkpoint_full_w4a16.sh"


class TestFullW4A16Variant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quant_src = QUANT_SCRIPT.read_text(encoding="utf-8")
        cls.prepare_model_src = PREPARE_MODEL.read_text(encoding="utf-8")
        cls.prepare_env_src = PREPARE_ENV.read_text(encoding="utf-8")
        cls.wait_script_src = WAIT_SCRIPT.read_text(encoding="utf-8")
        cls.now_script_src = NOW_SCRIPT.read_text(encoding="utf-8")
        cls.decide_script_src = DECIDE_SCRIPT.read_text(encoding="utf-8")
        cls.sharded_script_src = SHARDED_SCRIPT.read_text(encoding="utf-8")
        cls.checkpoint_script_src = CHECKPOINT_SCRIPT.read_text(encoding="utf-8")

    def _argparse_default(self, arg_name: str):
        tree = ast.parse(self.quant_src)
        defaults = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = getattr(node.func, "attr", "")
            if func != "add_argument":
                continue
            has_arg = any(
                isinstance(arg, ast.Constant) and arg.value == arg_name
                for arg in node.args
            )
            if not has_arg:
                continue
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults.append(kw.value.value)
        self.assertEqual(len(defaults), 1, f"expected one default for {arg_name}, got {defaults}")
        return defaults[0]

    def test_prepare_env_is_real_file(self):
        self.assertTrue(PREPARE_ENV.exists())
        self.assertFalse(
            PREPARE_ENV.is_symlink(),
            "full-W4A16 prepare_env.sh must not symlink to the MLP-only package",
        )

    def test_default_group_size_is_64(self):
        self.assertEqual(self._argparse_default("--group-size"), 64)
        self.assertIn('GROUP_SIZE="${GROUP_SIZE:-64}"', self.prepare_model_src)
        self.assertIn('--group-size "${GROUP_SIZE}"', self.prepare_model_src)
        self.assertIn("FULL W4A16 group size", self.prepare_model_src)

    def test_platform_acc_profile_defaults(self):
        self.assertEqual(self._argparse_default("--num-calib"), 150)
        self.assertEqual(self._argparse_default("--calib-window-mode"), "multi-adaptive")
        self.assertEqual(self._argparse_default("--max-calib-windows"), 4)
        self.assertIn('FULL_QUANT_PROFILE="${FULL_QUANT_PROFILE:-platform_acc}"', self.prepare_model_src)
        self.assertIn('NUM_CALIB="${NUM_CALIB:-150}"', self.prepare_model_src)
        self.assertIn('CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"', self.prepare_model_src)
        self.assertIn('MAX_CALIB_WINDOWS="${MAX_CALIB_WINDOWS:-4}"', self.prepare_model_src)
        self.assertIn('QUANT_TIMEOUT_MIN="${QUANT_TIMEOUT_MIN:-120}"', self.prepare_model_src)
        self.assertIn("FULL W4A16 calibration:", self.prepare_model_src)

    def test_full_module_set_not_pruned(self):
        for module in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            self.assertIn(module, self.quant_src)
        self.assertNotIn('-:.*self_attn.*', self.quant_src)

    def test_answer_aware_calibration_handles_list_gold(self):
        self.assertIn("def _compact_gold_answer", self.quant_src)
        self.assertIn("isinstance(gold, list)", self.quant_src)
        self.assertIn('"role": "assistant"', self.quant_src)
        self.assertIn("add_generation_prompt=(len(messages) == 1)", self.quant_src)
        self.assertIn("Answer: {answer}", self.quant_src)
        self.assertIn("list-valued", self.quant_src)

    def test_group_size_and_dynamic_are_passed_to_gptqmodel(self):
        self.assertNotIn("inspect.signature(ConfigClass)", self.quant_src)
        self.assertIn('"group_size": group_size', self.quant_src)
        self.assertIn('"dynamic": dynamic', self.quant_src)
        self.assertIn("actual_group_size != group_size", self.quant_src)
        self.assertIn("OPTIONAL_SALA_DYNAMIC_SKIPS", self.quant_src)
        self.assertIn('"-:.*o_gate$": {}', self.quant_src)

    def test_mixed_sensitive_layer_skip_entrypoints(self):
        self.assertIn("--mixed-skip-layers", self.quant_src)
        self.assertIn("--mixed-skip-modules", self.quant_src)
        self.assertIn("def build_dynamic_config", self.quant_src)
        self.assertIn('MIXED_SKIP_LAYERS="${MIXED_SKIP_LAYERS:-}"', self.prepare_model_src)
        self.assertIn('MIXED_SKIP_MODULES="${MIXED_SKIP_MODULES:-all}"', self.prepare_model_src)
        self.assertIn("--mixed-skip-layers", self.prepare_model_src)
        self.assertIn("FULL W4A16 mixed skip layers", self.prepare_model_src)

    def test_gptq_marlin_loader_supports_group64(self):
        marlin_utils = REPO / "python" / "sglang" / "srt" / "layers" / "quantization" / "marlin_utils.py"
        src = marlin_utils.read_text(encoding="utf-8")
        self.assertIn("MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]", src)

    def test_chat_template_renders_gold_as_assistant_turn(self):
        spec = importlib.util.spec_from_file_location("full_quant", QUANT_SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        class FakeTensor:
            def __init__(self):
                self.shape = (1, 4)

            def numel(self):
                return 4

            def to(self, *_args, **_kwargs):
                return self

            def long(self):
                return self

        class FakeTokenizer:
            chat_template = "{{ messages }}"
            truncation_side = "right"

            def __init__(self):
                self.messages = None

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                self.add_generation_prompt = add_generation_prompt
                return "rendered"

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

        tokenizer = FakeTokenizer()
        module.tokenize_calibration(
            tokenizer,
            [{"question": "Q?", "answer": "A", "text": "Q?\nAnswer: A"}],
            max_len=16,
            window_mode="tail",
        )
        self.assertEqual(
            tokenizer.messages,
            [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "A"},
            ],
        )
        self.assertFalse(tokenizer.add_generation_prompt)

    def test_idle_runner_is_non_destructive_and_uses_platform_acc(self):
        self.assertTrue(WAIT_SCRIPT.exists())
        self.assertTrue(WAIT_SCRIPT.stat().st_mode & 0o111)
        self.assertIn("nvidia-smi --query-gpu=memory.used", self.wait_script_src)
        self.assertIn('FULL_QUANT_PROFILE="${FULL_QUANT_PROFILE:-platform_acc}"', self.wait_script_src)
        self.assertIn('CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"', self.wait_script_src)
        self.assertIn('MAX_CALIB_WINDOWS="${MAX_CALIB_WINDOWS:-4}"', self.wait_script_src)
        self.assertIn('SHARDED_EVAL="${SHARDED_EVAL:-1}"', self.wait_script_src)
        self.assertIn('SHARD_SIZE="${SHARD_SIZE:-30}"', self.wait_script_src)
        self.assertIn("scripts/local_eval_sharded.sh", self.wait_script_src)
        self.assertIn("--force-requant", self.wait_script_src)
        tokens = shlex.split(self.wait_script_src, comments=True)
        self.assertNotIn("kill", tokens)
        self.assertNotIn("pkill", tokens)
        self.assertNotIn("killall", tokens)

    def test_manual_runner_does_not_wait_or_kill(self):
        self.assertTrue(NOW_SCRIPT.exists())
        self.assertTrue(NOW_SCRIPT.stat().st_mode & 0o111)
        self.assertIn("refusing to start full-W4A16", self.now_script_src)
        self.assertIn('GROUP_SIZE="${GROUP_SIZE:-64}"', self.now_script_src)
        self.assertIn('CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"', self.now_script_src)
        self.assertIn('MAX_CALIB_WINDOWS="${MAX_CALIB_WINDOWS:-4}"', self.now_script_src)
        self.assertIn('SHARDED_EVAL="${SHARDED_EVAL:-1}"', self.now_script_src)
        self.assertIn('SHARD_SIZE="${SHARD_SIZE:-30}"', self.now_script_src)
        self.assertIn("scripts/local_eval_sharded.sh", self.now_script_src)
        self.assertIn("--force-requant", self.now_script_src)
        self.assertNotRegex(self.now_script_src, r"\bsleep\b")
        tokens = shlex.split(self.now_script_src, comments=True)
        self.assertNotIn("kill", tokens)
        self.assertNotIn("pkill", tokens)
        self.assertNotIn("killall", tokens)

    def test_sharded_eval_writes_incremental_results(self):
        self.assertTrue(SHARDED_SCRIPT.exists())
        self.assertTrue(SHARDED_SCRIPT.stat().st_mode & 0o111)
        self.assertIn('cd "${REPO_ROOT}"', self.sharded_script_src)
        self.assertIn('RESULTS_CSV="${REPO_ROOT}/scripts/eval_shards.csv"', self.sharded_script_src)
        self.assertIn("cumulative_acc", self.sharded_script_src)
        self.assertIn('CHECKPOINT_AFTER_SHARD="${CHECKPOINT_AFTER_SHARD:-1}"', self.sharded_script_src)
        self.assertIn("--no-checkpoint", self.sharded_script_src)
        self.assertIn("checkpointing partial results", self.sharded_script_src)
        self.assertIn('RUN_CHECKS=0 MESSAGE="checkpoint full w4a16 shard ${SHARD_INDEX} cumulative ${CUM_ACC}"', self.sharded_script_src)
        self.assertIn("--max-samples", self.sharded_script_src)
        self.assertIn("--shard-size", self.sharded_script_src)
        self.assertIn("shard_acc=${SHARD_ACC} cumulative_acc=${CUM_ACC}", self.sharded_script_src)
        self.assertIn("sharded_predictions_${VARIANT_NAME}_${STAMP}_${SHARD_INDEX}.jsonl", self.sharded_script_src)
        self.assertIn('cp "${PRED_PATH}" "${ARCHIVED_PRED}"', self.sharded_script_src)
        self.assertIn("shard_{shard_idx:03d}_{start:03d}_{end:03d}.jsonl", self.sharded_script_src)

    def test_checkpoint_script_pushes_scoped_full_checkpoint(self):
        self.assertTrue(CHECKPOINT_SCRIPT.exists())
        self.assertTrue(CHECKPOINT_SCRIPT.stat().st_mode & 0o111)
        self.assertIn('BRANCH="${BRANCH:-exp/full-w4a16}"', self.checkpoint_script_src)
        self.assertIn('"HEAD:refs/heads/${BRANCH}"', self.checkpoint_script_src)
        self.assertIn("git add -A -- submission_gptqmodel_full_w4a16", self.checkpoint_script_src)
        self.assertIn("scripts/eval_shards.csv", self.checkpoint_script_src)
        self.assertIn("scripts/logs/sharded_predictions_*.jsonl", self.checkpoint_script_src)
        self.assertNotIn("git add .", self.checkpoint_script_src)

    def test_decision_helper_is_cpu_only_and_full_scoped(self):
        self.assertTrue(DECIDE_SCRIPT.exists())
        self.assertTrue(DECIDE_SCRIPT.stat().st_mode & 0o111)
        self.assertIn('DEFAULT_VARIANT = "submission_gptqmodel_full_w4a16"', self.decide_script_src)
        self.assertIn("KNOWN_FULL_PLATFORM_BASELINE_ACC_ORI = 78.27", self.decide_script_src)
        self.assertIn("--baseline-acc-ori", self.decide_script_src)
        self.assertIn("--shards-csv", self.decide_script_src)
        self.assertIn("eval_shards.csv", self.decide_script_src)
        self.assertIn("Decision: PARTIAL", self.decide_script_src)
        self.assertIn("acc_ori >= threshold", self.decide_script_src)
        self.assertIn("parse_quant_losses", self.decide_script_src)
        self.assertIn("MIXED_SKIP_LAYERS", self.decide_script_src)
        self.assertIn("--quant-log", self.decide_script_src)
        self.assertIn("full_preflight.sh", self.decide_script_src)
        self.assertIn("GPTQ_DESC_ACT=True GPTQ_STATIC_GROUPS=True", self.decide_script_src)
        self.assertNotIn("nvidia-smi", self.decide_script_src)
        self.assertNotIn("local_eval.sh", self.decide_script_src)
        tokens = shlex.split(self.decide_script_src, comments=True)
        self.assertNotIn("kill", tokens)
        self.assertNotIn("pkill", tokens)
        self.assertNotIn("killall", tokens)

    def test_decision_helper_treats_78_27_to_80_as_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "eval_results.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "variant",
                        "quant_model",
                        "num_samples",
                        "concurrency",
                        "acc_ori",
                        "acc_overall",
                        "duration_s",
                        "server_log",
                        "eval_log",
                    ]
                )
                writer.writerow(
                    [
                        "2026-05-26 00:00:00",
                        "submission_gptqmodel_full_w4a16",
                        "/tmp/model",
                        "150",
                        "32",
                        "79.0",
                        "79.0",
                        "1",
                        "",
                        "",
                    ]
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(DECIDE_SCRIPT),
                    "--csv",
                    str(csv_path),
                    "--shards-csv",
                    str(Path(tmp) / "missing_eval_shards.csv"),
                ],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Decision: CANDIDATE", result.stdout)
        self.assertIn("full-g128 platform baseline acc_ori=78.27", result.stdout)
        self.assertIn("Candidate pack command", result.stdout)
        self.assertNotIn("BELOW FULL BASELINE", result.stdout)

    def test_decision_helper_treats_incomplete_shards_as_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards_csv = Path(tmp) / "eval_shards.csv"
            with shards_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "variant",
                        "quant_model",
                        "shard_index",
                        "shard_start",
                        "shard_end",
                        "shard_samples",
                        "cumulative_samples",
                        "concurrency",
                        "shard_acc",
                        "cumulative_acc",
                        "shard_duration_s",
                        "total_duration_s",
                        "server_log",
                        "eval_log",
                        "predictions_path",
                    ]
                )
                writer.writerow(
                    [
                        "2026-05-26 00:30:00",
                        "submission_gptqmodel_full_w4a16",
                        "/tmp/model",
                        "2",
                        "30",
                        "59",
                        "30",
                        "60",
                        "32",
                        "81.0",
                        "82.0",
                        "10",
                        "20",
                        "",
                        "",
                        "",
                    ]
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(DECIDE_SCRIPT),
                    "--csv",
                    str(Path(tmp) / "missing_eval_results.csv"),
                    "--shards-csv",
                    str(shards_csv),
                ],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Decision: PARTIAL", result.stdout)
        self.assertIn("60/150 samples evaluated", result.stdout)
        self.assertIn("Current cumulative acc 82.00", result.stdout)
        self.assertNotIn("Next: pack for platform", result.stdout)

    def test_decision_helper_prints_sensitive_layer_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards_csv = Path(tmp) / "eval_shards.csv"
            quant_log = Path(tmp) / "quant.log"
            with shards_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "variant",
                        "quant_model",
                        "shard_index",
                        "shard_start",
                        "shard_end",
                        "shard_samples",
                        "cumulative_samples",
                        "concurrency",
                        "shard_acc",
                        "cumulative_acc",
                        "shard_duration_s",
                        "total_duration_s",
                        "server_log",
                        "eval_log",
                        "predictions_path",
                    ]
                )
                writer.writerow(
                    [
                        "2026-05-26 00:30:00",
                        "submission_gptqmodel_full_w4a16",
                        "/tmp/model",
                        "4",
                        "120",
                        "149",
                        "30",
                        "150",
                        "32",
                        "75.0",
                        "77.0",
                        "10",
                        "50",
                        "",
                        "",
                        "",
                    ]
                )
            quant_log.write_text(
                "\n".join(
                    [
                        "| gptq | 0 | self_attn.q_proj | 4096, 4096 | bf16 | 0.0000010000 |",
                        "| gptq | 1 | mlp.down_proj    | 16384, 4096 | bf16 | 0.0000090000 |",
                        "| gptq | 2 | self_attn.o_proj | 4096, 4096 | bf16 | 0.0000040000 |",
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(DECIDE_SCRIPT),
                    "--csv",
                    str(Path(tmp) / "missing_eval_results.csv"),
                    "--shards-csv",
                    str(shards_csv),
                    "--quant-log",
                    str(quant_log),
                    "--top-loss-layers",
                    "2",
                ],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Top GPTQ loss modules", result.stdout)
        self.assertIn("layer= 1 module=mlp.down_proj", result.stdout)
        self.assertIn("Mixed BF16 skip candidate", result.stdout)
        self.assertIn("MIXED_SKIP_LAYERS=1,2", result.stdout)

    def test_decision_helper_quant_only_does_not_require_eval_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            quant_log = Path(tmp) / "quant.log"
            quant_log.write_text(
                "\n".join(
                    [
                        "| gptq | 4 | mlp.down_proj    | 16384, 4096 | bf16 | 0.0000190000 |",
                        "| gptq | 7 | self_attn.o_proj | 4096, 4096 | bf16 | 0.0000140000 |",
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(DECIDE_SCRIPT),
                    "--csv",
                    str(Path(tmp) / "missing_eval_results.csv"),
                    "--shards-csv",
                    str(Path(tmp) / "missing_eval_shards.csv"),
                    "--quant-log",
                    str(quant_log),
                    "--quant-only",
                    "--top-loss-layers",
                    "1",
                ],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Top GPTQ loss modules", result.stdout)
        self.assertIn("Mixed BF16 skip candidate", result.stdout)
        self.assertIn("MIXED_SKIP_LAYERS=4", result.stdout)

    def test_prepare_env_uses_verified_bf16_runtime_stack(self):
        self.assertIn('TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-4.57.1}"', self.prepare_env_src)
        self.assertIn("import smoke test: transformers", self.prepare_env_src)
        self.assertIn('find -L "${SUBMISSION_DIR}"', self.prepare_env_src)
        self.assertIn("SKIPPING the fp16 sed-patch", self.prepare_env_src)
        self.assertNotIn("sed -i 's/torch\\.bfloat16/torch.float16/g'", self.prepare_env_src)
        self.assertRegex(
            self.prepare_env_src,
            r'SGLANG_SERVER_ARGS="[^"]*--chunked-prefill-size 32768[^"]*'
            r'--max-prefill-tokens 32768[^"]*--mem-fraction-static 0\.70[^"]*'
            r'--quantization gptq_marlin[^"]*--dtype bfloat16',
        )

    def test_dry_run_does_not_import_transformers_at_module_load(self):
        # The compatibility shim imports transformers internally, but it must
        # be called only after the dry-run exit branch. This keeps dry-run
        # useful in shells without the full HF dependency set installed.
        top_level_call = re.search(
            r"^_stub_transformers_for_gptqmodel_7\(\)",
            self.quant_src,
            flags=re.MULTILINE,
        )
        self.assertIsNone(top_level_call)

        main_body = re.search(
            r"def main\(\) -> int:\n(?P<body>.*)\n\nif __name__ ==",
            self.quant_src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(main_body)
        body = main_body.group("body")
        self.assertLess(body.index("if args.dry_run:"), body.index("_stub_transformers_for_gptqmodel_7()"))


if __name__ == "__main__":
    unittest.main()
