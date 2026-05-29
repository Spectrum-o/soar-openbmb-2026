#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import full_selective_decide_next as decide  # noqa: E402


class TestFullSelectiveDecideNext(unittest.TestCase):
    def test_parse_platform_score_json(self):
        metrics = decide.parse_score_text(
            """
            Score
            {
              "acc": 100.0,
              "acc_ori": 80.51,
              "final_score": 20.73,
              "benchmark_duration": {
                "S1": 708.97,
                "S8": 1071.87,
                "Smax": 2376.48
              }
            }
            """
        )
        self.assertEqual(metrics["acc"], 100.0)
        self.assertEqual(metrics["acc_ori"], 80.51)
        self.assertEqual(metrics["final_score"], 20.73)
        self.assertEqual(metrics["S1"], 708.97)
        self.assertEqual(metrics["S8"], 1071.87)
        self.assertEqual(metrics["Smax"], 2376.48)

    def test_parse_platform_status_uses_latest_status_line(self):
        status = decide.parse_platform_status_text(
            """
            [2026-05-28 14:47:29] [PENDING] 任务已接收，等待处理
            [2026-05-28 15:14:44] [INFERENCING] SGLang 服务已就绪，正在执行推理评测
            """
        )
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "INFERENCING")
        self.assertIn("SGLang", status.message)

    def test_read_score_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score_path = Path(tmpdir) / "score.log"
            score_path.write_text('Score\n{"acc_ori": 80.12}\n', encoding="utf-8")
            text = decide.read_score_input(None, str(score_path))
        self.assertIn('"acc_ori": 80.12', text)
        self.assertEqual(decide.parse_score_text(text)["acc_ori"], 80.12)

    def test_read_score_rejects_json_and_file_together(self):
        with self.assertRaises(ValueError):
            decide.read_score_input('{"acc_ori": 80.0}', "score.log")

    def test_current_submit_prints_calib600_multi120(self):
        out = StringIO()
        with redirect_stdout(out):
            rc = decide.print_current_submit()
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("current next submit", text)
        self.assertIn(
            "soar_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib600_multi120_20260529_1420.tar.gz",
            text,
        )
        self.assertIn("45a28194e3daa94e6804b596d8f8bbbf", text)

    def test_variant_list_matches_primary_candidates(self):
        out = StringIO()
        with redirect_stdout(out):
            rc = decide.print_variant_list()
        self.assertEqual(rc, 0)
        variants = out.getvalue().splitlines()
        self.assertEqual(
            variants,
            [cand.variant for cand in decide.sorted_primary_candidates()],
        )
        self.assertEqual(len(variants), 23)
        self.assertNotIn(
            decide.PROVEN_FALLBACKS["last8attn"].variant,
            variants,
        )

    def test_pre_submit_script_uses_helper_variant_list(self):
        script = (decide.REPO_ROOT / "scripts/full_selective_pre_submit.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("python3 tools/full_selective_decide_next.py --variants", script)
        self.assertNotIn("submission_gptqmodel_full_w4a16_selective_bf16_last7attn", script)

    def test_queue_map_prints_key_decision_edges(self):
        out = StringIO()
        with redirect_stdout(out):
            rc = decide.print_queue_map()
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("current next submit: last7attn_rmsopfusion_calib600_multi120", text)
        self.assertIn(
            "- last7attn_rmsopfusion: pass -> last7attn_rmsopfusion_calib600_multi120; fail -> stop -> last7attn fallback",
            text,
        )
        self.assertIn(
            "- last7attn_rmsopfusion_calib600_multi120: pass -> last6attn; fail -> stop -> last7attn_rmsopfusion fallback",
            text,
        )
        self.assertIn("- last7attn: pass -> last6attn; fail -> last7attn_g64", text)
        self.assertIn("- last7attn_g64: pass -> stop; fail -> last8qkv_last7oproj", text)
        self.assertIn("- last6attn: pass -> last5attn; fail -> last6attn_g64", text)
        self.assertIn("- last5attn: pass -> last4attn; fail -> last5attn_g64", text)
        self.assertIn("- last4attn: pass -> last4qkv; fail -> last4attn_g64", text)
        self.assertIn("- last8qkv_last7oproj: pass -> last8qkv_last6oproj; fail -> stop -> last8attn fallback", text)
        self.assertIn("- last8qkv_last6oproj: pass -> last8qkv_last4oproj; fail -> stop -> last8qkv_last7oproj fallback", text)

    def test_queue_edges_point_to_known_candidates(self):
        missing = set()
        for key in set(decide.PASS_NEXT) | set(decide.FAIL_NEXT):
            if decide.candidate_by_key(key) is None:
                missing.add(key)
        for next_key in list(decide.PASS_NEXT.values()) + [
            key for key in decide.FAIL_NEXT.values() if key is not None
        ]:
            if decide.candidate_by_key(next_key) is None:
                missing.add(next_key)
        self.assertEqual(missing, set())

    def test_all_candidates_have_expected_md5_pins(self):
        keys = {cand.key for cand in decide.CANDIDATES}
        keys.update(decide.PROVEN_FALLBACKS)
        self.assertEqual(keys - set(decide.EXPECTED_MD5_BY_KEY), set())

    def test_rms_opfusion_tarball_is_pinned(self):
        cand = decide.candidate_by_key("last7attn_rmsopfusion")
        self.assertIsNotNone(cand)
        path = decide.REPO_ROOT / cand.tarball
        self.assertEqual(
            decide.md5_file(path),
            decide.EXPECTED_MD5_BY_KEY["last7attn_rmsopfusion"],
        )

    def test_current_submit_md5_is_pinned(self):
        cand = decide.candidate_by_key("last7attn_rmsopfusion_calib600_multi120")
        self.assertIsNotNone(cand)
        path = decide.REPO_ROOT / cand.tarball
        self.assertEqual(
            decide.md5_file(path),
            decide.EXPECTED_MD5_BY_KEY["last7attn_rmsopfusion_calib600_multi120"],
        )

    def test_last8attn_recommends_last7attn(self):
        decision = decide.decide_next(
            "last8attn", {"acc_ori": 80.51, "S1": 708.97}
        )
        self.assertEqual(decision.action, "submit")
        self.assertEqual(decision.candidate.key, "last7attn")
        self.assertIn("proven", decision.reason)

    def test_last7attn_gate_pass_continues_complete_attn_axis(self):
        decision = decide.decide_next(
            "last7attn", {"acc_ori": 80.2, "S1": 680.0}
        )
        self.assertEqual(decision.action, "submit")
        self.assertEqual(decision.candidate.key, "last6attn")
        self.assertIn("next faster", decision.reason)

    def test_last7attn_gate_fail_switches_to_middle_path(self):
        decision = decide.decide_next(
            "last7attn", {"acc_ori": 79.8, "S1": 650.0}
        )
        self.assertEqual(decision.action, "submit")
        self.assertEqual(decision.candidate.key, "last7attn_g64")
        self.assertIn("failed the 80 gate", decision.reason)

    def test_last7attn_high_final_score_continues_speed_axis(self):
        decision = decide.decide_next(
            "last7attn",
            {
                "acc": 98.39,
                "acc_ori": 78.71,
                "final_score": 20.86,
                "S1": 650.93,
                "S8": 1024.02,
                "Smax": 2330.54,
            },
        )
        self.assertEqual(decision.action, "submit")
        self.assertEqual(decision.candidate.key, "last6attn")
        self.assertIn("final_score=20.86", decision.reason)

    def test_rms_opfusion_success_recommends_heavier_calibration(self):
        decision = decide.decide_next(
            "last7attn_rmsopfusion",
            {
                "acc": 99.83,
                "acc_ori": 79.87,
                "final_score": 21.73,
                "S1": 650.28,
                "S8": 1023.59,
                "Smax": 2334.43,
            },
        )
        self.assertEqual(decision.action, "submit")
        self.assertEqual(decision.candidate.key, "last7attn_rmsopfusion_calib600_multi120")
        self.assertIn("final_score=21.73", decision.reason)

    def test_calib600_multi120_failure_falls_back_to_proven_rms_opfusion(self):
        decision = decide.decide_next(
            "last7attn_rmsopfusion_calib600_multi120",
            {"acc_ori": 77.9, "final_score": 19.0},
        )
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.candidate.key, "last7attn_rmsopfusion")

    def test_last7attn_g64_gate_fail_switches_to_middle_path(self):
        decision = decide.decide_next(
            "last7attn_g64", {"acc_ori": 79.8, "S1": 660.0}
        )
        self.assertEqual(decision.action, "submit")
        self.assertEqual(decision.candidate.key, "last8qkv_last7oproj")

    def test_last6attn_gate_pass_recommends_last5attn(self):
        decision = decide.decide_next("last6attn", {"acc_ori": 80.1})
        self.assertEqual(decision.candidate.key, "last5attn")

    def test_last6attn_gate_fail_recommends_middle_path(self):
        decision = decide.decide_next("last6attn", {"acc_ori": 79.9})
        self.assertEqual(decision.candidate.key, "last6attn_g64")

    def test_last6attn_g64_gate_pass_recommends_last5attn(self):
        decision = decide.decide_next("last6attn_g64", {"acc_ori": 80.1})
        self.assertEqual(decision.candidate.key, "last5attn")

    def test_last6attn_g64_gate_fail_recommends_middle_path(self):
        decision = decide.decide_next("last6attn_g64", {"acc_ori": 79.9})
        self.assertEqual(decision.candidate.key, "last7qkv_last6oproj")

    def test_last5attn_gate_pass_recommends_last4attn(self):
        decision = decide.decide_next("last5attn", {"acc_ori": 80.1})
        self.assertEqual(decision.candidate.key, "last4attn")

    def test_last5attn_gate_fail_recommends_boundary(self):
        decision = decide.decide_next("last5attn", {"acc_ori": 79.9})
        self.assertEqual(decision.candidate.key, "last5attn_g64")

    def test_last5attn_g64_gate_pass_recommends_last4attn(self):
        decision = decide.decide_next("last5attn_g64", {"acc_ori": 80.1})
        self.assertEqual(decision.candidate.key, "last4attn")

    def test_last5attn_g64_gate_fail_recommends_boundary(self):
        decision = decide.decide_next("last5attn_g64", {"acc_ori": 79.9})
        self.assertEqual(decision.candidate.key, "last6qkv_last5oproj")

    def test_last4attn_gate_fail_recommends_boundary(self):
        decision = decide.decide_next("last4attn", {"acc_ori": 79.9})
        self.assertEqual(decision.candidate.key, "last4attn_g64")

    def test_last4attn_g64_gate_pass_recommends_matching_qkv(self):
        decision = decide.decide_next("last4attn_g64", {"acc_ori": 80.1})
        self.assertEqual(decision.candidate.key, "last4qkv")

    def test_last4attn_g64_gate_fail_recommends_boundary(self):
        decision = decide.decide_next("last4attn_g64", {"acc_ori": 79.9})
        self.assertEqual(decision.candidate.key, "last5qkv_last4oproj")

    def test_last4attn_gate_pass_recommends_matching_qkv(self):
        decision = decide.decide_next("last4attn", {"acc_ori": 80.1})
        self.assertEqual(decision.candidate.key, "last4qkv")

    def test_conservative_middle_path_gate_pass_advances_on_same_axis(self):
        decision = decide.decide_next("last8qkv_last6oproj", {"acc_ori": 80.1})
        self.assertEqual(decision.candidate.key, "last8qkv_last4oproj")

    def test_boundary_path_gate_pass_advances_to_next_boundary(self):
        decision = decide.decide_next("last7qkv_last6oproj", {"acc_ori": 80.1})
        self.assertEqual(decision.candidate.key, "last6qkv_last5oproj")

    def test_middle_path_gate_fail_returns_safer_point(self):
        decision = decide.decide_next("last8qkv_last4oproj", {"acc_ori": 79.9})
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.candidate.key, "last8qkv_last6oproj")
        self.assertIn("instead of resubmitting", decision.reason)

    def test_closest_below_last8_gate_fail_stops_at_proven_fallback(self):
        decision = decide.decide_next("last8qkv_last7oproj", {"acc_ori": 79.9})
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.candidate.key, "last8attn")
        self.assertIn("no safer untested", decision.reason)

    def test_qkv_axis_gate_pass_advances_lower_qkv(self):
        decision = decide.decide_next("last6qkv", {"acc_ori": 80.1})
        self.assertEqual(decision.candidate.key, "last5qkv")

    def test_missing_acc_stops_instead_of_guessing(self):
        decision = decide.decide_next("last7attn", {})
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.candidate.key, "last7attn")
        self.assertIn("acc_ori missing", decision.reason)

    def test_failed_platform_status_stops_instead_of_advancing_queue(self):
        status = decide.parse_platform_status_text(
            "[2026-05-28 21:00:21] [FAILED] 任务失败: SGLang 服务启动失败\n"
        )
        decision = decide.decide_next("last7attn", {}, status)
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.candidate.key, "last7attn")
        self.assertIn("failed before a Score JSON", decision.reason)

    def test_inferencing_platform_status_stops_instead_of_advancing_queue(self):
        status = decide.parse_platform_status_text(
            "[2026-05-28 15:14:44] [INFERENCING] SGLang 服务已就绪，正在执行推理评测\n"
        )
        decision = decide.decide_next("last7attn", {}, status)
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.candidate.key, "last7attn")
        self.assertIn("still inferencing", decision.reason)

    def test_success_score_still_advances_from_metrics(self):
        text = """
        [2026-05-28 17:17:43] [SUCCESS] 评测完成
        Score
        {"acc_ori": 80.12, "benchmark_duration": {"S1": 690.0}}
        """
        decision = decide.decide_next(
            "last7attn",
            decide.parse_score_text(text),
            decide.parse_platform_status_text(text),
        )
        self.assertEqual(decision.action, "submit")
        self.assertEqual(decision.candidate.key, "last6attn")

    def test_last7attn_g64_has_fixed_group_size(self):
        cand = decide.candidate_by_key("last7attn_g64")
        self.assertIsNotNone(cand)
        problems = decide.audit_candidate_tarball(cand)
        self.assertEqual(problems, [])

    def test_last6attn_g64_has_fixed_group_size(self):
        cand = decide.candidate_by_key("last6attn_g64")
        self.assertIsNotNone(cand)
        problems = decide.audit_candidate_tarball(cand)
        self.assertEqual(problems, [])

    def test_last5attn_g64_has_fixed_group_size(self):
        cand = decide.candidate_by_key("last5attn_g64")
        self.assertIsNotNone(cand)
        problems = decide.audit_candidate_tarball(cand)
        self.assertEqual(problems, [])

    def test_last4attn_g64_has_fixed_group_size(self):
        cand = decide.candidate_by_key("last4attn_g64")
        self.assertIsNotNone(cand)
        problems = decide.audit_candidate_tarball(cand)
        self.assertEqual(problems, [])

    def test_all_primary_candidate_tarballs_exist(self):
        missing = [
            cand.tarball
            for cand in decide.CANDIDATES
            if not (decide.REPO_ROOT / cand.tarball).is_file()
        ]
        self.assertEqual(missing, [])

    def test_proven_fallback_tarballs_exist_and_pass_audit(self):
        failures = {}
        for key, cand in decide.PROVEN_FALLBACKS.items():
            if not (decide.REPO_ROOT / cand.tarball).is_file():
                failures[key] = [f"missing tarball: {cand.tarball}"]
                continue
            problems = decide.audit_candidate_tarball(cand)
            if problems:
                failures[key] = problems
        self.assertEqual(failures, {})

    def test_candidates_have_expected_default_lines(self):
        missing = [cand.key for cand in decide.CANDIDATES if not cand.expected_lines]
        self.assertEqual(missing, [])

    def test_all_candidate_tarballs_pass_audit(self):
        failures = {
            cand.key: decide.audit_candidate_tarball(cand)
            for cand in decide.CANDIDATES
        }
        failures = {key: problems for key, problems in failures.items() if problems}
        self.assertEqual(failures, {})

    def test_audit_all_tarballs_includes_proven_fallbacks(self):
        out = StringIO()
        with redirect_stdout(out):
            rc = decide.audit_all_tarballs()
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("[PASS] last8attn", text)
        self.assertIn(decide.EXPECTED_MD5_BY_KEY["last8attn"], text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
