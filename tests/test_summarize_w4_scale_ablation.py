from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import summarize_w4_scale_ablation as summary  # noqa: E402


class TestSummarizeW4ScaleAblation(unittest.TestCase):
    def test_bottleneck_read_marks_accuracy_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "logs").mkdir()
            (work / "w4_scale_recalib_apply.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "modules_ok": 3,
                            "modules_changed": 3,
                            "modules_skipped": 0,
                            "avg_mse_gain_pct": 4.25,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (work / "bench_summary.csv").write_text(
                "\n".join(
                    [
                        "tag,profile,num_prompts,input_len,output_len,concurrency,output_tok_s,input_tok_s,mean_ttft_ms,mean_tpot_ms,mean_itl_ms,duration_s,jsonl,stdout",
                        "orig,decode,32,1024,512,1,100.0,200.0,50.0,10.0,10.0,12.0,a,b",
                        "recal,decode,32,1024,512,1,101.0,200.0,51.0,10.0,10.0,12.0,a,b",
                        "orig,longctx,16,32768,256,4,80.0,800.0,1000.0,12.0,12.0,20.0,a,b",
                        "recal,longctx,16,32768,256,4,80.5,801.0,990.0,12.0,12.0,20.0,a,b",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (work / "logs" / "eval_orig.log").write_text(
                '{"acc_ori": 79.64}\n', encoding="utf-8"
            )
            (work / "logs" / "eval_recal.log").write_text(
                '{"acc_ori": 80.28}\n', encoding="utf-8"
            )

            text = summary.summarize(work)

        self.assertIn("## Bottleneck Read", text)
        self.assertIn("weight_mse_gain_pct: 4.250%", text)
        self.assertIn("eval_acc_ori_delta: 0.640", text)
        self.assertIn("plausible accuracy bottleneck", text)
        self.assertIn("Runtime read: throughput is flat", text)

    def test_flat_eval_after_mse_gain_points_away_from_global_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "logs").mkdir()
            (work / "w4_scale_recalib_apply.json").write_text(
                json.dumps({"summary": {"avg_mse_gain_pct": 6.0}}),
                encoding="utf-8",
            )
            (work / "logs" / "eval_orig.log").write_text(
                "ori_accuracy = 79.64\n", encoding="utf-8"
            )
            (work / "logs" / "eval_recal.log").write_text(
                "ori_accuracy = 79.70\n", encoding="utf-8"
            )

            text = summary.summarize(work)

        self.assertIn("accuracy stayed flat", text)
        self.assertIn("not global `.scales` MSE", text)


if __name__ == "__main__":
    unittest.main()
