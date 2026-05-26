#!/usr/bin/env python3
"""Decide what to do after a full-W4A16 local eval.

The full experiment has two accuracy reference points:

  - known full-g128 platform baseline: acc_ori=78.27, final_score=22.85
  - preferred target for the next full submission: acc_ori >= 80

This script reads scripts/eval_results.csv, finds the latest full-W4A16 row,
locates the predictions.jsonl path from the eval log, prints per-task analysis,
and emits the next command:

  - acc_ori >= threshold: pack the current full package for platform upload
  - baseline <= acc_ori < threshold: candidate only; usually keep iterating
  - acc_ori < baseline: run the next single-variable accuracy knob

It is CPU-only and safe to run while another GPU job is active.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_VARIANT = "submission_gptqmodel_full_w4a16"
DEFAULT_QUANT_OUT = "/root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16_platform_acc-quantized"
KNOWN_FULL_PLATFORM_BASELINE_ACC_ORI = 78.27
KNOWN_FULL_PLATFORM_BASELINE_FINAL_SCORE = 22.85


def _float(value: str | None) -> float | None:
    if value is None or value == "" or value == "?":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_latest_result(csv_path: Path, variant: str) -> dict[str, str] | None:
    if not csv_path.is_file():
        return None
    rows = list(csv.DictReader(csv_path.open(newline="")))
    matches = [r for r in rows if r.get("variant") == variant]
    return matches[-1] if matches else None


def find_predictions(eval_log: Path) -> Path | None:
    if not eval_log.is_file():
        return None
    text = eval_log.read_text(errors="replace")
    matches = re.findall(r"Detailed results saved to\s+(\S+)/predictions\.jsonl", text)
    if not matches:
        matches = re.findall(r"Saving results to\s+(\S+)", text)
    if not matches:
        return None
    out_dir = Path(matches[-1])
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    pred = out_dir / "predictions.jsonl"
    return pred if pred.is_file() else None


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_predictions(rows: list[dict]) -> str:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task", "?"))
        score = _float(str(row.get("score", "")))
        if score is not None:
            by_task[task].append(score)
    lines = ["per-task mean score:"]
    for task in sorted(by_task):
        vals = by_task[task]
        lines.append(f"  {task:<8} n={len(vals):>3} mean={statistics.mean(vals) * 100:>6.2f}%")
    return "\n".join(lines)


def run_analysis(predictions: Path, top_failures: int) -> None:
    analyzer = REPO / "tools" / "analyze_predictions.py"
    if analyzer.is_file():
        subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--input",
                str(predictions),
                "--top-failures",
                str(top_failures),
            ],
            check=False,
        )
        return
    print(summarize_predictions(load_predictions(predictions)))


def print_pack_command(output_name: str) -> None:
    print()
    print("Next: pack for platform")
    print("```bash")
    print(
        "bash scripts/full_preflight.sh "
        f"--variant {DEFAULT_VARIANT} --pack --output {output_name}"
    )
    print("```")


def print_candidate_pack_command(output_name: str) -> None:
    print()
    print("Candidate pack command, if you choose to spend a platform slot:")
    print("```bash")
    print(
        "bash scripts/full_preflight.sh "
        f"--variant {DEFAULT_VARIANT} --pack --output {output_name}"
    )
    print("```")


def print_next_experiments() -> None:
    print()
    print("Next single-variable accuracy knobs, in order:")
    print("```bash")
    print(
        "GROUP_SIZE=64 NUM_CALIB=256 CALIB_WINDOW_MODE=multi-adaptive "
        "MAX_CALIB_LEN=8192 MAX_CALIB_WINDOWS=3 "
        "bash scripts/run_full_w4a16_platform_acc_now.sh"
    )
    print(
        "GROUP_SIZE=64 NUM_CALIB=150 CALIB_WINDOW_MODE=multi-adaptive "
        "MAX_CALIB_LEN=16384 MAX_CALIB_WINDOWS=3 "
        "bash scripts/run_full_w4a16_platform_acc_now.sh"
    )
    print(
        "GPTQ_DESC_ACT=True GPTQ_STATIC_GROUPS=True GROUP_SIZE=64 NUM_CALIB=150 "
        "CALIB_WINDOW_MODE=multi-adaptive bash scripts/run_full_w4a16_platform_acc_now.sh"
    )
    print("```")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(REPO / "scripts" / "eval_results.csv"))
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument(
        "--baseline-acc-ori",
        type=float,
        default=KNOWN_FULL_PLATFORM_BASELINE_ACC_ORI,
        help=(
            "known platform acc_ori for the previous full-g128 run; below this "
            "the full variant is not an accuracy improvement"
        ),
    )
    parser.add_argument("--top-failures", type=int, default=2)
    parser.add_argument(
        "--pack-output",
        default="soar_gptqmodel_full_w4a16_platform_acc.tar.gz",
    )
    args = parser.parse_args()

    result = load_latest_result(Path(args.csv), args.variant)
    if result is None:
        print(f"No eval_results.csv row found for variant={args.variant!r}")
        print("Run the local eval first:")
        print()
        print("```bash")
        print("bash scripts/run_full_w4a16_platform_acc_now.sh")
        print("```")
        return 1

    acc = _float(result.get("acc_ori"))
    print("Latest full-W4A16 local eval row:")
    for key in ("timestamp", "variant", "quant_model", "num_samples", "concurrency", "acc_ori", "duration_s"):
        print(f"  {key:<12} {result.get(key, '')}")

    eval_log_value = result.get("eval_log", "")
    eval_log = Path(eval_log_value)
    predictions = find_predictions(eval_log) if eval_log_value else None
    if predictions is not None:
        print(f"\npredictions: {predictions}")
        run_analysis(predictions, args.top_failures)
    else:
        print("\npredictions: not found from eval log; skipping per-task analysis")

    print(
        "\nReference: full-g128 platform baseline "
        f"acc_ori={args.baseline_acc_ori:.2f}, "
        f"final_score={KNOWN_FULL_PLATFORM_BASELINE_FINAL_SCORE:.2f}; "
        f"preferred target={args.threshold:.2f}"
    )

    if acc is not None and acc >= args.threshold:
        print(f"\nDecision: PASS local gate ({acc:.2f} >= {args.threshold:.2f})")
        print_pack_command(args.pack_output)
        return 0

    if acc is not None and acc >= args.baseline_acc_ori:
        print(
            f"\nDecision: CANDIDATE ({acc:.2f} >= full baseline "
            f"{args.baseline_acc_ori:.2f}, but < preferred target "
            f"{args.threshold:.2f})"
        )
        print(
            "This may still be platform-viable because the known full-g128 "
            "run scored nonzero at acc_ori=78.27. Prefer the next accuracy "
            "knob unless throughput is clearly better or platform slots are "
            "available."
        )
        print_candidate_pack_command(args.pack_output)
        print_next_experiments()
        return 1

    if acc is None:
        print("\nDecision: no parseable acc_ori; inspect eval/server logs before submitting")
    else:
        print(
            f"\nDecision: BELOW FULL BASELINE ({acc:.2f} < "
            f"{args.baseline_acc_ori:.2f}); do not submit yet"
        )
    print_next_experiments()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
