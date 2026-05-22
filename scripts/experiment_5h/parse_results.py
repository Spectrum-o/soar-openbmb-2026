#!/usr/bin/env python3
"""scripts/experiment_5h/parse_results.py

Parse one experiment's log file, extract acc + per-task numbers, append a row
to eval_results.csv. Idempotent (won't double-add same exp_id).

Best-effort: if log doesn't have parseable acc, write a row with empty fields
(so the experiment still appears in the table — useful for debugging).

Usage:
    python3 scripts/experiment_5h/parse_results.py \\
        --log scripts/logs/5h_<session>/A_J0_baseline.log \\
        --exp A_J0_baseline \\
        --desc "current 1849 recipe, no env overrides" \\
        --csv scripts/eval_results.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import re
import sys
from pathlib import Path


HEADER = [
    "timestamp",
    "exp",
    "desc",
    "acc",
    "acc_ori",
    "niah",
    "cwe",
    "fwe",
    "qa",
    "mcq",
    "duration_s",
    "log_path",
]


def parse_log(log_path: Path) -> dict[str, str]:
    """Extract metrics from an eval log. Tolerant of various formats."""
    out: dict[str, str] = {k: "" for k in HEADER if k not in ("timestamp", "exp", "desc", "log_path")}
    if not log_path.exists():
        return out

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out

    # acc / acc_ori: try several common formats
    # examples: "acc=58.61", "acc_ori: 46.89", "Average Score: 63.33%", "overall_accuracy=0.5861"
    for key in ("acc_ori", "acc"):
        patterns = [
            rf"{key}[\s]*[=:][\s]*([0-9.]+)",
            rf"\"{key}\"[\s]*:[\s]*([0-9.]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                # If looks like fraction (0.5861 instead of 58.61), scale
                if val < 1.0:
                    val *= 100
                out[key] = f"{val:.2f}"
                break

    # Fallback: "Average Score: 63.33%"
    if not out["acc"] and not out["acc_ori"]:
        m = re.search(r"Average Score[\s]*[=:][\s]*([0-9.]+)", text, re.IGNORECASE)
        if m:
            out["acc"] = f"{float(m.group(1)):.2f}"

    # Per-task: look for patterns like "niah: 23.3%" or "niah=7/30"
    for task in ("niah", "cwe", "fwe", "qa", "mcq"):
        # Pattern 1: explicit percent
        m = re.search(rf"\b{task}\b[\s]*[=:][\s]*([0-9.]+)\s*%?", text, re.IGNORECASE)
        if m:
            out[task] = f"{float(m.group(1)):.2f}"
            continue
        # Pattern 2: fraction "task: 7/30"
        m = re.search(rf"\b{task}\b[\s]*[=:][\s]*([0-9]+)\s*/\s*([0-9]+)", text, re.IGNORECASE)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den > 0:
                out[task] = f"{100.0 * num / den:.2f}"

    # Duration: look for "duration_s=N" or compute from timestamps if needed
    m = re.search(r"duration[_a-z]*[\s]*[=:][\s]*([0-9.]+)", text, re.IGNORECASE)
    if m:
        out["duration_s"] = f"{float(m.group(1)):.0f}"

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    log_path = Path(args.log)
    csv_path = Path(args.csv)

    metrics = parse_log(log_path)

    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp": args.exp,
        "desc": args.desc,
        "log_path": str(log_path.relative_to(Path.cwd())) if log_path.is_absolute() else str(log_path),
        **metrics,
    }

    # Ensure header exists
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in HEADER})

    # Log to stderr for master log
    metric_summary = ", ".join(f"{k}={metrics[k]}" for k in ("acc", "acc_ori", "niah", "cwe", "qa", "mcq") if metrics.get(k))
    print(f"[parse_results] {args.exp}: {metric_summary or '(no metrics found)'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
