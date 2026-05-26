#!/usr/bin/env python3
"""Offline CPU analysis for zyn_partition_eval.py outputs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_latest_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state") == "ok":
                latest[int(row["index"])] = row
    return [latest[i] for i in sorted(latest)]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def task_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("task", "unknown")].append(row)

    summary = []
    for task, items in sorted(groups.items()):
        scores = [float(r.get("score", 0.0)) for r in items]
        out_lens = [int(r.get("output_tokens", 0)) for r in items]
        in_lens = [int(r.get("input_tokens", 0)) for r in items]
        summary.append(
            {
                "task": task,
                "count": len(items),
                "accuracy": round(sum(scores) / len(items) * 100, 2),
                "avg_input_tokens": round(mean(in_lens), 1) if in_lens else 0,
                "avg_output_tokens": round(mean(out_lens), 1) if out_lens else 0,
                "max_output_tokens": max(out_lens) if out_lens else 0,
            }
        )
    return summary


def write_failures_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    failures = [r for r in rows if float(r.get("score", 0.0)) < 1.0]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "task",
                "score",
                "gold",
                "extracted",
                "input_tokens",
                "output_tokens",
                "prediction_head",
            ],
        )
        writer.writeheader()
        for row in failures:
            pred = str(row.get("prediction", ""))
            writer.writerow(
                {
                    "index": row.get("index"),
                    "task": row.get("task"),
                    "score": row.get("score"),
                    "gold": json.dumps(row.get("gold"), ensure_ascii=False),
                    "extracted": row.get("extracted"),
                    "input_tokens": row.get("input_tokens"),
                    "output_tokens": row.get("output_tokens"),
                    "prediction_head": pred[:500].replace("\n", "\\n"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_path = args.run_dir / "predictions.jsonl"
    rows = read_latest_rows(result_path)
    total_score = sum(float(r.get("score", 0.0)) for r in rows)
    total_tokens = sum(int(r.get("output_tokens", 0)) for r in rows)
    overall = {
        "completed": len(rows),
        "ori_accuracy": round(total_score / len(rows) * 100, 2) if rows else 0,
        "overall_accuracy": min(round(total_score / len(rows) * 125, 2), 100) if rows else 0,
        "total_output_tokens": total_tokens,
        "tasks": task_summary(rows),
    }
    write_json(args.run_dir / "analysis_summary.json", overall)
    write_failures_csv(args.run_dir / "failures.csv", rows)
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
