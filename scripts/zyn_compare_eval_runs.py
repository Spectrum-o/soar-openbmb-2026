#!/usr/bin/env python3
"""Compare zyn partition-eval prediction files by task and wrong index."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_rows(path: Path, start: int | None, end: int | None) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state", "ok") != "ok" or "index" not in row:
                continue
            index = int(row["index"])
            if start is not None and index < start:
                continue
            if end is not None and index >= end:
                continue
            latest[index] = row
    return [latest[i] for i in sorted(latest)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_score = sum(float(r.get("score", 0.0)) for r in rows)
    output_tokens = [int(r.get("output_tokens", 0)) for r in rows]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row.get("task", "unknown")].append(row)

    tasks = {}
    for task, items in sorted(by_task.items()):
        scores = [float(r.get("score", 0.0)) for r in items]
        task_out = [int(r.get("output_tokens", 0)) for r in items]
        tasks[task] = {
            "count": len(items),
            "accuracy": round(sum(scores) / len(items) * 100.0, 2),
            "wrong_indices": [
                int(r["index"]) for r in items if float(r.get("score", 0.0)) < 1.0
            ],
            "avg_output_tokens": round(mean(task_out), 1) if task_out else 0,
            "max_output_tokens": max(task_out) if task_out else 0,
        }

    return {
        "count": len(rows),
        "accuracy": round(total_score / len(rows) * 100.0, 2) if rows else 0,
        "total_output_tokens": sum(output_tokens),
        "avg_output_tokens": round(mean(output_tokens), 1) if output_tokens else 0,
        "max_output_tokens": max(output_tokens) if output_tokens else 0,
        "tail_ge_10k": sum(1 for x in output_tokens if x >= 10000),
        "tail_ge_60k": sum(1 for x in output_tokens if x >= 60000),
        "tasks": tasks,
    }


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=/path/to/predictions.jsonl")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        help="Run to compare, formatted as NAME=/path/to/predictions.jsonl.",
    )
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = {}
    for name, path in args.run:
        rows = load_rows(path, args.start_index, args.end_index)
        summaries[name] = summarize(rows)

    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0

    for name, summary in summaries.items():
        print(
            f"{name}: n={summary['count']} acc={summary['accuracy']} "
            f"out={summary['total_output_tokens']} "
            f"avg_out={summary['avg_output_tokens']} "
            f"max_out={summary['max_output_tokens']} "
            f"tail10k={summary['tail_ge_10k']} "
            f"tail60k={summary['tail_ge_60k']}"
        )
        for task, task_summary in summary["tasks"].items():
            print(
                f"  {task}: n={task_summary['count']} "
                f"acc={task_summary['accuracy']} "
                f"wrong={task_summary['wrong_indices']} "
                f"avg_out={task_summary['avg_output_tokens']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
