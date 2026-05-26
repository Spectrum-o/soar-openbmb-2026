#!/usr/bin/env python3
"""Print a compact live report for zyn_partition_eval.py run directories."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_latest_ok(path: Path) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state", "ok") == "ok" and "index" in row:
                latest[int(row["index"])] = row
    return [latest[i] for i in sorted(latest)]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(r.get("score", 0.0)) for r in rows]
    outs = [int(r.get("output_tokens", 0)) for r in rows]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task", "unknown"))].append(row)

    task_lines = []
    for task, items in sorted(by_task.items()):
        task_scores = [float(r.get("score", 0.0)) for r in items]
        task_outs = [int(r.get("output_tokens", 0)) for r in items]
        wrong = [
            int(r["index"])
            for r in items
            if float(r.get("score", 0.0)) < 1.0
        ]
        task_lines.append(
            {
                "task": task,
                "n": len(items),
                "acc": round(sum(task_scores) / len(items) * 100.0, 2),
                "avg_out": round(mean(task_outs), 1) if task_outs else 0,
                "max_out": max(task_outs) if task_outs else 0,
                "wrong": wrong,
            }
        )

    top_tails = sorted(
        (
            {
                "index": int(r["index"]),
                "task": r.get("task", "unknown"),
                "score": float(r.get("score", 0.0)),
                "output_tokens": int(r.get("output_tokens", 0)),
            }
            for r in rows
        ),
        key=lambda x: x["output_tokens"],
        reverse=True,
    )[:8]

    return {
        "n": len(rows),
        "acc": round(sum(scores) / len(rows) * 100.0, 2) if rows else 0,
        "total_out": sum(outs),
        "avg_out": round(mean(outs), 1) if outs else 0,
        "max_out": max(outs) if outs else 0,
        "tail10k": sum(1 for x in outs if x >= 10000),
        "tail60k": sum(1 for x in outs if x >= 60000),
        "tasks": task_lines,
        "top_tails": top_tails,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_latest_ok(args.run_dir / "predictions.jsonl")
    run_config = read_json(args.run_dir / "run_config.json")
    progress = read_json(args.run_dir / "progress.json")
    report = summarize(rows)
    target_total = progress.get("target_total")
    if target_total is None:
        start = int(run_config.get("start_index", 0))
        end = int(run_config.get("end_index", start + report["n"]))
        target_total = max(0, end - start)
    report["target_total"] = target_total
    report["errors"] = count_lines(args.run_dir / "errors.jsonl")
    report["run_dir"] = str(args.run_dir)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(
        f"{args.run_dir}: completed={report['n']}/{target_total} "
        f"acc={report['acc']} out={report['total_out']} "
        f"avg_out={report['avg_out']} max_out={report['max_out']} "
        f"tail10k={report['tail10k']} tail60k={report['tail60k']} "
        f"errors={report['errors']}"
    )
    for task in report["tasks"]:
        print(
            f"  {task['task']}: n={task['n']} acc={task['acc']} "
            f"avg_out={task['avg_out']} max_out={task['max_out']} "
            f"wrong={task['wrong']}"
        )
    if report["top_tails"]:
        print("  top_tails:")
        for item in report["top_tails"]:
            print(
                f"    index={item['index']} task={item['task']} "
                f"score={item['score']} out={item['output_tokens']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
