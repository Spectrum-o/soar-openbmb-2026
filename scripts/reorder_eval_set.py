#!/usr/bin/env python3
"""Reorder perf_public_set.jsonl from task-grouped to round-robin interleaved.

Why: the SOAR public set groups all 30 samples per task contiguously
(mcq 0-29, niah 30-59, qa 60-89, fwe 90-119, cwe 120-149). Partial-eval
inspection — peek.sh, `wc -l predictions.jsonl`, mid-run aggregate
acc — can only see mcq early in the run, masking the per-task failure
modes (niah haystack-echo loops, qa task-difficulty short outputs,
etc.) until late.

Round-robin interleaving makes the first N*K rows (K = number of task
types = 5) cover all task types K times, so partial acc is
representative throughout the run.

The `index` field on each row is preserved (it's the source position in
the original SOAR-Toolkit jsonl), so `tools/diff_predictions.py` and
`tools/analyze_predictions.py` still align across runs that used
different orders.

Calibration is unaffected: GPTQModel's per-layer Hessian is a commutative
sum of outer products, so the order in which calibration prompts are
fed does not change the resulting quantization.

Idempotent: re-running on an already-round-robin file is a no-op.

Usage:
    python3 scripts/reorder_eval_set.py \\
        --input submission_gptqmodel_calib_w4a16/perf_public_set.jsonl
    python3 scripts/reorder_eval_set.py --input X --output Y
    python3 scripts/reorder_eval_set.py --input X --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def reorder_round_robin(rows: list[dict], task_key: str = "task") -> list[dict]:
    """Round-robin interleave rows by task. Stable within each task."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[str(r.get(task_key, "?"))].append(r)
    tasks = sorted(by_task.keys())
    max_per_task = max((len(by_task[t]) for t in tasks), default=0)
    out: list[dict] = []
    for i in range(max_per_task):
        for t in tasks:
            if i < len(by_task[t]):
                out.append(by_task[t][i])
    return out


def fmt_bin_distribution(rows: list[dict], bin_size: int = 25) -> str:
    out_lines = []
    for start in range(0, len(rows), bin_size):
        chunk = rows[start:start + bin_size]
        c = Counter(r.get("task", "?") for r in chunk)
        d = {k: c[k] for k in sorted(c)}
        out_lines.append(f"  rows {start:3d}-{start + len(chunk) - 1:3d}: {d}")
    return "\n".join(out_lines)


def load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with p.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[reorder] line {i} bad json: {exc}", file=sys.stderr)
    return rows


def write_jsonl(rows: list[dict], p: Path) -> None:
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="perf_public_set.jsonl path")
    ap.add_argument("--output", default=None,
                    help="default: overwrite input in place")
    ap.add_argument("--dry-run", action="store_true",
                    help="show planned reorder; do not write")
    ap.add_argument("--task-key", default="task",
                    help="row field name to interleave by (default: 'task')")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        print(f"input not found: {src}", file=sys.stderr)
        return 2
    dst = Path(args.output) if args.output else src

    rows = load_jsonl(src)
    if not rows:
        print(f"[reorder] {src}: empty", file=sys.stderr)
        return 2

    print(f"[reorder] {src}: {len(rows)} rows")
    print(f"[reorder] BEFORE distribution:")
    print(fmt_bin_distribution(rows))

    reordered = reorder_round_robin(rows, args.task_key)

    # Detect no-op: if every row's task at position i is already different
    # from the row at position i-1, we're already round-robin-ish; explicit
    # check via index-equality is the strict signal.
    same = (
        len(rows) == len(reordered)
        and all(rows[i] is reordered[i] for i in range(len(rows)))
    )
    # Looser check: same order if index sequence matches
    same_index = (
        len(rows) == len(reordered)
        and all(
            rows[i].get("index") == reordered[i].get("index")
            for i in range(len(rows))
        )
    )

    print(f"[reorder] AFTER distribution:")
    print(fmt_bin_distribution(reordered))
    print(f"[reorder] first 5 rows' tasks: "
          f"{[r.get(args.task_key) for r in reordered[:5]]}")
    print(f"[reorder] first 10 rows' indexes: "
          f"{[r.get('index') for r in reordered[:10]]}")

    if same or same_index:
        print("[reorder] file is already in target order; no change written")
        return 0

    if args.dry_run:
        print(f"[dry-run] would write {len(reordered)} rows to {dst}")
        return 0

    write_jsonl(reordered, dst)
    print(f"[reorder] wrote {len(reordered)} rows to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
