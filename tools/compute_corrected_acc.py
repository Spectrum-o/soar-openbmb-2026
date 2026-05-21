#!/usr/bin/env python3
"""Compute corrected accuracy on a SOAR predictions.jsonl.

The platform's reported acc is a simple mean over all samples, but the
breakdown by task type reveals that some tasks are dramatically easier
than others. Per the 2026-05-22 v21 analysis:

  - fwe (frequent-word extraction): trivial, 100% pass, ~30 samples
  - mcq (multiple-choice): real signal, quant-sensitive (B<->D bias)
  - cwe (chinese-word extraction): real signal, repetition-collapse-prone
  - niah (needle-in-haystack): real signal, repetition-collapse-prone
  - qa (open-ended QA): partly task-intrinsic difficulty, not quant

This tool computes:
  - Per-task pass rate (binary @ 0.5 threshold) and mean continuous score
  - "Corrected aggregate" = mean over tasks EXCLUDING fwe-style free credit
    so we get a clearer signal of quant quality
  - Per-task contribution to the standard aggregate (helps see where
    movement in aggregate is actually coming from)

This is COMPLEMENTARY to tools/analyze_predictions.py, which does more
exhaustive per-task error-mode classification. Use this when you want a
quick "what's my real accuracy" number, or when comparing two runs.

Usage:
    # Single run
    python3 tools/compute_corrected_acc.py --input outputs/<dir>/predictions.jsonl

    # Compare two runs side by side
    python3 tools/compute_corrected_acc.py \\
        --input outputs/<v23>/predictions.jsonl \\
        --compare outputs/<v21_baseline>/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


# Tasks that are EXCLUDED from the corrected aggregate. Add or remove
# entries here as we learn more about per-task quality signals.
# Source: memory/project_v21_per_task_breakdown.md
FREE_CREDIT_TASKS = ("fwe",)


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[load] line {i}: skip bad json: {exc}", file=sys.stderr)
    return rows


def _score(row: dict) -> float | None:
    if "score" not in row:
        return None
    try:
        return float(row["score"])
    except (TypeError, ValueError):
        return None


def _task(row: dict) -> str:
    return str(row.get("task", "?"))


def per_task_stats(rows: list[dict]) -> dict[str, dict]:
    """Return per-task stats: count, pass_rate, mean_score."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        s = _score(r)
        if s is not None:
            by_task[_task(r)].append(s)
    out = {}
    for task, scores in by_task.items():
        n = len(scores)
        passed = sum(1 for s in scores if s >= 0.5)
        out[task] = {
            "n": n,
            "pass_rate": passed / n if n else 0.0,
            "mean_score": statistics.mean(scores) if scores else 0.0,
        }
    return out


def aggregate(rows: list[dict]) -> dict:
    """Compute (standard, corrected) aggregates and per-task contributions."""
    stats = per_task_stats(rows)
    total_n = sum(d["n"] for d in stats.values())
    total_score = sum(d["mean_score"] * d["n"] for d in stats.values())
    standard_acc = total_score / total_n if total_n else 0.0

    # Corrected = simple mean over non-free-credit tasks' pass_rate
    real_tasks = {t: d for t, d in stats.items() if t not in FREE_CREDIT_TASKS}
    if real_tasks:
        corrected_acc = statistics.mean(d["pass_rate"] for d in real_tasks.values())
    else:
        corrected_acc = 0.0

    # Per-task contribution to the standard aggregate (in percentage points)
    contributions = {}
    for t, d in stats.items():
        if total_n:
            contributions[t] = (d["mean_score"] * d["n"]) / total_n
        else:
            contributions[t] = 0.0

    return {
        "total_n": total_n,
        "standard_acc": standard_acc,
        "corrected_acc": corrected_acc,
        "per_task": stats,
        "contributions": contributions,
        "free_credit_tasks_excluded": list(FREE_CREDIT_TASKS),
    }


def render_single(p: Path, agg: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Corrected accuracy: `{p}`")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Total samples: **{agg['total_n']}**")
    lines.append(f"- Standard aggregate (mean score over all samples): **{agg['standard_acc']*100:.2f}%**")
    lines.append(f"- Corrected aggregate (mean pass-rate over real tasks, excluding {', '.join(agg['free_credit_tasks_excluded'])}): **{agg['corrected_acc']*100:.2f}%**")
    lines.append(f"- Delta (corrected vs standard): **{(agg['corrected_acc'] - agg['standard_acc'])*100:+.2f}pp** — negative means free-credit tasks were inflating the headline; positive means they were hurting it.")
    lines.append("")
    lines.append("## Per-task breakdown")
    lines.append("")
    lines.append("| task | n | pass rate | mean score | contribution to standard aggregate |")
    lines.append("|---|---:|---:|---:|---:|")
    for t in sorted(agg["per_task"]):
        d = agg["per_task"][t]
        c = agg["contributions"][t]
        free = " *(excluded from corrected)*" if t in FREE_CREDIT_TASKS else ""
        lines.append(
            f"| `{t}`{free} | {d['n']} | {d['pass_rate']*100:.1f}% | {d['mean_score']*100:.1f} | {c*100:+.2f}pp |"
        )
    lines.append("")
    return "\n".join(lines)


def render_compare(p_a: Path, agg_a: dict, p_b: Path, agg_b: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Corrected-acc compare: A=`{p_a}`  vs  B=`{p_b}`")
    lines.append("")
    lines.append("## Headlines")
    lines.append("")
    lines.append("| Metric | A | B | Δ (B − A) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Standard aggregate (%) | {agg_a['standard_acc']*100:.2f} | "
        f"{agg_b['standard_acc']*100:.2f} | "
        f"{(agg_b['standard_acc']-agg_a['standard_acc'])*100:+.2f}pp |"
    )
    lines.append(
        f"| Corrected aggregate (%) | {agg_a['corrected_acc']*100:.2f} | "
        f"{agg_b['corrected_acc']*100:.2f} | "
        f"{(agg_b['corrected_acc']-agg_a['corrected_acc'])*100:+.2f}pp |"
    )
    lines.append(f"| Total samples | {agg_a['total_n']} | {agg_b['total_n']} | {agg_b['total_n']-agg_a['total_n']:+d} |")
    lines.append("")
    lines.append("## Per-task movement")
    lines.append("")
    lines.append("| task | n (A→B) | pass A | pass B | Δ | mean A | mean B | Δ |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    all_tasks = sorted(set(agg_a["per_task"]) | set(agg_b["per_task"]))
    for t in all_tasks:
        da = agg_a["per_task"].get(t, {"n": 0, "pass_rate": 0.0, "mean_score": 0.0})
        db = agg_b["per_task"].get(t, {"n": 0, "pass_rate": 0.0, "mean_score": 0.0})
        pa = da["pass_rate"] * 100
        pb = db["pass_rate"] * 100
        ma = da["mean_score"] * 100
        mb = db["mean_score"] * 100
        free = " *(free credit)*" if t in FREE_CREDIT_TASKS else ""
        lines.append(
            f"| `{t}`{free} | {da['n']}→{db['n']} | "
            f"{pa:.1f}% | {pb:.1f}% | {pb-pa:+.1f}pp | "
            f"{ma:.1f} | {mb:.1f} | {mb-ma:+.1f}pp |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if abs(agg_b["corrected_acc"] - agg_a["corrected_acc"]) < 0.005:
        lines.append("- Corrected acc moved by less than 0.5pp. Within noise; the change between A and B did not materially affect quant quality.")
    elif agg_b["corrected_acc"] > agg_a["corrected_acc"]:
        lines.append(f"- Corrected acc IMPROVED by {(agg_b['corrected_acc']-agg_a['corrected_acc'])*100:.2f}pp. Look at the largest positive per-task Δ to identify which task drove the improvement.")
    else:
        lines.append(f"- Corrected acc REGRESSED by {(agg_a['corrected_acc']-agg_b['corrected_acc'])*100:.2f}pp. Look at the largest negative per-task Δ to identify which task drove the regression.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="predictions.jsonl path")
    ap.add_argument("--compare", default=None,
                    help="Optional second predictions.jsonl to compare against")
    ap.add_argument("--output-md", default=None,
                    help="Write markdown report here")
    ap.add_argument("--output-json", default=None,
                    help="Write structured json here (always the --input agg; "
                         "with --compare, writes both as a list)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"input not found: {in_path}", file=sys.stderr)
        return 2

    rows_a = load_predictions(in_path)
    agg_a = aggregate(rows_a)

    if args.compare:
        cmp_path = Path(args.compare)
        if not cmp_path.is_file():
            print(f"--compare path not found: {cmp_path}", file=sys.stderr)
            return 2
        rows_b = load_predictions(cmp_path)
        agg_b = aggregate(rows_b)
        report = render_compare(in_path, agg_a, cmp_path, agg_b)
        print(f"A standard={agg_a['standard_acc']*100:.2f}% corrected={agg_a['corrected_acc']*100:.2f}% n={agg_a['total_n']}")
        print(f"B standard={agg_b['standard_acc']*100:.2f}% corrected={agg_b['corrected_acc']*100:.2f}% n={agg_b['total_n']}")
    else:
        agg_b = None
        report = render_single(in_path, agg_a)
        print(f"standard={agg_a['standard_acc']*100:.2f}% corrected={agg_a['corrected_acc']*100:.2f}% n={agg_a['total_n']}")

    if args.output_md:
        Path(args.output_md).write_text(report)
        print(f"[compute_corrected_acc] wrote {args.output_md}", file=sys.stderr)
    else:
        print()
        print(report)

    if args.output_json:
        payload = agg_a if agg_b is None else [agg_a, agg_b]
        Path(args.output_json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"[compute_corrected_acc] wrote {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
