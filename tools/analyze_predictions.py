#!/usr/bin/env python3
"""Analyze a single predictions.jsonl: per-task scores, output length, failure modes.

Schema expected (from SOAR-Toolkit eval_model.py output):
    index, task, question, gold, prediction, score, extracted,
    input_tokens, output_tokens

Important schema notes (verified on outputs/20260520_234226/predictions.jsonl):
    - `score` is partial-credit-capable. cwe rows produce score in [0.0, 1.0]
      in 0.1 steps; mcq is strict {0, 1}; niah/qa/fwe are {0.0, 1.0} floats.
      We always compute mean score (the true accuracy metric) plus a binary
      pass-rate at score >= 0.5 for intuition.
    - `extracted` is often the literal string "None" for non-MCQ tasks
      because eval_model.py grades `prediction` directly (fuzzy/substring
      match against `gold`). Empty `extracted` does NOT imply a format
      error outside of MCQ.

Usage:
    python3 tools/analyze_predictions.py \\
        --input outputs/20260520_234226/predictions.jsonl

    python3 tools/analyze_predictions.py \\
        --input outputs/20260521_xxxxxx/predictions.jsonl \\
        --output-md analysis_expB.md \\
        --top-failures 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# IO + low-level helpers
# ---------------------------------------------------------------------------
def load_predictions(path: Path) -> list[dict]:
    """Load a JSONL file. Skips (with a warning) lines that fail to parse."""
    rows = []
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
    """Get score as float, returning None when missing / unparseable."""
    if "score" not in row:
        return None
    try:
        return float(row["score"])
    except (TypeError, ValueError):
        return None


def is_pass(row: dict) -> bool:
    s = _score(row)
    return s is not None and s >= 0.5


def is_fail(row: dict) -> bool:
    """A row is failed only if it has a real score below 0.5.

    Rows with a missing/unparseable score are NOT counted as failed — they
    are dropped from rate denominators, so a corrupt row never silently
    inflates the failure count.
    """
    s = _score(row)
    return s is not None and s < 0.5


def quantiles(vals: list[float], qs: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1.0)) -> list[float | None]:
    if not vals:
        return [None] * len(qs)
    s = sorted(vals)
    return [s[min(int(q * (len(s) - 1)), len(s) - 1)] for q in qs]


def short_repr(s, limit: int = 80) -> str:
    if s is None:
        return "<None>"
    s = str(s).replace("\n", " | ")
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


def fnum(x, sig: int = 1) -> str:
    if x is None:
        return "?"
    try:
        return f"{float(x):.{sig}f}"
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------
def summarize_overall(rows: list[dict]) -> str:
    n = len(rows)
    scored = [r for r in rows if _score(r) is not None]
    n_scored = len(scored)
    if not scored:
        return f"Total samples: {n}\n(no scoreable rows — all rows had missing/unparseable `score`)"
    mean = statistics.mean(_score(r) for r in scored)
    pass_n = sum(1 for r in scored if is_pass(r))
    fail_n = n_scored - pass_n
    return (
        f"Total samples: {n}\n"
        f"  scoreable: {n_scored}\n"
        f"  skipped (missing score): {n - n_scored}\n"
        f"Mean score (continuous): {mean * 100:.2f}%   "
        f"<-- this is what SOAR-Toolkit reports as Average Score\n"
        f"Binary pass rate (score>=0.5): {pass_n}/{n_scored} = "
        f"{pass_n / n_scored * 100:.2f}%\n"
        f"  passed: {pass_n}\n"
        f"  failed: {fail_n}"
    )


def summarize_per_task(rows: list[dict]) -> tuple[str, dict[str, list[dict]]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[str(r.get("task", "?"))].append(r)
    lines = [
        f"{'task':<8} {'n':>4} {'mean':>7} {'pass%':>7} "
        f"{'avg_in':>10} {'avg_out':>10} {'med_out':>10}",
        "-" * 64,
    ]
    for task in sorted(by_task):
        items = by_task[task]
        scored = [r for r in items if _score(r) is not None]
        if scored:
            mean_pct = statistics.mean(_score(r) for r in scored) * 100
            passed = sum(1 for r in scored if is_pass(r))
            pass_pct = passed / len(scored) * 100
        else:
            mean_pct = float("nan")
            pass_pct = float("nan")
        in_toks = [r.get("input_tokens") or 0 for r in items]
        out_toks = [r.get("output_tokens") or 0 for r in items]
        avg_in = statistics.mean(in_toks) if in_toks else 0
        avg_out = statistics.mean(out_toks) if out_toks else 0
        med_out = statistics.median(out_toks) if out_toks else 0
        lines.append(
            f"{task:<8} {len(items):>4} {mean_pct:>6.1f}% {pass_pct:>6.1f}% "
            f"{avg_in:>10,.0f} {avg_out:>10,.0f} {med_out:>10,.0f}"
        )
    return "\n".join(lines), by_task


def analyze_score_distribution(by_task: dict[str, list[dict]]) -> str:
    """Score quantiles per task. Crucial for partial-credit tasks (cwe)."""
    lines = ["Score quantiles (min / p25 / p50 / p75 / max) and bucket counts:"]
    for task in sorted(by_task):
        items = by_task[task]
        scores = [s for s in (_score(r) for r in items) if s is not None]
        if not scores:
            lines.append(f"  {task:<8} (no scoreable rows)")
            continue
        qs = quantiles(scores)
        zero = sum(1 for s in scores if s == 0.0)
        full = sum(1 for s in scores if s == 1.0)
        partial = len(scores) - zero - full
        lines.append(
            f"  {task:<8} qs={[fnum(q, 2) for q in qs]}  "
            f"zero={zero}  partial={partial}  full={full}"
        )
    return "\n".join(lines)


def analyze_mcq_failures(by_task: dict[str, list[dict]]) -> str:
    """MCQ extracted should be {A,B,C,D}. Diagnose format vs content errors."""
    lines = ["MCQ extraction analysis:"]
    items = by_task.get("mcq", [])
    if not items:
        lines.append("  (no mcq samples)")
        return "\n".join(lines)
    failed = [r for r in items if is_fail(r)]
    none_extract = sum(
        1 for r in failed if str(r.get("extracted", "")).strip() in ("", "None", "null", "<None>")
    )
    wrong = len(failed) - none_extract
    lines.append(f"  Failed: {len(failed)}/{len(items)}")
    lines.append(f"    format-fail (extracted is None/empty): {none_extract}")
    lines.append(f"    content-fail (wrong choice picked):    {wrong}")
    confusion: Counter[tuple[str, str]] = Counter()
    for r in failed:
        g = str(r.get("gold", "?"))
        e = str(r.get("extracted", "?"))
        confusion[(g, e)] += 1
    if confusion:
        lines.append("  Confusion (gold -> extracted), top 10:")
        for (g, e), n in confusion.most_common(10):
            lines.append(f"    {g} -> {e}: {n}x")
    return "\n".join(lines)


def analyze_output_length_by_score(by_task: dict[str, list[dict]]) -> str:
    """Failed samples shorter (gave up) or longer (rambled)?"""
    lines = ["Output token avg by pass/fail:"]
    for task in sorted(by_task):
        items = by_task[task]
        passed = [r.get("output_tokens") or 0 for r in items if is_pass(r)]
        failed = [r.get("output_tokens") or 0 for r in items if is_fail(r)]
        p_avg = statistics.mean(passed) if passed else None
        f_avg = statistics.mean(failed) if failed else None
        delta = (f_avg - p_avg) if (p_avg is not None and f_avg is not None) else None
        delta_s = f"  Δ={delta:+,.0f}" if delta is not None else ""
        lines.append(
            f"  {task:<8} pass(n={len(passed):>3}): {fnum(p_avg, 0):>7}   "
            f"fail(n={len(failed):>3}): {fnum(f_avg, 0):>7}{delta_s}"
        )
    return "\n".join(lines)


def top_failures(by_task: dict[str, list[dict]], limit: int = 3) -> str:
    """Sample longest failed predictions for visual inspection."""
    lines = [f"Top {limit} failures per task (longest first):"]
    for task in sorted(by_task):
        items = [r for r in by_task[task] if is_fail(r)]
        if not items:
            continue
        items.sort(key=lambda r: -(r.get("output_tokens") or 0))
        lines.append(f"\n  [{task}] {len(items)} failure(s) total")
        for r in items[:limit]:
            idx = r.get("index", "?")
            score = r.get("score", "?")
            gold = short_repr(r.get("gold", ""), 70)
            ext = short_repr(r.get("extracted", ""), 40)
            pred = str(r.get("prediction", ""))
            out_t = r.get("output_tokens", "?")
            pred_tail = short_repr(pred[-200:], 150)
            lines.append(
                f"    idx={idx} score={score} out_tokens={out_t}\n"
                f"      gold:        {gold}\n"
                f"      extracted:   {ext}\n"
                f"      pred[-200:]: {pred_tail}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_report(rows: list[dict], path: Path, top_n: int) -> str:
    sections = [f"# Predictions analysis: `{path}`\n"]
    sections.append("## Overall\n```")
    sections.append(summarize_overall(rows))
    sections.append("```\n")

    sections.append("## Per-task summary\n```")
    table, by_task = summarize_per_task(rows)
    sections.append(table)
    sections.append("```\n")

    sections.append("## Score distribution\n```")
    sections.append(analyze_score_distribution(by_task))
    sections.append("```\n")

    sections.append("## MCQ extraction (format vs content)\n```")
    sections.append(analyze_mcq_failures(by_task))
    sections.append("```\n")

    sections.append("## Output length by pass/fail\n```")
    sections.append(analyze_output_length_by_score(by_task))
    sections.append("```\n")

    sections.append("## Failure samples\n```")
    sections.append(top_failures(by_task, top_n))
    sections.append("```\n")
    return "\n".join(sections)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Path to predictions.jsonl")
    ap.add_argument("--output-md", default=None, help="Optional markdown output path")
    ap.add_argument(
        "--top-failures",
        type=int,
        default=3,
        help="Show top-N longest failed samples per task (default: 3)",
    )
    args = ap.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"input not found: {path}", file=sys.stderr)
        return 2
    rows = load_predictions(path)
    if not rows:
        print("no rows loaded", file=sys.stderr)
        return 2

    text = build_report(rows, path, args.top_failures)
    print(text)
    if args.output_md:
        Path(args.output_md).write_text(text)
        print(f"\n[analyze] wrote {args.output_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
