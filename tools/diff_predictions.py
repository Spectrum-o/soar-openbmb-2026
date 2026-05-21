#!/usr/bin/env python3
"""Diff two predictions.jsonl files joined by `index`.

Produces a per-task transition matrix (both_pass / regressed / improved /
both_fail), highlights regressed/improved samples, and computes a per-task
delta in mean continuous score.

Pass/fail is `score >= 0.5` per row (consistent with analyze_predictions.py).
Partial-credit changes (cwe especially) are also tracked separately as
"score delta" so a row going 0.7 -> 0.3 isn't hidden behind a binary view.

Usage:
    python3 tools/diff_predictions.py \\
        --baseline outputs/20260520_234226/predictions.jsonl \\
        --candidate outputs/20260521_xxxxxx/predictions.jsonl

    # with markdown output
    python3 tools/diff_predictions.py \\
        --baseline ... --candidate ... --output-md diff.md

The two files must share the `index` field. Missing-on-one-side indices are
reported separately and NOT included in the transition matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[load {path}] line {i}: skip bad json: {exc}", file=sys.stderr)
                continue
            if "index" not in row:
                print(f"[load {path}] line {i}: no `index` field, skipped", file=sys.stderr)
                continue
            rows[row["index"]] = row
    return rows


def _score(row: dict) -> float | None:
    if "score" not in row:
        return None
    try:
        return float(row["score"])
    except (TypeError, ValueError):
        return None


def passed(row: dict) -> bool:
    s = _score(row)
    return s is not None and s >= 0.5


def short(s, limit: int = 60) -> str:
    if s is None:
        return "<None>"
    s = str(s).replace("\n", " | ")
    return s if len(s) <= limit else s[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def compute_transitions(
    base: dict[int, dict], cand: dict[int, dict]
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]], list[dict]]:
    """Build per-task transition counts, mean-score deltas, and per-sample
    record list (for top-N sorting)."""
    shared = sorted(set(base) & set(cand))

    # by task: bucket counts and continuous mean deltas
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"both_pass": 0, "regressed": 0, "improved": 0, "both_fail": 0, "n": 0}
    )
    score_sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {"base_sum": 0.0, "cand_sum": 0.0, "n_scored": 0}
    )
    samples: list[dict] = []
    for idx in shared:
        b = base[idx]
        c = cand[idx]
        # Task is from baseline (consistent); if mismatch we still trust base.
        t = str(b.get("task", "?"))
        bs = _score(b)
        cs = _score(c)
        bp = passed(b)
        cp = passed(c)
        d = counts[t]
        d["n"] += 1
        if bp and cp:
            d["both_pass"] += 1
            kind = "both_pass"
        elif bp and not cp:
            d["regressed"] += 1
            kind = "regressed"
        elif (not bp) and cp:
            d["improved"] += 1
            kind = "improved"
        else:
            d["both_fail"] += 1
            kind = "both_fail"
        if bs is not None and cs is not None:
            sd = score_sums[t]
            sd["base_sum"] += bs
            sd["cand_sum"] += cs
            sd["n_scored"] += 1
        samples.append(
            {
                "index": idx,
                "task": t,
                "base_score": bs,
                "cand_score": cs,
                "delta": (cs - bs) if (bs is not None and cs is not None) else None,
                "kind": kind,
                "base_extracted": b.get("extracted"),
                "cand_extracted": c.get("extracted"),
                "gold": b.get("gold"),
                "base_out_tokens": b.get("output_tokens"),
                "cand_out_tokens": c.get("output_tokens"),
            }
        )
    return counts, score_sums, samples


def format_transition_table(
    counts: dict[str, dict[str, int]], score_sums: dict[str, dict[str, float]]
) -> str:
    lines = [
        f"{'task':<8} {'n':>4} {'both_pass':>9} {'improved':>8} {'regressed':>9} "
        f"{'both_fail':>9} {'base_acc':>9} {'cand_acc':>9} {'binary Δ':>9} {'mean Δ':>9}",
        "-" * 92,
    ]
    total = {"n": 0, "both_pass": 0, "regressed": 0, "improved": 0, "both_fail": 0}
    total_mean = {"base_sum": 0.0, "cand_sum": 0.0, "n_scored": 0}
    for task in sorted(counts):
        d = counts[task]
        if d["n"] == 0:
            continue
        base_pass = d["both_pass"] + d["regressed"]
        cand_pass = d["both_pass"] + d["improved"]
        bacc = base_pass / d["n"] * 100
        cacc = cand_pass / d["n"] * 100
        bin_d = cacc - bacc
        sd = score_sums[task]
        if sd["n_scored"]:
            bmean = sd["base_sum"] / sd["n_scored"] * 100
            cmean = sd["cand_sum"] / sd["n_scored"] * 100
            mean_d = cmean - bmean
            mean_d_str = f"{mean_d:+.1f}"
        else:
            mean_d_str = "?"
        lines.append(
            f"{task:<8} {d['n']:>4} {d['both_pass']:>9} {d['improved']:>8} "
            f"{d['regressed']:>9} {d['both_fail']:>9} {bacc:>8.1f}% {cacc:>8.1f}% "
            f"{bin_d:>+8.1f} {mean_d_str:>9}"
        )
        for k in total:
            total[k] += d[k]
        for k in total_mean:
            total_mean[k] += sd[k]
    if total["n"]:
        base_pass = total["both_pass"] + total["regressed"]
        cand_pass = total["both_pass"] + total["improved"]
        bacc = base_pass / total["n"] * 100
        cacc = cand_pass / total["n"] * 100
        bin_d = cacc - bacc
        if total_mean["n_scored"]:
            bmean = total_mean["base_sum"] / total_mean["n_scored"] * 100
            cmean = total_mean["cand_sum"] / total_mean["n_scored"] * 100
            mean_d = cmean - bmean
            mean_d_str = f"{mean_d:+.1f}"
        else:
            mean_d_str = "?"
        lines.append("-" * 92)
        lines.append(
            f"{'TOTAL':<8} {total['n']:>4} {total['both_pass']:>9} "
            f"{total['improved']:>8} {total['regressed']:>9} {total['both_fail']:>9} "
            f"{bacc:>8.1f}% {cacc:>8.1f}% {bin_d:>+8.1f} {mean_d_str:>9}"
        )
    return "\n".join(lines)


def show_samples_by_kind(samples: list[dict], kind: str, limit: int) -> str:
    """Show top-N samples in a given category. Sorted by |delta| desc."""
    sub = [s for s in samples if s["kind"] == kind]
    if not sub:
        return f"  (no {kind} samples)"
    sub.sort(key=lambda s: -abs(s["delta"] or 0))
    lines: list[str] = []
    for s in sub[:limit]:
        gold = short(s["gold"], 50)
        be = short(s["base_extracted"], 25)
        ce = short(s["cand_extracted"], 25)
        bs = s["base_score"]
        cs = s["cand_score"]
        bt = s["base_out_tokens"]
        ct = s["cand_out_tokens"]
        lines.append(
            f"    idx={s['index']:>3} [{s['task']:<4}] "
            f"score: {bs} -> {cs}  "
            f"out_tokens: {bt} -> {ct}\n"
            f"      gold:      {gold}\n"
            f"      base.ext:  {be}\n"
            f"      cand.ext:  {ce}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_report(
    baseline_path: Path,
    candidate_path: Path,
    base: dict[int, dict],
    cand: dict[int, dict],
    top_n: int,
) -> str:
    shared = sorted(set(base) & set(cand))
    only_base = sorted(set(base) - set(cand))
    only_cand = sorted(set(cand) - set(base))
    counts, score_sums, samples = compute_transitions(base, cand)

    sections: list[str] = []
    sections.append("# Predictions diff\n")
    sections.append(f"- baseline:  `{baseline_path}` ({len(base)} rows)")
    sections.append(f"- candidate: `{candidate_path}` ({len(cand)} rows)")
    sections.append(f"- shared:    {len(shared)} indices")
    if only_base:
        sections.append(f"- only in baseline: {only_base[:10]}{' ...' if len(only_base) > 10 else ''}")
    if only_cand:
        sections.append(f"- only in candidate: {only_cand[:10]}{' ...' if len(only_cand) > 10 else ''}")
    sections.append("")

    sections.append("## Per-task transition matrix\n```")
    sections.append(format_transition_table(counts, score_sums))
    sections.append("```\n")

    sections.append(f"## Regressed samples (top {top_n}, sorted by |Δ|)\n```")
    sections.append(show_samples_by_kind(samples, "regressed", top_n))
    sections.append("```\n")

    sections.append(f"## Improved samples (top {top_n}, sorted by |Δ|)\n```")
    sections.append(show_samples_by_kind(samples, "improved", top_n))
    sections.append("```\n")

    return "\n".join(sections)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--baseline", required=True, help="Baseline predictions.jsonl")
    ap.add_argument("--candidate", required=True, help="Candidate predictions.jsonl")
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    bp = Path(args.baseline)
    cp = Path(args.candidate)
    if not bp.is_file():
        print(f"baseline not found: {bp}", file=sys.stderr)
        return 2
    if not cp.is_file():
        print(f"candidate not found: {cp}", file=sys.stderr)
        return 2
    base = load(bp)
    cand = load(cp)
    if not base or not cand:
        print("empty input(s)", file=sys.stderr)
        return 2

    text = build_report(bp, cp, base, cand, args.top_n)
    print(text)
    if args.output_md:
        Path(args.output_md).write_text(text)
        print(f"\n[diff] wrote {args.output_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
