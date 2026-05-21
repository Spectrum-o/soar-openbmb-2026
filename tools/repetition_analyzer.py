#!/usr/bin/env python3
"""Detect and classify repetition-loop collapse in generation outputs.

For each prediction we look for the smallest substring s such that the tail
of the output equals s repeated k>=3 times. We then report where the loop
begins (char offset), how much of the output it consumed, and the 200-char
context immediately before the loop -- the physical "trigger" of collapse.

Schema expected (same as outputs/.../predictions.jsonl from eval_model.py):
    index, task, question, gold, prediction, score, extracted,
    input_tokens, output_tokens

Usage:
    python3 tools/repetition_analyzer.py \\
        --input outputs/20260520_234226/predictions.jsonl \\
        --compare-input outputs/20260520_224618/predictions.jsonl \\
        --output-md /tmp/repetition_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# IO + low-level helpers (mirror analyze_predictions.py style)
# ---------------------------------------------------------------------------
def load_predictions(path: Path) -> list[dict]:
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


def escape_inline(s: str, limit: int = 200) -> str:
    """One-line, printable preview of a substring."""
    if s is None:
        return ""
    out = s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if len(out) > limit:
        out = out[:limit] + "..."
    return out


# ---------------------------------------------------------------------------
# Core loop-detection
# ---------------------------------------------------------------------------
def find_tail_loop(text: str, max_len: int = 200, min_repeats: int = 3) -> dict | None:
    """Find the smallest substring s of length 1..max_len such that
    text endswith s*k for some k >= min_repeats. Returns None if not found.

    Returns dict with: substring, length, start_offset, repeat_count.
    """
    if not text:
        return None
    n = len(text)
    upper = min(max_len, n // min_repeats)
    for L in range(1, upper + 1):
        cand = text[n - L : n]
        # how many times does `cand` repeat at the tail?
        k = 1
        pos = n - L
        while pos - L >= 0 and text[pos - L : pos] == cand:
            k += 1
            pos -= L
        if k >= min_repeats:
            start = n - k * L
            # try to extend backwards: maybe the loop body actually started
            # earlier with a partial repeat ending at `start`. We accept
            # `start` as-is to keep "where the model fell in" stable.
            return {
                "substring": cand,
                "length": L,
                "start_offset": start,
                "repeat_count": k,
            }
    return None


def classify_loop_position(loop: dict | None, total_chars: int) -> str:
    if loop is None or total_chars == 0:
        return "clean"
    frac = loop["start_offset"] / total_chars
    if frac < 0.25:
        return "early_loop"
    if frac < 0.75:
        return "mid_loop"
    return "late_loop"


def trigger_context(text: str, loop: dict, ctx_chars: int = 200) -> str:
    start = loop["start_offset"]
    lo = max(0, start - ctx_chars)
    return text[lo:start]


# ---------------------------------------------------------------------------
# Per-row analysis
# ---------------------------------------------------------------------------
def analyze_row(row: dict) -> dict:
    pred = row.get("prediction", "") or ""
    total = len(pred)
    loop = find_tail_loop(pred)
    cls = classify_loop_position(loop, total)
    rec = {
        "index": row.get("index"),
        "task": row.get("task"),
        "total_chars": total,
        "output_tokens": row.get("output_tokens"),
        "input_tokens": row.get("input_tokens"),
        "class": cls,
        "loop": loop,
    }
    if loop is not None:
        rec["loop_substring"] = loop["substring"]
        rec["loop_length_chars"] = loop["length"]
        rec["loop_start_char_offset"] = loop["start_offset"]
        rec["loop_repeat_count"] = loop["repeat_count"]
        rec["tail_fraction_loop"] = (loop["repeat_count"] * loop["length"]) / total if total else 0
        rec["trigger_ctx"] = trigger_context(pred, loop)
    return rec


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------
def aggregate(recs: list[dict]) -> dict:
    classes = Counter(r["class"] for r in recs)
    looped = [r for r in recs if r["loop"] is not None]
    clean = [r for r in recs if r["loop"] is None]

    # top substrings -- normalize escape so equivalent strings collapse
    sub_counter: Counter[str] = Counter()
    sub_examples: dict[str, list] = defaultdict(list)
    for r in looped:
        key = escape_inline(r["loop_substring"], limit=200)
        sub_counter[key] += 1
        if len(sub_examples[key]) < 5:
            sub_examples[key].append(r["index"])

    # start position distribution
    starts_pct = []
    for r in looped:
        if r["total_chars"]:
            starts_pct.append(100 * r["loop_start_char_offset"] / r["total_chars"])

    # mean output tokens per class
    by_class_tokens: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        ot = r.get("output_tokens") or 0
        by_class_tokens[r["class"]].append(ot)

    # prompt-length correlation: simple Pearson between input_tokens and
    # is_looped (0/1). Stdlib only -- inline formula.
    xs = [r.get("input_tokens") or 0 for r in recs]
    ys = [0 if r["loop"] is None else 1 for r in recs]
    pearson = _pearson(xs, ys)

    # trigger context aggregation: what character/token immediately precedes
    # the loop?
    last_char_counter: Counter[str] = Counter()
    last_word_counter: Counter[str] = Counter()
    for r in looped:
        ctx = r.get("trigger_ctx", "")
        if ctx:
            last_char_counter[ctx[-1]] += 1
            # crude "last word" -- last whitespace-delimited token
            tail = ctx.rstrip()
            if tail:
                last_word = tail.split()[-1] if tail.split() else ""
                last_word_counter[escape_inline(last_word, 40)] += 1

    # tail_fraction stats
    tail_fracs = [r["tail_fraction_loop"] for r in looped]

    return {
        "n_total": len(recs),
        "n_looped": len(looped),
        "n_clean": len(clean),
        "classes": dict(classes),
        "top_substrings": sub_counter.most_common(10),
        "sub_examples": sub_examples,
        "starts_pct": starts_pct,
        "by_class_tokens": by_class_tokens,
        "pearson_promptlen_loop": pearson,
        "last_char_counter": last_char_counter,
        "last_word_counter": last_word_counter,
        "tail_fracs": tail_fracs,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _hist(values: list[float], bins: int = 10, lo: float = 0, hi: float = 100) -> list[str]:
    if not values:
        return ["  (no values)"]
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, max(0, int((v - lo) / step)))
        counts[idx] += 1
    out = []
    mx = max(counts) or 1
    for i, c in enumerate(counts):
        bar = "#" * int(40 * c / mx)
        out.append(f"  [{lo + i*step:>5.1f}, {lo + (i+1)*step:>5.1f}%) {c:>4}  {bar}")
    return out


def render_report(label: str, path: Path, recs: list[dict], stats: dict) -> str:
    L = [f"# Repetition analysis: `{path}` ({label})\n"]
    n_total = stats["n_total"]
    n_looped = stats["n_looped"]
    L.append("## Overall\n```")
    L.append(f"records:       {n_total}")
    L.append(f"looped:        {n_looped} ({100*n_looped/max(1,n_total):.1f}%)")
    L.append(f"clean:         {stats['n_clean']}")
    L.append(f"class counts:  {stats['classes']}")
    if stats["tail_fracs"]:
        L.append(
            f"tail_fraction_loop mean={statistics.mean(stats['tail_fracs']):.3f}  "
            f"median={statistics.median(stats['tail_fracs']):.3f}"
        )
    L.append("```\n")

    L.append("## Mean output_tokens per class\n```")
    for cls, toks in stats["by_class_tokens"].items():
        if toks:
            L.append(f"  {cls:<12} n={len(toks):>4}  mean={statistics.mean(toks):>8.0f}  "
                     f"median={statistics.median(toks):>8.0f}  max={max(toks):>8}")
    L.append("```\n")

    L.append("## Loop start position distribution (% of total output chars)\n```")
    L.extend(_hist(stats["starts_pct"], bins=10))
    L.append("```\n")

    L.append("## Top-10 loop substrings\n```")
    if not stats["top_substrings"]:
        L.append("  (none)")
    for sub, n in stats["top_substrings"]:
        ex = stats["sub_examples"].get(sub, [])
        L.append(f"  [{n:>3}x] examples={ex}")
        L.append(f"         sub={sub!r}")
    L.append("```\n")

    L.append("## Trigger context (char immediately before loop)\n```")
    L.append("  Top last-characters before loop start:")
    for ch, n in stats["last_char_counter"].most_common(10):
        L.append(f"    {ch!r:>6}  {n}")
    L.append("  Top last-whitespace-tokens before loop start:")
    for w, n in stats["last_word_counter"].most_common(10):
        L.append(f"    [{n:>3}x] {w}")
    L.append("```\n")

    L.append("## Prompt length correlation\n```")
    p = stats["pearson_promptlen_loop"]
    L.append(f"  Pearson(input_tokens, is_looped) = {p if p is None else f'{p:.3f}'}")
    L.append("```\n")

    L.append("## Per-record loop catalog (first 30 looped rows)\n```")
    looped = [r for r in recs if r["loop"] is not None]
    for r in looped[:30]:
        L.append(
            f"  idx={r['index']:<4} task={r['task']:<5} class={r['class']:<11} "
            f"out_tok={r.get('output_tokens'):<6} loop_len={r['loop_length_chars']:<4} "
            f"start={r['loop_start_char_offset']:<6} reps={r['loop_repeat_count']:<5} "
            f"tail_frac={r['tail_fraction_loop']:.2f}"
        )
        L.append(f"     trig=...{escape_inline(r['trigger_ctx'][-80:], 80)}")
        L.append(f"     sub ={escape_inline(r['loop_substring'], 80)}")
    L.append("```\n")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_one(path: Path, label: str) -> tuple[str, dict]:
    rows = load_predictions(path)
    recs = [analyze_row(r) for r in rows]
    stats = aggregate(recs)
    return render_report(label, path, recs, stats), stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="Path to predictions.jsonl")
    ap.add_argument("--compare-input", default=None, help="Optional second jsonl for contrast")
    ap.add_argument("--output-md", default=None, help="Optional markdown output path")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"input not found: {path}", file=sys.stderr)
        return 2

    report, stats = run_one(path, "primary")

    if args.compare_input:
        cpath = Path(args.compare_input)
        if cpath.is_file():
            report2, stats2 = run_one(cpath, "compare")
            report = report + "\n\n---\n\n" + report2
        else:
            print(f"[warn] compare input not found: {cpath}", file=sys.stderr)

    if args.output_md:
        Path(args.output_md).write_text(report)
        print(f"[repetition] wrote {args.output_md}", file=sys.stderr)

    # one-line stdout summary on primary file
    cls = stats["classes"]
    print(
        f"primary={path.name}  n={stats['n_total']}  looped={stats['n_looped']}  "
        f"early={cls.get('early_loop',0)} mid={cls.get('mid_loop',0)} "
        f"late={cls.get('late_loop',0)} clean={cls.get('clean',0)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
