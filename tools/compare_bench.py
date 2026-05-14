#!/usr/bin/env python3
"""
Compare sglang bench_serving JSON outputs and print a side-by-side table.

Usage:
  python tools/compare_bench.py file1.bench.json file2.bench.json ...
  python tools/compare_bench.py /path/to/sweep_dir/*.bench.json
  python tools/compare_bench.py *.bench.json --output summary.md

The first file passed (or alphabetically first if a glob) is the baseline;
deltas are reported relative to that.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


# Metrics to display, in the order they appear in the table.
# Field names match sglang.bench_serving's JSONL schema (see python/sglang/bench_serving.py:2520).
# (key_in_json, display_name, unit, "higher_is_better")
METRICS = [
    ("output_throughput",         "out_tput",   "tok/s",  True),
    ("total_throughput",          "tot_tput",   "tok/s",  True),
    ("input_throughput",          "in_tput",    "tok/s",  True),
    ("max_output_tokens_per_s",   "peak_out",   "tok/s",  True),
    ("request_throughput",        "req/s",      "req/s",  True),
    ("mean_ttft_ms",              "TTFT_mean",  "ms",     False),
    ("median_ttft_ms",            "TTFT_p50",   "ms",     False),
    ("p99_ttft_ms",               "TTFT_p99",   "ms",     False),
    ("mean_itl_ms",               "ITL_mean",   "ms",     False),
    ("median_itl_ms",             "ITL_p50",    "ms",     False),
    ("p99_itl_ms",                "ITL_p99",    "ms",     False),
    ("mean_tpot_ms",              "TPOT_mean",  "ms",     False),
    ("mean_e2e_latency_ms",       "E2E_mean",   "ms",     False),
    ("duration",                  "duration",   "s",      False),
    ("completed",                 "succ",       "req",    None),
]


def load_bench(path: str) -> dict:
    """Read either a single JSON object or a JSONL file. Returns the LAST record
    if multiple (bench_serving appends; the last is the most recent run)."""
    with open(path) as f:
        text = f.read().strip()
    if not text:
        raise ValueError("empty file")
    # Try full-file JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to JSONL: take the last non-empty line
    last = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    if last is None:
        raise ValueError("no parseable JSON record found")
    return last


def find_metric(d: dict, key: str):
    # Try a few naming variations sglang has used over time
    candidates = [
        key,
        key.replace("_ms", ""),
        key.replace("_ms", "_msec"),
    ]
    for c in candidates:
        if c in d:
            return d[c]
    return None


def fmt(v, unit):
    if v is None:
        return "n/a"
    if isinstance(v, (int,)):
        return f"{v}"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.3f}"
    return str(v)


def pct_delta(new, base, higher_is_better):
    if base is None or new is None or base == 0:
        return ""
    delta = (new - base) / base * 100
    sign = "+" if delta >= 0 else ""
    arrow = ""
    if higher_is_better is True:
        arrow = " 🚀" if delta > 1 else (" ⚠️ " if delta < -1 else "")
    elif higher_is_better is False:
        arrow = " 🚀" if delta < -1 else (" ⚠️ " if delta > 1 else "")
    return f"{sign}{delta:.1f}%{arrow}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="bench_serving JSON files")
    ap.add_argument("--output", "-o", default="-",
                    help="output markdown file ('-' for stdout)")
    ap.add_argument("--baseline", default=None,
                    help="explicit baseline tag (filename stem); default = first file")
    args = ap.parse_args()

    # Expand globs and dedupe
    expanded = []
    for pat in args.files:
        matches = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        for m in matches:
            if m not in expanded:
                expanded.append(m)
    if not expanded:
        print("no input files", file=sys.stderr)
        sys.exit(2)

    benches = []
    for p in expanded:
        try:
            data = load_bench(p)
        except Exception as e:
            print(f"warn: skip {p}: {e}", file=sys.stderr)
            continue
        tag = Path(p).stem.replace(".bench", "")
        benches.append((tag, data, p))

    if not benches:
        print("no readable bench files", file=sys.stderr)
        sys.exit(2)

    # Resolve baseline
    base_idx = 0
    if args.baseline:
        for i, (tag, _, _) in enumerate(benches):
            if tag == args.baseline:
                base_idx = i
                break
    base_tag, base_data, _ = benches[base_idx]

    # Build markdown
    lines = []
    lines.append(f"# Bench comparison")
    lines.append("")
    lines.append(f"Baseline: **{base_tag}**")
    lines.append(f"Configs: {len(benches)}")
    lines.append("")

    # Table header: metric | base | each_other | delta(each)
    headers = ["metric"]
    for tag, _, _ in benches:
        if tag == base_tag:
            headers.append(f"**{tag}** (base)")
        else:
            headers.append(tag)
            headers.append(f"Δ vs base")

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for key, name, unit, hib in METRICS:
        row = [f"{name} ({unit})"]
        base_v = find_metric(base_data, key)
        for tag, data, _ in benches:
            v = find_metric(data, key)
            if tag == base_tag:
                row.append(fmt(v, unit))
            else:
                row.append(fmt(v, unit))
                row.append(pct_delta(v, base_v, hib))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Files")
    for tag, _, path in benches:
        lines.append(f"- `{tag}` -> {path}")
    lines.append("")

    output_text = "\n".join(lines)

    if args.output == "-":
        print(output_text)
    else:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
