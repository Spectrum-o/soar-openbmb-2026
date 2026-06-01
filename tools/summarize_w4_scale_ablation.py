#!/usr/bin/env python3
"""Summarize a W4 scale recalibration ablation work directory.

Input is the directory produced by scripts/w4_scale_recalib_ablation.sh.
The summary is intentionally conservative: it reports measured deltas and a
small decision hint, but it does not claim a platform result from local random
benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_float(value: str | None) -> float | None:
    if value is None or value in ("", "-", "?"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_bench_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (row.get("tag", ""), row.get("profile", "")): row
            for row in csv.DictReader(f)
        }


def extract_eval_score(path: Path) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r'"acc_ori"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        r"ori_accuracy[^0-9-]*([0-9]+(?:\.[0-9]+)?)",
        r"Average Score[^0-9-]*([0-9]+(?:\.[0-9]+)?)",
    ]
    found: list[float] = []
    for pat in patterns:
        found.extend(float(x) for x in re.findall(pat, text, flags=re.I))
    return found[-1] if found else None


def pct_delta(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (new - old) / old * 100.0


def fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def bench_pct(
    bench: dict[tuple[str, str], dict[str, str]], profile: str, field: str
) -> float | None:
    orig = parse_float(bench.get(("orig", profile), {}).get(field))
    recal = parse_float(bench.get(("recal", profile), {}).get(field))
    if orig is None or recal is None:
        return None
    return pct_delta(recal, orig)


def scale_gain(apply_report: dict[str, Any] | None) -> float | None:
    if not apply_report:
        return None
    gain = apply_report.get("summary", {}).get("avg_mse_gain_pct")
    return float(gain) if isinstance(gain, (int, float)) else None


def bottleneck_lines(
    apply_report: dict[str, Any] | None,
    bench: dict[tuple[str, str], dict[str, str]],
    eval_orig: float | None,
    eval_recal: float | None,
) -> list[str]:
    lines: list[str] = []
    gain = scale_gain(apply_report)
    eval_delta = (
        eval_recal - eval_orig
        if eval_orig is not None and eval_recal is not None
        else None
    )
    decode_out = bench_pct(bench, "decode", "output_tok_s")
    concurrent_out = bench_pct(bench, "concurrent", "output_tok_s")
    longctx_out = bench_pct(bench, "longctx", "output_tok_s")
    longctx_ttft = bench_pct(bench, "longctx", "mean_ttft_ms")

    lines.append("- Scale-only rewrite should not materially change speed. Treat >3% speed deltas as noise/JIT/server-arg evidence until repeated.")
    if gain is not None:
        lines.append(f"- weight_mse_gain_pct: {fmt(gain, '%')}")
    if eval_delta is not None:
        lines.append(f"- eval_acc_ori_delta: {fmt(eval_delta)}")
    if decode_out is not None:
        lines.append(f"- decode_output_tok_s_delta: {fmt(decode_out, '%')}")
    if concurrent_out is not None:
        lines.append(f"- concurrent_output_tok_s_delta: {fmt(concurrent_out, '%')}")
    if longctx_out is not None:
        lines.append(f"- longctx_output_tok_s_delta: {fmt(longctx_out, '%')}")
    if longctx_ttft is not None:
        lines.append(f"- longctx_ttft_delta: {fmt(longctx_ttft, '%')} (positive is slower prefill/first-token)")

    if eval_delta is not None and eval_delta >= 0.5:
        lines.append("- Bottleneck read: W4 scale reconstruction is a plausible accuracy bottleneck. Package only if a repeat keeps speed neutral and dequant verification stays clean.")
    elif eval_delta is not None and eval_delta <= -0.5:
        lines.append("- Bottleneck read: global weight-only scale rewrite is accuracy-negative. Try a narrower module regex before considering any package.")
    elif eval_delta is not None:
        if gain is not None and gain > 1.0:
            lines.append("- Bottleneck read: static weight reconstruction improved but accuracy stayed flat; current bottleneck is likely not global `.scales` MSE. Look at module selection, activation-aware calibration, or runtime kernels.")
        else:
            lines.append("- Bottleneck read: neither reconstruction nor accuracy moved enough. W4 scale recalibration is probably not the next strong lever.")
    else:
        lines.append("- Bottleneck read: eval is missing, so accuracy bottleneck is unclassified. Use bench rows only for runtime pressure.")

    runtime_deltas = [x for x in (decode_out, concurrent_out, longctx_out) if x is not None]
    if runtime_deltas and all(abs(x) <= 3.0 for x in runtime_deltas):
        lines.append("- Runtime read: throughput is flat across the random bench matrix; scale-only rewrite has no speed effect, as expected.")
    elif runtime_deltas:
        lines.append("- Runtime read: at least one throughput profile moved by more than 3%; repeat with the same warmed environment before attributing this to scale tensors.")
    if longctx_ttft is not None and longctx_ttft > 5.0:
        lines.append("- Long-context read: recalibrated run has worse TTFT. Since scale-only should not affect prefill cost, inspect server logs, CUDA graph/JIT warmup, and GPU memory pressure.")
    return lines


def summarize(work_dir: Path) -> str:
    apply_report = read_json(work_dir / "w4_scale_recalib_apply.json")
    bench = read_bench_summary(work_dir / "bench_summary.csv")
    eval_orig = extract_eval_score(work_dir / "logs" / "eval_orig.log")
    eval_recal = extract_eval_score(work_dir / "logs" / "eval_recal.log")

    lines: list[str] = []
    lines.append("# W4 Scale Recalibration Summary")
    lines.append("")
    lines.append(f"work_dir: `{work_dir}`")

    if apply_report:
        summary = apply_report.get("summary", {})
        lines.append("")
        lines.append("## Scale Rewrite")
        lines.append(f"- modules_ok: {summary.get('modules_ok', 'n/a')}")
        lines.append(f"- modules_changed: {summary.get('modules_changed', 'n/a')}")
        lines.append(f"- modules_skipped: {summary.get('modules_skipped', 'n/a')}")
        gain = summary.get("avg_mse_gain_pct")
        lines.append(f"- avg_weight_mse_gain_pct: {fmt(float(gain) if isinstance(gain, (int, float)) else None, '%')}")

    profiles = sorted({profile for _, profile in bench})
    if profiles:
        lines.append("")
        lines.append("## Bench Deltas")
        lines.append("")
        lines.append("| profile | orig out tok/s | recal out tok/s | delta | orig TTFT ms | recal TTFT ms |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for profile in profiles:
            orig = bench.get(("orig", profile), {})
            recal = bench.get(("recal", profile), {})
            orig_out = parse_float(orig.get("output_tok_s"))
            recal_out = parse_float(recal.get("output_tok_s"))
            orig_ttft = parse_float(orig.get("mean_ttft_ms"))
            recal_ttft = parse_float(recal.get("mean_ttft_ms"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        profile,
                        fmt(orig_out),
                        fmt(recal_out),
                        fmt(pct_delta(recal_out, orig_out) if orig_out is not None and recal_out is not None else None, "%"),
                        fmt(orig_ttft),
                        fmt(recal_ttft),
                    ]
                )
                + " |"
            )

    if eval_orig is not None or eval_recal is not None:
        lines.append("")
        lines.append("## Eval Delta")
        delta = (
            eval_recal - eval_orig
            if eval_orig is not None and eval_recal is not None
            else None
        )
        lines.append(f"- orig_acc_ori: {fmt(eval_orig)}")
        lines.append(f"- recal_acc_ori: {fmt(eval_recal)}")
        lines.append(f"- delta_acc_ori: {fmt(delta)}")

    lines.append("")
    lines.append("## Bottleneck Read")
    lines.extend(bottleneck_lines(apply_report, bench, eval_orig, eval_recal))

    lines.append("")
    lines.append("## Decision Hint")
    if apply_report is None:
        lines.append("- Missing scale report; first check whether recalibration actually ran.")
    elif eval_orig is not None and eval_recal is not None:
        delta = eval_recal - eval_orig
        if delta >= 0.5:
            lines.append("- Recalibration improved public-set accuracy. Next bottleneck test should check whether the same artifact keeps platform speed and package constraints.")
        elif delta <= -0.5:
            lines.append("- Recalibration hurt public-set accuracy. Do not package this globally; try a narrower module regex or abandon weight-only scale LS.")
        else:
            lines.append("- Accuracy is flat. Weight-only W4 scale error is probably not the main bottleneck; focus on module selection, runtime kernels, or activation-aware calibration.")
    else:
        lines.append("- Eval logs are missing. Use bench deltas only for runtime bottlenecks; run with `--eval` before making an accuracy decision.")

    lines.append("- Random benchmark deltas locate runtime pressure but are not a platform-score substitute.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    text = summarize(args.work_dir)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
