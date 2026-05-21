#!/usr/bin/env python3
"""Scrape experiment logs for headline metrics → results.csv + summary.md.

Designed to ingest scripts/logs/local_eval_{quant,server,eval}_*.log and
/root/autodl-fs/zyn/logs/*.log. Resilient: every field can be missing
without crashing (we record "?"), so the tool is safe to run mid-experiment.

Regex anchors (verified against scripts/logs/):

  eval logs
    "Average Score: XX.XX%"                       — final eval accuracy
    "Generation completed in XX seconds"          — wall time
    "Processing time: XX.XX s"                    — server-side time
    "Total Tokens: In=X, Out=Y"
    "Average Tokens/Sample: In=XX, Out=XX"
    "Overall TPS (Output): XX tokens/s"
    "Testing with X samples"
    "Saving results to outputs/XXXX"              — predictions dir

  quant logs (prepare_model.sh + quantize_gptqmodel_w4a16.py)
    "[prepare_model] using calibration from: X"
    "[prepare_model] chat template DISABLED (DISABLE_CHAT_TEMPLATE=1)"
    "[prepare_model] quantize timeout: XX min"
    "[calib] loaded XX prompts from XX"
    "[calib] task distribution: {...}"
    "[calib] applying chat template..." / "[calib] chat template DISABLED..."
    "[calib] multi-adaptive: N prompts -> M windows ..."
    "[versions] PKG=VERSION"
    "[register] layer_modules=[...]"
    "[quantize] starting GPTQModel W4A16 group_size=128 samples=256 max_len=8192"
    "[qzeros-fix] total N qzeros tensors patched..."

  any (crash detection)
    "Traceback (most recent call last)"
    "out of memory" | "OOM" | "OutOfMemoryError" | "CUDA error"
    "Killed" | "SIGKILL"

Usage:
    python3 tools/parse_experiment_logs.py \\
        --log-dir scripts/logs \\
        --output-csv results.csv \\
        --output-md summary.md

    # Scan multiple dirs
    python3 tools/parse_experiment_logs.py \\
        --log-dir scripts/logs \\
        --log-dir /root/autodl-fs/zyn/logs \\
        --output-csv /tmp/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Patterns. Each value: compiled regex, or (regex, group-index/name) tuple.
# All patterns are designed to match the LAST occurrence in a file (we use
# re.findall and take [-1]).
# ---------------------------------------------------------------------------
_RE = {
    # eval metrics
    "average_score":   re.compile(r"Average Score:\s*([\d.]+)\s*%"),
    "generation_time": re.compile(r"Generation completed in\s*([\d.]+)\s*seconds"),
    "processing_time": re.compile(r"Processing time:\s*([\d.]+)\s*s"),
    "total_in_toks":   re.compile(r"Total Tokens:\s*In=(\d+)"),
    "total_out_toks":  re.compile(r"Total Tokens:.*Out=(\d+)"),
    "avg_in_toks":     re.compile(r"Average Tokens/Sample:\s*In=([\d.]+)"),
    "avg_out_toks":    re.compile(r"Average Tokens/Sample:.*Out=([\d.]+)"),
    "tps_output":      re.compile(r"Overall TPS \(Output\):\s*([\d.]+)\s*tokens/s"),
    "num_samples":     re.compile(r"Testing with\s*(\d+)\s*samples"),
    "outputs_dir":     re.compile(r"Saving results to\s*(\S+)"),
    # legacy / alternate accuracy formats
    "ori_accuracy":    re.compile(r"ori_accuracy[^\d-]*([\d.]+)"),
    "overall_acc":     re.compile(r"overall_accuracy[^\d-]*([\d.]+)"),
    # quant config
    "calib_jsonl":     re.compile(r"\[prepare_model\] using calibration from:\s*(\S+)"),
    "calib_chat_off":  re.compile(r"\[prepare_model\] chat template DISABLED"),
    "calib_window":    re.compile(r"\[calib\] multi-adaptive:\s*(\d+)\s*prompts\s*->\s*(\d+)\s*windows"),
    "calib_applying":  re.compile(r"\[calib\] applying chat template to calibration prompts"),
    "calib_no_tpl":    re.compile(r"\[calib\] chat template DISABLED via --no-chat-template"),
    "calib_no_meta":   re.compile(r"\[calib\] tokenizer has no chat_template"),
    "calib_num":       re.compile(r"\[calib\] loaded\s*(\d+)\s*prompts"),
    "calib_dist":      re.compile(r"\[calib\] task distribution:\s*(\{[^}]+\})"),
    "quant_start":     re.compile(r"\[quantize\] starting GPTQModel W(\d+)A16\s*group_size=(\d+)\s*samples=(\d+)\s*max_len=(\d+)"),
    "qzeros_patched":  re.compile(r"\[qzeros-fix\] total\s*(\d+)\s*qzeros tensors patched"),
    "module_set":      re.compile(r"\[register\] layer_modules=(\[\[.+?\]\])"),
    "quant_timeout":   re.compile(r"\[prepare_model\] quantize timeout:\s*(\d+)\s*min"),
    "version_torch":   re.compile(r"\[versions\] torch=(\S+)"),
    "version_gptqm":   re.compile(r"\[versions\] gptqmodel=(\S+)"),
    "version_xform":   re.compile(r"\[versions\] transformers=(\S+)"),
    # crashes
    "traceback":       re.compile(r"^Traceback \(most recent call last\)", re.MULTILINE),
    "oom":             re.compile(r"out of memory|OutOfMemoryError|CUDA error|OOM", re.IGNORECASE),
    "killed":          re.compile(r"\bKilled\b|SIGKILL|SIGTERM", re.IGNORECASE),
    "json_decode_err": re.compile(r"json\.decoder\.JSONDecodeError"),
}

# Field group: what we report and in what order in the CSV.
CSV_FIELDS = [
    "filename",
    "size_kb",
    "variant",
    "stage",
    "timestamp",
    # eval
    "average_score",
    "ori_accuracy",
    "overall_acc",
    "generation_time",
    "processing_time",
    "tps_output",
    "num_samples",
    "avg_in_toks",
    "avg_out_toks",
    "total_in_toks",
    "total_out_toks",
    "outputs_dir",
    # quant
    "quant_w_bits",
    "quant_group_size",
    "quant_samples",
    "quant_max_len",
    "qzeros_patched",
    "calib_jsonl",
    "calib_num",
    "calib_chat_state",
    "calib_window_mode",
    "calib_dist",
    "module_set",
    "quant_timeout",
    # versions
    "version_torch",
    "version_gptqmodel",
    "version_transformers",
    # crashes
    "has_traceback",
    "has_oom",
    "has_killed",
    "has_json_decode_err",
]

# Filename like `local_eval_<stage>_<variant>_<unix_ts>.log` or
# `expB_no_chat_template_<date>_<time>.log` or arbitrary other.
_FILENAME_PAT = re.compile(r"local_eval_(eval|server|quant)_(.+)_(\d+)$")
_TEE_PAT = re.compile(r"^(exp[A-Z]\w*|quant_v\d+\w*)_(\d{8}_\d{6})$")


def parse_filename(path: Path) -> dict:
    """Extract metadata from log filename."""
    name = path.stem
    out = {"variant": "?", "stage": "?", "timestamp": "?"}
    m = _FILENAME_PAT.match(name)
    if m:
        out["stage"] = m.group(1)
        out["variant"] = m.group(2)
        out["timestamp"] = m.group(3)
        return out
    m = _TEE_PAT.match(name)
    if m:
        out["stage"] = "tee"
        out["variant"] = m.group(1)
        out["timestamp"] = m.group(2)
        return out
    return out


def _last_match(rx: re.Pattern[str], text: str) -> str | None:
    matches = rx.findall(text)
    if not matches:
        return None
    last = matches[-1]
    # `findall` returns tuples for multi-group regexes; we just want first group
    if isinstance(last, tuple):
        return last[0] if last else None
    return last


def _scan(rx: re.Pattern[str], text: str) -> bool:
    return rx.search(text) is not None


def scan_log(path: Path) -> dict:
    """Extract every field; missing fields become None."""
    out = {k: None for k in CSV_FIELDS}
    out["filename"] = path.name
    out["size_kb"] = max(1, path.stat().st_size // 1024)
    out.update(parse_filename(path))

    try:
        text = path.read_text(errors="replace")
    except Exception as exc:
        out["_read_error"] = str(exc)
        return out

    # Eval / common metrics — extract as raw strings; users can cast in spreadsheet.
    for key in (
        "average_score", "generation_time", "processing_time",
        "total_in_toks", "total_out_toks", "avg_in_toks", "avg_out_toks",
        "tps_output", "num_samples", "outputs_dir", "ori_accuracy", "overall_acc",
        "calib_jsonl", "calib_num", "calib_dist", "module_set", "quant_timeout",
        "version_torch", "version_gptqm", "version_xform", "qzeros_patched",
    ):
        v = _last_match(_RE[key], text)
        if v is not None:
            if key == "version_gptqm":
                out["version_gptqmodel"] = v
            elif key == "version_xform":
                out["version_transformers"] = v
            else:
                out[key] = v

    # Special: quant_start has 4 capture groups
    m = _RE["quant_start"].findall(text)
    if m:
        bits, gs, ns, ml = m[-1]
        out["quant_w_bits"] = bits
        out["quant_group_size"] = gs
        out["quant_samples"] = ns
        out["quant_max_len"] = ml

    # Special: window mode
    m = _RE["calib_window"].findall(text)
    if m:
        prompts, windows = m[-1]
        out["calib_window_mode"] = f"multi-adaptive ({prompts}p->{windows}w)"
    elif _RE["calib_applying"].search(text) or _RE["calib_no_tpl"].search(text):
        out["calib_window_mode"] = "tail"

    # Chat template state — three mutually exclusive log lines
    if _RE["calib_no_tpl"].search(text) or _RE["calib_chat_off"].search(text):
        out["calib_chat_state"] = "OFF"
    elif _RE["calib_applying"].search(text):
        out["calib_chat_state"] = "ON"
    elif _RE["calib_no_meta"].search(text):
        out["calib_chat_state"] = "n/a (no tokenizer chat_template)"

    # Crashes
    out["has_traceback"] = "yes" if _scan(_RE["traceback"], text) else "no"
    out["has_oom"] = "yes" if _scan(_RE["oom"], text) else "no"
    out["has_killed"] = "yes" if _scan(_RE["killed"], text) else "no"
    out["has_json_decode_err"] = "yes" if _scan(_RE["json_decode_err"], text) else "no"

    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # Convert None to empty cell (better for spreadsheets)
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS})


def fmt_pct(v) -> str:
    try:
        return f"{float(v):.2f}%"
    except (TypeError, ValueError):
        return "?"


def write_markdown(rows: list[dict], path: Path) -> None:
    eval_rows = [r for r in rows if r.get("stage") in ("eval", "tee")]
    quant_rows = [r for r in rows if r.get("stage") == "quant"]
    crashed = [r for r in rows if r.get("has_traceback") == "yes" or r.get("has_oom") == "yes"]

    lines: list[str] = [f"# Experiment log summary\n",
                        f"- scanned: {len(rows)} log file(s)",
                        f"- eval/tee logs: {len(eval_rows)}",
                        f"- quant logs: {len(quant_rows)}",
                        f"- crashed logs: {len(crashed)}\n"]

    if eval_rows:
        lines.append("## Eval results\n")
        lines.append("| timestamp | variant | acc | duration | tps | samples | chat tpl |")
        lines.append("|---|---|---|---|---|---|---|")
        eval_rows.sort(key=lambda r: r.get("timestamp", "") or "")
        for r in eval_rows:
            chat = r.get("calib_chat_state") or "?"
            lines.append(
                f"| {r.get('timestamp', '?')} "
                f"| `{r.get('variant', '?')}` "
                f"| {fmt_pct(r.get('average_score'))} "
                f"| {r.get('generation_time') or '?'}s "
                f"| {r.get('tps_output') or '?'} "
                f"| {r.get('num_samples') or '?'} "
                f"| {chat} |"
            )
        lines.append("")

    if quant_rows:
        lines.append("## Quant runs\n")
        lines.append("| timestamp | variant | W bits | group_size | samples | max_len | window | chat tpl | qzeros patched |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        quant_rows.sort(key=lambda r: r.get("timestamp", "") or "")
        for r in quant_rows:
            lines.append(
                f"| {r.get('timestamp', '?')} "
                f"| `{r.get('variant', '?')}` "
                f"| {r.get('quant_w_bits') or '?'} "
                f"| {r.get('quant_group_size') or '?'} "
                f"| {r.get('quant_samples') or '?'} "
                f"| {r.get('quant_max_len') or '?'} "
                f"| {r.get('calib_window_mode') or '?'} "
                f"| {r.get('calib_chat_state') or '?'} "
                f"| {r.get('qzeros_patched') or '?'} |"
            )
        lines.append("")

    if crashed:
        lines.append("## Crashes\n")
        for r in crashed:
            flags = []
            if r.get("has_traceback") == "yes":
                flags.append("traceback")
            if r.get("has_oom") == "yes":
                flags.append("OOM")
            if r.get("has_killed") == "yes":
                flags.append("killed")
            if r.get("has_json_decode_err") == "yes":
                flags.append("json-decode-err")
            lines.append(f"- `{r['filename']}` ({r['size_kb']}KB): {', '.join(flags) or 'unknown'}")
        lines.append("")

    Path(path).write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--log-dir",
        action="append",
        default=[],
        required=False,
        help="Directory to scan (can be passed multiple times)",
    )
    ap.add_argument(
        "--pattern",
        default="*.log",
        help='Glob pattern within each --log-dir (default: "*.log")',
    )
    ap.add_argument("--output-csv", default="experiment_results.csv")
    ap.add_argument("--output-md", default="experiment_summary.md")
    args = ap.parse_args()

    if not args.log_dir:
        # Default to scripts/logs if nothing passed
        args.log_dir = ["scripts/logs"]

    rows: list[dict] = []
    for d in args.log_dir:
        p = Path(d)
        if not p.is_dir():
            print(f"[parse] skip (not a dir): {p}", file=sys.stderr)
            continue
        files = sorted(p.rglob(args.pattern))
        print(f"[parse] scanning {p}: {len(files)} file(s)")
        for f in files:
            try:
                row = scan_log(f)
            except Exception as exc:
                print(f"[parse] error scanning {f}: {exc}", file=sys.stderr)
                continue
            rows.append(row)
            print(
                f"  {f.name:<70s}  "
                f"stage={row.get('stage'):<6s}  "
                f"acc={fmt_pct(row.get('average_score')):>8s}  "
                f"crashed={row.get('has_traceback')}"
            )

    if not rows:
        print("[parse] no logs scanned", file=sys.stderr)
        return 1

    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, Path(args.output_md))
    print(f"[parse] wrote {args.output_csv} ({len(rows)} rows)")
    print(f"[parse] wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
