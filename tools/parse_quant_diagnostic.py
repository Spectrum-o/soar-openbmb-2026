#!/usr/bin/env python3
"""Parse a quantization-run log and (optionally) diff two logs.

Designed to consume the DIAGNOSTIC blocks added 2026-05-21 to:
  - submission_*/prepare_env.sh    ([prepare_env] lines)
  - submission_*/prepare_model.sh  ([prepare_model] DIAGNOSTIC lines)
  - quantize_gptqmodel_w4a16.py    ([qzeros-fix] SUMMARY block)
  - GPTQModel runtime              ([versions] lines via print_versions)

Goal: turn ad-hoc stdout into structured info so platform-vs-AutoDL
divergence is spotted by comparing two parsed dicts instead of eyeballing
log files. The platform run is opaque except through this log; this tool
is what makes the next platform submission diagnosable.

Usage:
    # Parse one log -> stdout summary
    python3 tools/parse_quant_diagnostic.py --input run.log

    # Parse one log -> markdown + json
    python3 tools/parse_quant_diagnostic.py \\
        --input run.log \\
        --output-md report.md \\
        --output-json report.json

    # Diff two logs (canonical use: platform vs local)
    python3 tools/parse_quant_diagnostic.py \\
        --input platform_run.log \\
        --compare local_run.log \\
        --output-md diff_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Structured schema
# ---------------------------------------------------------------------------
@dataclass
class CalibInfo:
    path: str | None = None
    size_bytes: int | None = None
    row_count: int | None = None
    first_row_preview: str | None = None
    num_calib: int | None = None
    max_calib_len: int | None = None
    window_mode: str | None = None
    chat_template_disabled: bool | None = None


@dataclass
class QzerosSample:
    shard: str
    key: str
    dtype: str
    unique_decimal: list[int] = field(default_factory=list)
    unique_hex: list[str] = field(default_factory=list)


@dataclass
class PostCheckInfo:
    shard: str
    key: str
    first_val_decimal: int
    first_val_hex: str
    is_good: bool


@dataclass
class QzerosFixInfo:
    inventory_files: list[tuple[str, int]] = field(default_factory=list)  # (name, bytes)
    scanned_file_count: int | None = None
    per_shard_patched: list[tuple[str, int]] = field(default_factory=list)
    total_seen: int | None = None
    total_patched: int | None = None
    total_already_good: int | None = None
    total_unknown_dtype: int | None = None
    total_unknown_values: int | None = None
    samples: list[QzerosSample] = field(default_factory=list)
    post_check: PostCheckInfo | None = None
    ended_ok: bool = False
    fatal_message: str | None = None


@dataclass
class SedPatchInfo:
    file_name: str
    applied: bool


@dataclass
class InstallInfo:
    skipped: bool = False
    forced_version: str | None = None
    currently_installed_at_time_of_decision: str | None = None


@dataclass
class ParsedLog:
    log_path: str
    versions: dict[str, str] = field(default_factory=dict)
    gptqmodel_install: InstallInfo = field(default_factory=InstallInfo)
    sed_patches: list[SedPatchInfo] = field(default_factory=list)
    calib: CalibInfo = field(default_factory=CalibInfo)
    output_dir_after_quant: list[tuple[str, int]] = field(default_factory=list)
    qzeros_fix: QzerosFixInfo = field(default_factory=QzerosFixInfo)
    sglang_server_args: str | None = None
    quant_save_path: str | None = None
    quant_completed: bool = False
    fatals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex patterns (anchored to known prefixes for safety)
# ---------------------------------------------------------------------------
RE_VERSIONS = re.compile(r"^\[versions\] ([\w\-]+)=(\S+)\s*$")
RE_INSTALL_SKIPPED = re.compile(
    r"^\[prepare_env\] gptqmodel (\S+) already installed; skipping"
)
RE_INSTALL_FORCED = re.compile(
    r"^\[prepare_env\] forcing gptqmodel==(\S+) \(currently: (.+?)\)"
)
RE_SED_PATCHED = re.compile(
    r"^\[prepare_env\] patched (\S+) for fp16 quantized run"
)
RE_SGLANG_ARGS = re.compile(r"^\[prepare_env\] SGLANG_SERVER_ARGS=(.+)$")
RE_CALIB_PATH = re.compile(
    r"^\[prepare_model\] DIAGNOSTIC: calib jsonl path=(.+)$"
)
RE_CALIB_SIZE = re.compile(
    r"^\[prepare_model\] DIAGNOSTIC: calib jsonl size=(\d+) bytes"
)
RE_CALIB_ROWS = re.compile(
    r"^\[prepare_model\] DIAGNOSTIC: calib jsonl row count=(\d+)"
)
RE_CALIB_FIRST_ROW = re.compile(
    r"^\[prepare_model\] DIAGNOSTIC: calib jsonl first row \(first 200 chars\):"
)
RE_NUM_CALIB = re.compile(r"--num-calib (\d+)")
RE_MAX_CALIB_LEN = re.compile(r"--max-calib-len (\d+)")
RE_WINDOW_MODE = re.compile(r"--calib-window-mode (\S+)")
RE_NO_CHAT_TPL = re.compile(r"--no-chat-template")
RE_QUANT_SAVING = re.compile(r"^\[quantize\] saving to (\S+)")
RE_QUANT_DONE = re.compile(r"^\[quantize\] done\.")
RE_QZEROS_INVENTORY_LINE = re.compile(
    r"^\[qzeros-fix\]\s+(\S+)\s+\(([\d,]+) bytes\)"
)
RE_QZEROS_INVENTORY_HEADER = re.compile(
    r"^\[qzeros-fix\] output_dir inventory"
)
RE_QZEROS_SCAN_COUNT = re.compile(
    r"^\[qzeros-fix\] scanning (\d+) \.safetensors weight file"
)
RE_QZEROS_PER_SHARD = re.compile(
    r"^\[qzeros-fix\] (\S+): patched (\d+) qzeros tensors"
)
# Legacy (pre-2026-05-21) format kept so the tool works on historical
# logs in scripts/logs/.
RE_QZEROS_PER_SHARD_LEGACY = re.compile(
    r"^\[qzeros-fix\] rewrote (\d+) qzeros tensors in (\S+)"
)
RE_QZEROS_TOTAL_LEGACY = re.compile(
    r"^\[qzeros-fix\] total (\d+) qzeros(?: tensors patched)? 0x77777777 -> 0x88888888"
)
RE_QZEROS_SUMMARY = re.compile(r"^\[qzeros-fix\] SUMMARY:")
RE_QZEROS_SEEN = re.compile(
    r"^\[qzeros-fix\]\s+qzeros tensors seen:\s+(\d+)"
)
RE_QZEROS_PATCHED = re.compile(
    r"^\[qzeros-fix\]\s+patched \(had 0x77777777\):\s+(\d+)"
)
RE_QZEROS_GOOD = re.compile(
    r"^\[qzeros-fix\]\s+already good \(0x88888888\):\s+(\d+)"
)
RE_QZEROS_UNK_DTYPE = re.compile(
    r"^\[qzeros-fix\]\s+unknown dtype skipped:\s+(\d+)"
)
RE_QZEROS_UNK_VALUE = re.compile(
    r"^\[qzeros-fix\]\s+unknown values not patched:\s+(\d+)"
)
RE_QZEROS_SAMPLE = re.compile(
    r"^\[qzeros-fix\]\s+sample: (\S+)::(\S+) dtype=(\S+) "
    r"unique\[:6\]=(\[.+?\]) hex=(\[.+?\])\s*$"
)
RE_QZEROS_POSTCHECK = re.compile(
    r"^\[qzeros-fix\] POST-CHECK (\S+)::(\S+) first_val=(-?\d+) \((0x[0-9a-fA-F]+)\)"
)
RE_QZEROS_OK = re.compile(r"^\[qzeros-fix\] OK\s*$")
RE_QZEROS_FATAL = re.compile(r"^\[qzeros-fix\] FATAL: (.+)$")
RE_OUTPUT_DIR_LINE = re.compile(
    r"^\[prepare_model\] DIAGNOSTIC: output dir contents after quantize"
)
# ls -la output style: "-rw-r--r-- 1 user grp SIZE date name"
RE_LS_LINE = re.compile(
    r"^[-d][\w-]+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s*$"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_list_literal(text: str) -> list[Any]:
    """Parse a Python-list-literal like '[1, 2, 3]' or "['0xAB', '0xCD']'."""
    try:
        return json.loads(text.replace("'", '"'))
    except json.JSONDecodeError:
        return []


def parse_log(path: Path) -> ParsedLog:
    parsed = ParsedLog(log_path=str(path))
    in_qzeros_inventory = False
    in_qzeros_summary = False
    in_output_dir_listing = False
    next_line_is_calib_first_row = False

    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # capture first calib row preview (on the line AFTER the header)
            if next_line_is_calib_first_row:
                parsed.calib.first_row_preview = line[:200]
                next_line_is_calib_first_row = False
                continue

            m = RE_VERSIONS.match(line)
            if m:
                parsed.versions[m.group(1)] = m.group(2)
                continue
            m = RE_INSTALL_SKIPPED.match(line)
            if m:
                parsed.gptqmodel_install.skipped = True
                parsed.versions.setdefault("gptqmodel", m.group(1))
                continue
            m = RE_INSTALL_FORCED.match(line)
            if m:
                parsed.gptqmodel_install.forced_version = m.group(1)
                parsed.gptqmodel_install.currently_installed_at_time_of_decision = m.group(2)
                continue
            m = RE_SED_PATCHED.match(line)
            if m:
                parsed.sed_patches.append(SedPatchInfo(m.group(1), True))
                continue
            m = RE_SGLANG_ARGS.match(line)
            if m:
                parsed.sglang_server_args = m.group(1).strip()
                continue
            m = RE_CALIB_PATH.match(line)
            if m:
                parsed.calib.path = m.group(1).strip()
                continue
            m = RE_CALIB_SIZE.match(line)
            if m:
                parsed.calib.size_bytes = int(m.group(1))
                continue
            m = RE_CALIB_ROWS.match(line)
            if m:
                parsed.calib.row_count = int(m.group(1))
                continue
            if RE_CALIB_FIRST_ROW.match(line):
                next_line_is_calib_first_row = True
                continue

            # The quantize command line has all the calib args; pick them off
            # regardless of which prefix it carries.
            for re_obj, attr, cast in (
                (RE_NUM_CALIB, "num_calib", int),
                (RE_MAX_CALIB_LEN, "max_calib_len", int),
                (RE_WINDOW_MODE, "window_mode", str),
            ):
                m = re_obj.search(line)
                if m and getattr(parsed.calib, attr) is None:
                    setattr(parsed.calib, attr, cast(m.group(1)))
            if RE_NO_CHAT_TPL.search(line) and parsed.calib.chat_template_disabled is None:
                parsed.calib.chat_template_disabled = True

            m = RE_QUANT_SAVING.match(line)
            if m:
                parsed.quant_save_path = m.group(1)
                continue
            if RE_QUANT_DONE.match(line):
                parsed.quant_completed = True
                continue

            # qzeros-fix block parsing
            if RE_QZEROS_INVENTORY_HEADER.match(line):
                in_qzeros_inventory = True
                in_qzeros_summary = False
                continue
            if RE_QZEROS_SUMMARY.match(line):
                in_qzeros_inventory = False
                in_qzeros_summary = True
                continue
            if in_qzeros_inventory:
                m = RE_QZEROS_INVENTORY_LINE.match(line)
                if m:
                    parsed.qzeros_fix.inventory_files.append(
                        (m.group(1), int(m.group(2).replace(",", "")))
                    )
                    continue
                # leave inventory mode when we hit scanning count
                if RE_QZEROS_SCAN_COUNT.match(line):
                    in_qzeros_inventory = False
            m = RE_QZEROS_SCAN_COUNT.match(line)
            if m:
                parsed.qzeros_fix.scanned_file_count = int(m.group(1))
                continue
            m = RE_QZEROS_PER_SHARD.match(line)
            if m:
                parsed.qzeros_fix.per_shard_patched.append(
                    (m.group(1), int(m.group(2)))
                )
                continue
            m = RE_QZEROS_PER_SHARD_LEGACY.match(line)
            if m:
                # Legacy order: (count, shard) — flip to (shard, count) for
                # consistency with the new format.
                parsed.qzeros_fix.per_shard_patched.append(
                    (m.group(2), int(m.group(1)))
                )
                continue
            m = RE_QZEROS_TOTAL_LEGACY.match(line)
            if m and parsed.qzeros_fix.total_patched is None:
                # Legacy logs only emit the total. Derive ended_ok from the
                # presence of this line + `[quantize] done.` later.
                parsed.qzeros_fix.total_patched = int(m.group(1))
                parsed.qzeros_fix.ended_ok = True
                continue
            for re_obj, attr in (
                (RE_QZEROS_SEEN, "total_seen"),
                (RE_QZEROS_PATCHED, "total_patched"),
                (RE_QZEROS_GOOD, "total_already_good"),
                (RE_QZEROS_UNK_DTYPE, "total_unknown_dtype"),
                (RE_QZEROS_UNK_VALUE, "total_unknown_values"),
            ):
                m = re_obj.match(line)
                if m:
                    setattr(parsed.qzeros_fix, attr, int(m.group(1)))
                    break
            m = RE_QZEROS_SAMPLE.match(line)
            if m:
                parsed.qzeros_fix.samples.append(QzerosSample(
                    shard=m.group(1),
                    key=m.group(2),
                    dtype=m.group(3),
                    unique_decimal=_parse_list_literal(m.group(4)),
                    unique_hex=_parse_list_literal(m.group(5)),
                ))
                continue
            m = RE_QZEROS_POSTCHECK.match(line)
            if m:
                first_val = int(m.group(3))
                parsed.qzeros_fix.post_check = PostCheckInfo(
                    shard=m.group(1),
                    key=m.group(2),
                    first_val_decimal=first_val,
                    first_val_hex=m.group(4),
                    is_good=(first_val == -2004318072),
                )
                continue
            if RE_QZEROS_OK.match(line):
                parsed.qzeros_fix.ended_ok = True
                continue
            m = RE_QZEROS_FATAL.match(line)
            if m:
                parsed.qzeros_fix.fatal_message = m.group(1).strip()
                parsed.fatals.append(f"[qzeros-fix] {m.group(1).strip()}")
                continue

            # Output dir listing (after [prepare_model] DIAGNOSTIC)
            if RE_OUTPUT_DIR_LINE.match(line):
                in_output_dir_listing = True
                continue
            if in_output_dir_listing:
                m = RE_LS_LINE.match(line)
                if m:
                    parsed.output_dir_after_quant.append(
                        (m.group(2), int(m.group(1)))
                    )
                    continue
                # heuristic: leave listing mode on next [prefix] marker
                if line.startswith("["):
                    in_output_dir_listing = False

            # Generic FATAL / WARN catch
            if "FATAL" in line and line.startswith("["):
                parsed.fatals.append(line)
            elif "WARN" in line and line.startswith("["):
                parsed.warnings.append(line)

    return parsed


# ---------------------------------------------------------------------------
# Reporting (single log)
# ---------------------------------------------------------------------------
def _fmt_versions(versions: dict[str, str]) -> str:
    if not versions:
        return "_(no `[versions]` block found)_"
    order = ["python", "torch", "transformers", "gptqmodel", "flash-attn",
             "flash-linear-attention", "tokenizers", "huggingface-hub",
             "accelerate", "ninja", "sglang"]
    lines = []
    for k in order:
        if k in versions:
            lines.append(f"| `{k}` | `{versions[k]}` |")
    for k, v in versions.items():
        if k not in order:
            lines.append(f"| `{k}` | `{v}` |")
    return "| Package | Version |\n|---|---|\n" + "\n".join(lines)


def render_report(p: ParsedLog) -> str:
    lines: list[str] = []
    lines.append(f"# Quantization log: `{p.log_path}`")
    lines.append("")
    lines.append("## Environment / install")
    lines.append("")
    lines.append(_fmt_versions(p.versions))
    lines.append("")
    if p.gptqmodel_install.forced_version:
        lines.append(
            f"- gptqmodel install: **forced to** `{p.gptqmodel_install.forced_version}` "
            f"(was: `{p.gptqmodel_install.currently_installed_at_time_of_decision}`)"
        )
    elif p.gptqmodel_install.skipped:
        lines.append("- gptqmodel install: **skipped** (already installed)")
    else:
        lines.append("- gptqmodel install: (no signal)")
    if p.sed_patches:
        lines.append(f"- sed-patches applied: {', '.join(s.file_name for s in p.sed_patches)}")
    if p.sglang_server_args:
        lines.append(f"- SGLANG_SERVER_ARGS: `{p.sglang_server_args}`")
    lines.append("")

    lines.append("## Calibration")
    lines.append("")
    c = p.calib
    lines.append(f"- path: `{c.path}`")
    lines.append(f"- size: {c.size_bytes:,} bytes" if c.size_bytes else "- size: (not parsed)")
    lines.append(f"- rows: {c.row_count}")
    lines.append(f"- num_calib (CLI): {c.num_calib}")
    lines.append(f"- max_calib_len (CLI): {c.max_calib_len}")
    lines.append(f"- window_mode (CLI): `{c.window_mode}`")
    lines.append(f"- chat_template_disabled (CLI flag): {c.chat_template_disabled}")
    if c.first_row_preview:
        lines.append("")
        lines.append("First calib row (first 200 chars):")
        lines.append("```")
        lines.append(c.first_row_preview)
        lines.append("```")
    lines.append("")

    lines.append("## Quantize save")
    lines.append("")
    lines.append(f"- save path: `{p.quant_save_path}`")
    lines.append(f"- completed (`[quantize] done.`): {p.quant_completed}")
    if p.output_dir_after_quant:
        lines.append("")
        lines.append("Output dir after quantize:")
        lines.append("")
        lines.append("| File | Size (bytes) |")
        lines.append("|---|---|")
        for name, size in p.output_dir_after_quant:
            lines.append(f"| `{name}` | {size:,} |")
    lines.append("")

    lines.append("## qzeros-fix")
    lines.append("")
    q = p.qzeros_fix
    lines.append(f"- scanned safetensors files: {q.scanned_file_count}")
    if q.inventory_files:
        lines.append("- inventory snapshot:")
        for name, size in q.inventory_files:
            lines.append(f"  - `{name}` ({size:,} bytes)")
    if q.per_shard_patched:
        lines.append("- patched per shard:")
        for name, n in q.per_shard_patched:
            lines.append(f"  - `{name}`: {n} qzeros tensors")
    lines.append(f"- summary: seen={q.total_seen}, patched={q.total_patched}, "
                 f"already_good={q.total_already_good}, "
                 f"unknown_dtype={q.total_unknown_dtype}, "
                 f"unknown_values={q.total_unknown_values}")
    if q.samples:
        lines.append("")
        lines.append("Sample qzeros tensors observed:")
        lines.append("")
        lines.append("| shard | key | dtype | unique hex |")
        lines.append("|---|---|---|---|")
        for s in q.samples:
            lines.append(f"| `{s.shard}` | `{s.key}` | `{s.dtype}` | `{s.unique_hex}` |")
    if q.post_check:
        pc = q.post_check
        verdict = "OK" if pc.is_good else "BAD"
        lines.append("")
        lines.append(f"- POST-CHECK: `{pc.shard}::{pc.key}` first_val="
                     f"{pc.first_val_decimal} ({pc.first_val_hex}) → **{verdict}**")
    if q.fatal_message:
        lines.append("")
        lines.append(f"- **FATAL**: `{q.fatal_message}`")
    lines.append(f"- ended_ok flag: {q.ended_ok}")
    lines.append("")

    if p.fatals:
        lines.append("## FATAL lines (any prefix)")
        lines.append("")
        for f in p.fatals[:20]:
            lines.append(f"- `{f}`")
        lines.append("")

    if p.warnings:
        lines.append(f"## WARN lines (first {min(len(p.warnings), 20)})")
        lines.append("")
        for w in p.warnings[:20]:
            lines.append(f"- `{w}`")
        lines.append("")

    return "\n".join(lines)


def one_line_summary(p: ParsedLog) -> str:
    q = p.qzeros_fix
    fix_status = "OK" if q.ended_ok else ("FATAL" if q.fatal_message else "(unfinished/unknown)")
    return (
        f"{Path(p.log_path).name}: "
        f"gptqmodel={p.versions.get('gptqmodel', '?')} "
        f"transformers={p.versions.get('transformers', '?')} "
        f"calib_rows={p.calib.row_count} "
        f"safetensors_files={q.scanned_file_count} "
        f"qzeros_patched={q.total_patched}/{q.total_seen} "
        f"fix={fix_status} "
        f"quant_done={p.quant_completed}"
    )


# ---------------------------------------------------------------------------
# Diffing two logs
# ---------------------------------------------------------------------------
def _versions_diff(a: dict[str, str], b: dict[str, str]) -> list[tuple[str, str, str]]:
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        va, vb = a.get(k, "(missing)"), b.get(k, "(missing)")
        if va != vb:
            out.append((k, va, vb))
    return out


def _qzeros_diff_summary(a: QzerosFixInfo, b: QzerosFixInfo) -> list[tuple[str, str, str]]:
    rows = []
    fields = [
        ("scanned_file_count", str(a.scanned_file_count), str(b.scanned_file_count)),
        ("total_seen", str(a.total_seen), str(b.total_seen)),
        ("total_patched", str(a.total_patched), str(b.total_patched)),
        ("total_already_good", str(a.total_already_good), str(b.total_already_good)),
        ("total_unknown_dtype", str(a.total_unknown_dtype), str(b.total_unknown_dtype)),
        ("total_unknown_values", str(a.total_unknown_values), str(b.total_unknown_values)),
        ("ended_ok", str(a.ended_ok), str(b.ended_ok)),
        ("fatal_message", a.fatal_message or "(none)", b.fatal_message or "(none)"),
    ]
    for name, va, vb in fields:
        if va != vb:
            rows.append((name, va, vb))
    return rows


def render_diff_report(a: ParsedLog, b: ParsedLog) -> str:
    lines: list[str] = []
    lines.append(f"# Quant log diff: `{a.log_path}` (A) vs `{b.log_path}` (B)")
    lines.append("")
    lines.append(f"- A: {one_line_summary(a)}")
    lines.append(f"- B: {one_line_summary(b)}")
    lines.append("")

    lines.append("## Version differences")
    lines.append("")
    vdiff = _versions_diff(a.versions, b.versions)
    if vdiff:
        lines.append("| Package | A | B |")
        lines.append("|---|---|---|")
        for k, va, vb in vdiff:
            lines.append(f"| `{k}` | `{va}` | `{vb}` |")
    else:
        lines.append("_(no version differences detected — both logs report identical or missing packages)_")
    lines.append("")

    lines.append("## Calibration differences")
    lines.append("")
    cfields = [
        ("path", a.calib.path, b.calib.path),
        ("size_bytes", a.calib.size_bytes, b.calib.size_bytes),
        ("row_count", a.calib.row_count, b.calib.row_count),
        ("num_calib", a.calib.num_calib, b.calib.num_calib),
        ("max_calib_len", a.calib.max_calib_len, b.calib.max_calib_len),
        ("window_mode", a.calib.window_mode, b.calib.window_mode),
        ("chat_template_disabled", a.calib.chat_template_disabled,
         b.calib.chat_template_disabled),
    ]
    cdiff = [(n, va, vb) for n, va, vb in cfields if va != vb]
    if cdiff:
        lines.append("| Field | A | B |")
        lines.append("|---|---|---|")
        for n, va, vb in cdiff:
            lines.append(f"| `{n}` | `{va}` | `{vb}` |")
    else:
        lines.append("_(calibration setup matches)_")
    lines.append("")

    lines.append("## Output-dir layout differences")
    lines.append("")
    files_a = {n: s for n, s in a.output_dir_after_quant}
    files_b = {n: s for n, s in b.output_dir_after_quant}
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    both = sorted(set(files_a) & set(files_b))
    if only_a or only_b or any(files_a[n] != files_b[n] for n in both):
        lines.append("| File | Size in A | Size in B |")
        lines.append("|---|---|---|")
        for n in only_a:
            lines.append(f"| `{n}` | {files_a[n]:,} | _(absent)_ |")
        for n in only_b:
            lines.append(f"| `{n}` | _(absent)_ | {files_b[n]:,} |")
        for n in both:
            if files_a[n] != files_b[n]:
                lines.append(f"| `{n}` | {files_a[n]:,} | {files_b[n]:,} |")
    elif a.output_dir_after_quant or b.output_dir_after_quant:
        lines.append("_(output dir contents match)_")
    else:
        lines.append("_(no output-dir DIAGNOSTIC block in either log)_")
    lines.append("")

    lines.append("## qzeros-fix differences")
    lines.append("")
    qdiff = _qzeros_diff_summary(a.qzeros_fix, b.qzeros_fix)
    if qdiff:
        lines.append("| Field | A | B |")
        lines.append("|---|---|---|")
        for n, va, vb in qdiff:
            lines.append(f"| `{n}` | `{va}` | `{vb}` |")
    else:
        lines.append("_(qzeros-fix counters match)_")
    if a.qzeros_fix.samples or b.qzeros_fix.samples:
        lines.append("")
        lines.append("### Sample qzeros unique values")
        lines.append("")
        lines.append("**A samples:**")
        for s in a.qzeros_fix.samples:
            lines.append(f"- `{s.shard}::{s.key}` dtype=`{s.dtype}` hex={s.unique_hex}")
        lines.append("")
        lines.append("**B samples:**")
        for s in b.qzeros_fix.samples:
            lines.append(f"- `{s.shard}::{s.key}` dtype=`{s.dtype}` hex={s.unique_hex}")
    lines.append("")

    lines.append("## Verdict heuristic")
    lines.append("")
    if a.qzeros_fix.fatal_message and not b.qzeros_fix.fatal_message:
        lines.append(f"- A raised qzeros-fix FATAL; B did not. **A is the broken side.**")
    elif b.qzeros_fix.fatal_message and not a.qzeros_fix.fatal_message:
        lines.append(f"- B raised qzeros-fix FATAL; A did not. **B is the broken side.**")
    elif a.qzeros_fix.total_patched != b.qzeros_fix.total_patched:
        lines.append(f"- qzeros patched count diverges: A={a.qzeros_fix.total_patched} "
                     f"B={b.qzeros_fix.total_patched}. The side with fewer patches "
                     f"may have written qzeros in a layout fix didn't recognize.")
    elif a.qzeros_fix.scanned_file_count != b.qzeros_fix.scanned_file_count:
        lines.append(f"- safetensors file count diverges: A={a.qzeros_fix.scanned_file_count} "
                     f"B={b.qzeros_fix.scanned_file_count}. Possibly single-file vs "
                     f"sharded output -- check 'Output-dir layout differences' above.")
    elif vdiff:
        lines.append(f"- {len(vdiff)} package version difference(s). gptqmodel version "
                     f"drift is the prime suspect; cross-check sample qzeros hex values.")
    else:
        lines.append("_(no strong heuristic match -- inspect each section above manually)_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Path to a quantization log file")
    ap.add_argument("--compare", default=None,
                    help="Optional second log to diff against --input")
    ap.add_argument("--output-md", default=None, help="Write markdown report here")
    ap.add_argument("--output-json", default=None,
                    help="Write structured json here (always the --input log; "
                         "if --compare is set, writes both as a list)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"input not found: {in_path}", file=sys.stderr)
        return 2

    p_a = parse_log(in_path)

    if args.compare:
        cmp_path = Path(args.compare)
        if not cmp_path.is_file():
            print(f"--compare path not found: {cmp_path}", file=sys.stderr)
            return 2
        p_b = parse_log(cmp_path)
        report = render_diff_report(p_a, p_b)
        print(one_line_summary(p_a))
        print(one_line_summary(p_b))
    else:
        p_b = None
        report = render_report(p_a)
        print(one_line_summary(p_a))

    if args.output_md:
        Path(args.output_md).write_text(report)
        print(f"[parse-quant-diagnostic] wrote {args.output_md}", file=sys.stderr)
    else:
        print()
        print(report)

    if args.output_json:
        if p_b is None:
            payload = _to_jsonable(p_a)
        else:
            payload = [_to_jsonable(p_a), _to_jsonable(p_b)]
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"[parse-quant-diagnostic] wrote {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
