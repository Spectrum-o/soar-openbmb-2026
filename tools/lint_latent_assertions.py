#!/usr/bin/env python3
"""Latent-assertion lint for submission_*/quantize_*.py scripts.

Pattern detected: assertions, POST-CHECK loops, or sanity blocks that
hard-code `shards[0]` (or any other fixed index into a shard / safetensors
list) before deciding "verified". The v23 platform submission FAILED at
prepare_model because `fix_qzeros_for_marlin`'s POST-CHECK iterated
`shards[0]` blindly — but for MLP-only artifacts shard 0 contains
embedding + lm_head with NO .qzeros tensors at all → for-loop completed,
`verified` stayed False, spurious FATAL fired. The qzeros patch ITSELF had
succeeded (96/96 tensors). The 5h slot tested the assertion code, not the
hypothesis. Fix in commit 551826514.

This lint catches the same shape in any NEW code:
  - `shards[0]` / `safetensor_files[0]` / `*.safetensors")[0]` etc.
  - in proximity (within 30 lines) to "FATAL" / "verified" / "POST-CHECK"
    / "raise" / "sys.exit"

We deliberately keep this narrow:
  - Only run against submission_*/quantize_*.py (the artifact-producing
    scripts; that's the only place a wrong assertion costs a 5h slot).
  - Whitelist comments — false positives in docstrings shouldn't fail CI.
  - Whitelist `for shard in shards[0:]` (iteration, not indexing).

Usage:
    python3 tools/lint_latent_assertions.py                    # repo scan
    python3 tools/lint_latent_assertions.py path/to/file.py    # one file
    python3 tools/lint_latent_assertions.py --json             # machine-readable

Exit codes:
  0 — clean
  1 — at least one suspicious pattern found
  2 — CLI / IO error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Patterns that index a list of shards/files by a hardcoded integer.
# Examples we want to catch:
#   shards[0]
#   safetensor_files[0]
#   shard_paths[1]
#   tensor_files[0]
# Slice-with-step like shards[0:] is iteration, not indexing — NOT flagged.
_HARDCODED_SHARD_INDEX = re.compile(
    r"\b(\w*shard\w*|\w*safetensor\w*|\w*tensor_files\w*)\[(\d+)\](?!\s*:)"
)

# Tokens that turn an indexing op into a "this fires a critical sanity check".
_CHECK_TOKENS = ("FATAL", "verified", "POST-CHECK", "POST_CHECK", "raise ", "sys.exit", "AssertionError", "assert ")

# Window: an index op within this many lines of a check token is suspicious.
_WINDOW = 30


@dataclass
class Finding:
    path: str
    line: int
    line_text: str
    match: str  # e.g. "shards[0]"
    why: str  # human explanation


def _strip_comments_and_docstrings(src: str) -> list[tuple[int, str]]:
    """Return (lineno, text) for source lines with comments stripped.

    Docstrings are blanked out as a coarse pass — we only need approximate
    suppression so docstring examples don't trip the lint. We're not
    building a real Python tokenizer here.
    """
    out: list[tuple[int, str]] = []
    in_doc: str | None = None
    for i, raw in enumerate(src.splitlines(), start=1):
        line = raw
        if in_doc is not None:
            # inside a triple-quoted string; check for closer
            if in_doc in line:
                line = line.split(in_doc, 1)[1]
                in_doc = None
                # fall through to comment-strip
            else:
                out.append((i, ""))
                continue
        # detect open of triple-quote
        for q in ('"""', "'''"):
            if q in line:
                # opens AND closes on same line?
                rest = line.split(q, 1)[1]
                if q in rest:
                    line = line.split(q, 1)[0] + rest.split(q, 1)[1]
                else:
                    in_doc = q
                    line = line.split(q, 1)[0]
                    break
        # strip `#` comments
        if "#" in line:
            line = line.split("#", 1)[0]
        out.append((i, line))
    return out


def lint_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as e:
        return [Finding(str(path), 0, "", "", f"could not read file: {e}")]
    lines = _strip_comments_and_docstrings(src)

    # Pre-index check tokens by line number so the window scan is O(N).
    check_lines: set[int] = set()
    for lineno, text in lines:
        if any(tok in text for tok in _CHECK_TOKENS):
            check_lines.add(lineno)

    for lineno, text in lines:
        m = _HARDCODED_SHARD_INDEX.search(text)
        if not m:
            continue
        # Check window: any check token within +/- _WINDOW lines?
        near = [
            c for c in check_lines
            if abs(c - lineno) <= _WINDOW and c != lineno
        ]
        if not near:
            # Indexing exists but no critical assertion nearby — not the
            # v23 bug shape. (Could still be a code smell; lint only fires
            # on the dangerous flavor.)
            continue
        findings.append(Finding(
            path=str(path),
            line=lineno,
            line_text=text.rstrip(),
            match=m.group(0),
            why=(
                f"hardcoded shard index {m.group(0)!r} within {_WINDOW} lines "
                f"of a critical assertion (near line(s) {sorted(near)[:3]}). "
                f"v23 platform=0 was caused by this exact shape: iterating "
                f"shards[0] then asserting `verified=True` — but the chosen "
                f"shard had no qzeros tensors, so verified stayed False and "
                f"FATAL fired on a perfectly-good artifact. Fix: iterate the "
                f"list of shards you ACTUALLY wrote to, not a fixed index."
            ),
        ))
    return findings


def find_target_files(root: Path) -> list[Path]:
    """Default scan target: every submission_*/quantize_*.py."""
    out: list[Path] = []
    for sub in sorted(root.glob("submission_*")):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("quantize_*.py")):
            # Skip symlinks pointing into other submission dirs — we'll
            # cover their target file once via its canonical location.
            if f.is_symlink():
                continue
            out.append(f)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths", nargs="*",
        help="files or directories to lint (default: submission_*/quantize_*.py)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="machine-readable output (one Finding dict per line)",
    )
    parser.add_argument(
        "--root", default=".",
        help="repo root for default scan (default: .)",
    )
    args = parser.parse_args()

    if args.paths:
        targets: list[Path] = []
        for p in args.paths:
            pp = Path(p)
            if pp.is_dir():
                targets.extend(sorted(pp.rglob("quantize_*.py")))
            else:
                targets.append(pp)
    else:
        targets = find_target_files(Path(args.root))

    if not targets:
        print("[lint] no files to scan", file=sys.stderr)
        return 0

    all_findings: list[Finding] = []
    for t in targets:
        if t.is_symlink():
            continue
        all_findings.extend(lint_file(t))

    if args.json:
        for f in all_findings:
            print(json.dumps(f.__dict__))
        return 0 if not all_findings else 1

    for t in targets:
        print(f"[scan] {t}")
    print()
    if not all_findings:
        print(f"[lint] OK — scanned {len(targets)} file(s), no v23-shape "
              f"assertions detected")
        return 0

    for f in all_findings:
        print(f"[FOUND] {f.path}:{f.line}")
        print(f"        {f.line_text}")
        print(f"        ^ {f.match}")
        print(f"        {f.why}")
        print()
    print(f"[lint] {len(all_findings)} finding(s) across {len(targets)} file(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
