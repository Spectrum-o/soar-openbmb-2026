#!/usr/bin/env python3
"""Decide which variant to pack next based on the v23 platform log.

Reads a SOAR platform log file (produced by submitting a v23-or-later
tarball; saved by the operator from the platform UI or by `peek.sh`
locally), applies the V24_PLAN.md decision tree, and prints the
recommended next contingency variant + the ready-to-paste pack command.

This is the operational glue tying together:
  - tools/parse_quant_diagnostic.py (log parsing)
  - experiments/V24_PLAN.md (decision tree)
  - submission_gptqmodel_calib_w4a16_v24_*/  and v25_*/  (variant dirs)

Usage:
    python3 scripts/decide_next_variant.py \\
        --log /path/to/platform_v23.log \\
        --acc 0     # optional: explicit acc_ori from the SUCCESS line

If --acc is omitted, the script tries to extract it from the log.

Output: one-line recommendation + the pack command.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import parse_quant_diagnostic as pqd  # noqa: E402


# Known contingency variants. Keep in sync with submission_*/ dir names on disk.
VARIANTS = {
    "v24_perf": "submission_gptqmodel_calib_w4a16_v24",
    "v24_no_dtype_key": "submission_gptqmodel_calib_w4a16_v24_no_dtype_key",
    "v24_pin_transformers": "submission_gptqmodel_calib_w4a16_v24_pin_transformers",
    "v24_bits8": "submission_gptqmodel_calib_w4a16_v24_bits8",
    "v25_calib_multi_adaptive": "submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive",
    "v25_g64": "submission_gptqmodel_calib_w4a16_v25_g64",
}


def extract_acc_from_log(log_text: str) -> float | None:
    """Pull acc_ori from the platform SUCCESS line.

    Patterns observed in prior logs:
      "acc_ori": 0.0,
      "acc": 0.0,
      [SUCCESS] ... acc_ori=0.0
    """
    for pattern in (
        r'"acc_ori"\s*:\s*([\d.]+)',
        r'"acc"\s*:\s*([\d.]+)',
        r"acc_ori\s*=\s*([\d.]+)",
        r"Average Score\s+([\d.]+)%?",
    ):
        m = re.search(pattern, log_text)
        if m:
            return float(m.group(1))
    return None


def decide(parsed: pqd.ParsedLog, acc: float | None) -> tuple[str, str, str]:
    """Apply V24_PLAN.md decision tree.

    Returns (variant_key, one_line_recommendation, rationale).
    """
    q = parsed.qzeros_fix
    transformers_ver = parsed.versions.get("transformers", "")
    is_old_transformers = bool(
        transformers_ver and re.match(r"^4\.4[0-6]\.", transformers_ver)
    )

    # Branch 1: acc clearly recoverable (>= 30) → push perf
    if acc is not None and acc >= 30.0:
        return (
            "v24_perf",
            f"acc_ori={acc} >= 30 → H1+H4 fix worked. Push perf next.",
            "Apply chunked-prefill 65K (+83% throughput per local bench). "
            "Same artifact / quant scope / module set as v23 — perf flag only. "
            "Existing variant dir already pre-built.",
        )

    # Branch 2: acc = 0 + qzeros FATAL → extend fix_qzeros_for_marlin
    if q.fatal_message:
        return (
            "MANUAL_FIX",
            f"[qzeros-fix] FATAL on platform: {q.fatal_message[:80]}",
            "Read the FATAL message + sample hex values in the log "
            "(`grep -A2 'qzeros-fix.*sample' " + parsed.log_path + "`). "
            "Extend submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py's "
            "fix_qzeros_for_marlin to handle the new pattern. Then repack as v23b. "
            "No existing variant pre-fits this case.",
        )

    # Branch 3: acc = 0 + qzeros OK → branch by hypothesis cost
    if q.ended_ok and (acc is None or acc < 1.0):
        # H4 still viable if platform has < 4.47 transformers
        if is_old_transformers:
            return (
                "v24_pin_transformers",
                f"acc_ori=0 + qzeros OK + transformers={transformers_ver} (< 4.47) → "
                "platform silently ignores chat_template.jinja sidecar. "
                "Force-install transformers==4.57.1.",
                "v23's H4 fix overwrites tokenizer with base BF16, which has only "
                "inline chat_template. If that overwrite is succeeding on platform "
                "(unverifiable from current log), v24_pin_transformers is "
                "double-defense. If H4 isn't taking effect on platform for some "
                "reason, this directly bypasses the cause.",
            )
        # Else: H4 mechanism doesn't apply; try cheapest single-variable swap
        return (
            "v24_no_dtype_key",
            f"acc_ori=0 + qzeros OK + transformers={transformers_ver or '?'} (>= 4.47) → "
            "H4 doesn't apply here. Cheapest next hypothesis: drop dtype= config key.",
            "RTN-scalefix (platform acc=42) wrote ONLY torch_dtype; v17/v21/v22 "
            "wrote BOTH torch_dtype + dtype. The newer dtype= key may be silently "
            "miscomputing something. Single-line diff; symlinks share everything else.",
        )

    # Branch 4: pipeline crashed before serving (no qzeros-fix run at all)
    if not q.ended_ok and q.total_seen is None:
        if parsed.fatals:
            top_fatal = parsed.fatals[0][:120]
            return (
                "MANUAL_FIX",
                f"Crashed before fix_qzeros could run: {top_fatal}",
                "Pipeline died early. Most likely candidates: dtype mismatch "
                "(sed-patch missed a file), KeyError on dynamic skip (gptqmodel "
                "renamed a module), or flash_attn install failure. Read "
                "V23_PLATFORM_LOG_CHECKLIST.md Block 6/7. Patch v23 source in "
                "place; this is a bug fix, not a contingency variant.",
            )
        return (
            "MANUAL_FIX",
            "Log incomplete or unparseable; cannot decide.",
            "Open the log manually, look for FATAL / Error / Killed lines, "
            "and read V23_PLATFORM_LOG_CHECKLIST.md to walk through diagnosis.",
        )

    # Branch 5: 0 < acc < 30 → likely partial fix; push perf cautiously
    if acc is not None and 0 < acc < 30.0:
        return (
            "v25_calib_multi_adaptive",
            f"acc_ori={acc} (partial). H1+H4 partially worked but quality is poor.",
            "Multi-adaptive calibration windowing exposes the Hessian to wider "
            "context (not just tails). Combined with NUM_CALIB=300, ~420 effective "
            "samples cover the long-context patterns niah/cwe failures touched in "
            "v21 local analysis. ~25 min quant + 30 min eval.",
        )

    return (
        "UNCLEAR",
        "Could not decide from log alone.",
        "Inspect parse_quant_diagnostic output directly and consult V24_PLAN.md.",
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log", required=True, help="Path to v23 platform log")
    ap.add_argument("--acc", type=float, default=None,
                    help="Explicit acc_ori value (default: extract from log)")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"log not found: {log_path}", file=sys.stderr)
        return 2

    parsed = pqd.parse_log(log_path)
    acc = args.acc
    if acc is None:
        acc = extract_acc_from_log(log_path.read_text(errors="replace"))

    variant_key, recommendation, rationale = decide(parsed, acc)

    # Cap rationale and recommendation
    print("=" * 70)
    print(" decide_next_variant — V24_PLAN.md decision tree")
    print("=" * 70)
    print(f"  log:          {log_path}")
    print(f"  acc_ori:      {acc if acc is not None else '(not found in log)'}")
    print(f"  transformers: {parsed.versions.get('transformers', '?')}")
    print(f"  gptqmodel:    {parsed.versions.get('gptqmodel', '?')}")
    print(f"  python:       {parsed.versions.get('python', '?')}")
    print(f"  qzeros-fix:   {'OK' if parsed.qzeros_fix.ended_ok else 'NOT-OK'}"
          f" (patched {parsed.qzeros_fix.total_patched}/{parsed.qzeros_fix.total_seen})")
    print(f"  qzeros-fatal: {parsed.qzeros_fix.fatal_message or '(none)'}")
    print("-" * 70)
    print(f"  RECOMMENDATION: {variant_key}")
    print(f"  reason: {recommendation}")
    print()
    print(f"  rationale: {rationale}")
    print("=" * 70)

    if variant_key in VARIANTS:
        variant_dir = VARIANTS[variant_key]
        full_path = REPO_ROOT / variant_dir
        if full_path.is_dir():
            suffix = "_" + variant_key
            print()
            print("Ready-to-paste commands:")
            print()
            print(f"    python3 tools/pack_submission.py \\")
            print(f"        --variant {variant_dir} \\")
            print(f"        --check-only")
            print()
            print(f"    python3 tools/pack_submission.py \\")
            print(f"        --variant {variant_dir} \\")
            print(f"        --suffix {suffix} --output-dir .")
        else:
            print()
            print(f"⚠ variant dir not found at {full_path} — needs to be built first")
    elif variant_key == "MANUAL_FIX":
        print()
        print("Manual fix required — no pre-built variant for this branch.")
        print("See V24_PLAN.md Branch 2/4 (qzeros-FATAL or sglang-crash).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
