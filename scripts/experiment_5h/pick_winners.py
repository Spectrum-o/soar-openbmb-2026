#!/usr/bin/env python3
"""scripts/experiment_5h/pick_winners.py

Look at eval_results.csv, find experiments from the current 5h session whose
acc_ori exceeds the baseline (J0) by >= threshold, return their env-var
overrides as a space-separated string for shell `eval`.

Usage:
    WINNERS=$(python3 scripts/experiment_5h/pick_winners.py \\
        --csv scripts/eval_results.csv \\
        --session 20260523_010000_full \\
        --baseline-exp A_J0_baseline \\
        --threshold 2.0)
    eval "env $WINNERS" bash scripts/local_eval.sh ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# Map experiment ID prefix → env var override (must match run_plan.sh)
EXP_ENV_MAP = {
    "D1_sym_false":     "GPTQ_SYM=False",
    "D2_damp_01":       "GPTQ_DAMPENING_FRAC=0.1",
    "D3_calib_512":     "NUM_CALIB=512",
    "D4_calib_len_16k": "MAX_CALIB_LEN=16384",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--session", required=True, help="Session tag to filter on")
    ap.add_argument("--baseline-exp", default="A_J0_baseline")
    ap.add_argument("--threshold", type=float, default=2.0, help="pp above baseline to qualify")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("", end="")  # empty winners
        return 0

    # Load CSV rows
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Filter to this session ONLY. Without this filter, a smoke run's D1
    # acc (often 0 or tiny because of 3-sample eval) could be picked as a
    # winner against a baseline from an earlier full run, and vice versa.
    # Session tag is encoded in log_path (scripts/logs/5h_{SESSION}/X.log).
    session_marker = f"5h_{args.session}"
    in_session = [
        r for r in rows
        if session_marker in (r.get("log_path") or "")
    ]
    if not in_session:
        # No rows from this session yet. Don't fall back to other sessions —
        # that's the bug we're fixing.
        print(f"[pick_winners] no rows for session {args.session}", file=sys.stderr)
        print("", end="")
        return 0
    rows = in_session

    # Find baseline acc_ori (latest occurrence WITHIN this session)
    baseline_acc: float | None = None
    for r in reversed(rows):
        if r.get("exp") == args.baseline_exp:
            try:
                baseline_acc = float(r["acc_ori"]) if r["acc_ori"] else float(r["acc"])
                break
            except (ValueError, KeyError):
                continue
    if baseline_acc is None:
        print(
            f"[pick_winners] no baseline ({args.baseline_exp}) found in session "
            f"{args.session} with parseable acc — cannot pick winners",
            file=sys.stderr,
        )
        print("", end="")
        return 0

    # Find Phase D experiments that beat baseline + threshold
    winners: list[str] = []
    for r in rows:
        exp = r.get("exp", "")
        if exp not in EXP_ENV_MAP:
            continue
        try:
            acc = float(r["acc_ori"]) if r["acc_ori"] else float(r["acc"])
        except (ValueError, KeyError):
            continue
        if acc >= baseline_acc + args.threshold:
            winners.append(EXP_ENV_MAP[exp])
            print(f"[pick_winners] {exp}: acc_ori={acc:.2f} (baseline {baseline_acc:.2f}, +{acc-baseline_acc:.2f}pp) → winner",
                  file=sys.stderr)
        else:
            delta = acc - baseline_acc
            sign = "+" if delta >= 0 else ""
            print(f"[pick_winners] {exp}: acc_ori={acc:.2f} ({sign}{delta:.2f}pp) — below threshold",
                  file=sys.stderr)

    print(" ".join(winners), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
