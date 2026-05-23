#!/usr/bin/env python3
"""scripts/experiment_5h/parse_results.py

Parse one experiment's log file, extract acc + per-task numbers, append a row
to eval_results.csv.

Per-task numbers come from SOAR Toolkit's predictions.jsonl (eval_model.py
writes one row per sample with fields: task, score, gold, prediction). We
locate the predictions.jsonl by mtime — the one most recently modified in
the outputs/ directory (or any directory containing predictions.jsonl) that
falls between experiment start and end.

Usage:
    python3 scripts/experiment_5h/parse_results.py \\
        --log scripts/logs/5h_<session>/A_J0_baseline.log \\
        --exp A_J0_baseline \\
        --desc "current 1849 recipe, no env overrides" \\
        --csv scripts/eval_results.csv

If predictions.jsonl can't be found, per-task fields stay empty but the
aggregate acc is still extracted from the log.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import statistics
import sys
from pathlib import Path


HEADER = [
    "timestamp",
    "exp",
    "desc",
    "acc",
    "acc_ori",
    "niah",
    "cwe",
    "fwe",
    "qa",
    "mcq",
    "duration_s",
    "log_path",
    "predictions_path",
]


# ---------------------------------------------------------------------------
# Aggregate-level extraction from eval log
# ---------------------------------------------------------------------------
def parse_aggregate(log_path: Path) -> dict[str, str]:
    """Extract acc / acc_ori / duration from eval log. Tolerant of multiple
    SOAR Toolkit versions:
      - newer: "ori_accuracy: 46.89", "overall_accuracy: 58.61"
      - older: "Average Score: 63.33%"
      - SUCCESS JSON output: {"acc": 58.61, "acc_ori": 46.89, ...}
    """
    out: dict[str, str] = {k: "" for k in ("acc", "acc_ori", "duration_s")}
    if not log_path.exists():
        return out

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out

    # acc / acc_ori — try canonical names first.
    # Patterns are tried in priority order:
    #   1. JSON-style "acc": N — most authoritative (final summary block)
    #   2. line-anchored "acc = N" or "acc: N" — common in CLI output
    #   3. inline "acc=N" — fallback (also matches per-sample log lines, so
    #      we LAST-match these to bias toward the summary at end-of-log)
    # Within (1) and (2) we take the LAST match because summary blocks
    # are typically printed once at end-of-log. (3) is fall-through only.
    for key in ("acc_ori", "acc"):
        # Build pattern list with priority. Order matters.
        prioritized: list[tuple[str, bool]] = [
            # (pattern, take_first_match) — JSON-style is high-confidence
            (rf"\"{key}\"\s*:\s*([0-9.]+)", False),
            # Line-anchored — e.g. "acc_ori = 46.89" on its own line
            (rf"(?m)^\s*{key}\s*[=:]\s*([0-9.]+)", False),
            # Inline anywhere — last-resort, biased to end-of-log
            (rf"\b{key}\s*[=:]\s*([0-9.]+)", False),
        ]
        for pat, _ in prioritized:
            matches = re.findall(pat, text, re.IGNORECASE)
            if not matches:
                continue
            # Filter to plausible accuracy values (0..100 or 0..1). Per-sample
            # logs sometimes print raw token counts as acc=N; the range gate
            # drops those.
            plausible: list[float] = []
            for raw in matches:
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if 0.0 <= v <= 100.0:
                    plausible.append(v)
            if not plausible:
                continue
            val = plausible[-1]  # last plausible match (closest to summary)
            if val <= 1.0:  # fraction → percent
                val *= 100
            out[key] = f"{val:.2f}"
            break

    # acc_ori variants in older SOAR scripts
    if not out["acc_ori"]:
        m = re.search(r"\bori_accuracy\b[\s=:]*([0-9.]+)", text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val < 1.0:
                val *= 100
            out["acc_ori"] = f"{val:.2f}"

    if not out["acc"]:
        m = re.search(r"\boverall_accuracy\b[\s=:]*([0-9.]+)", text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val < 1.0:
                val *= 100
            out["acc"] = f"{val:.2f}"

    # Fallback: "Average Score: 63.33%"
    if not out["acc"] and not out["acc_ori"]:
        m = re.search(r"Average Score[\s=:]+([0-9.]+)", text, re.IGNORECASE)
        if m:
            out["acc"] = f"{float(m.group(1)):.2f}"

    # Duration: look for "duration_s = N" or "Time elapsed: Ns" etc
    for pat in (
        r"duration[_a-z]*[\s=:]+([0-9.]+)",
        r"elapsed[\s=:]+([0-9.]+)",
        r"wall[\s=:]+([0-9.]+)",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out["duration_s"] = f"{float(m.group(1)):.0f}"
            break

    return out


# ---------------------------------------------------------------------------
# Per-task extraction from predictions.jsonl
# ---------------------------------------------------------------------------
def find_predictions_jsonl(repo_root: Path, log_path: Path) -> Path | None:
    """Find the predictions.jsonl associated with this experiment.

    Strategy (strictest first):
      1. The log itself often mentions an explicit output dir; grep for it.
      2. Otherwise scope by mtime: predictions.jsonl whose mtime falls between
         the experiment's log start (ctime) and end (mtime). Across a 5h
         pipeline running 9 experiments, this is the only way to avoid
         cross-contaminating one experiment's per-task scores with the
         predictions.jsonl from a DIFFERENT experiment that also ran in the
         last 12 hours. The pre-fix heuristic (12h backward window from log
         mtime) was almost always returning the SAME predictions.jsonl for
         every phase.
      3. Last resort: most recent predictions.jsonl globally.
    """
    candidates: list[Path] = []
    for root in (repo_root / "outputs", repo_root.parent / "outputs", repo_root / "scripts" / "logs"):
        if not root.exists():
            continue
        candidates.extend(root.rglob("predictions.jsonl"))

    if not candidates:
        return None

    # Strategy 1: log file explicitly mentions an output dir
    if log_path.exists():
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            log_text = ""
        # SOAR Toolkit logs typically print "Output dir: <path>" or similar
        import re as _re
        for m in _re.finditer(r"(?:output[_\s-]*dir|saving to|writing to)\s*[:=]?\s*(\S+)",
                              log_text, _re.IGNORECASE):
            mentioned = Path(m.group(1).strip().rstrip(":,;"))
            if not mentioned.is_absolute():
                mentioned = repo_root / mentioned
            candidate = mentioned / "predictions.jsonl"
            if candidate.exists() and candidate in candidates:
                return candidate

    # Strategy 2: time-scoped to the experiment's log lifetime
    if log_path.exists():
        try:
            log_stat = log_path.stat()
            # ctime ≈ when run_exp opened the log (start of experiment)
            log_start = min(log_stat.st_ctime, log_stat.st_mtime)
            log_end = log_stat.st_mtime
        except OSError:
            log_start = log_end = 0.0

        if log_start > 0:
            scoped: list[Path] = []
            for c in candidates:
                try:
                    cm = c.stat().st_mtime
                except OSError:
                    continue
                # predictions.jsonl was written during this experiment's run.
                # Allow a 60-second slop on each side for clock skew / writes
                # that happen just after the eval log is closed.
                if (log_start - 60) <= cm <= (log_end + 60):
                    scoped.append(c)
            if scoped:
                # Pick the one closest to log_end (most likely the FINAL save)
                scoped.sort(key=lambda p: abs(p.stat().st_mtime - log_end))
                return scoped[0]

    # Strategy 3: most recent globally — last-resort fallback. Warn-quality
    # match; the caller should treat per-task numbers as suggestive.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_per_task(predictions_path: Path) -> dict[str, str]:
    """Compute mean per-task score from predictions.jsonl.

    Returns dict with niah/cwe/fwe/qa/mcq as keys (formatted as "XX.XX" pct).
    Missing tasks → empty string.
    """
    out: dict[str, str] = {k: "" for k in ("niah", "cwe", "fwe", "qa", "mcq")}
    if not predictions_path or not predictions_path.exists():
        return out

    scores_by_task: dict[str, list[float]] = {}
    try:
        with predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task = (row.get("task") or "").lower()
                score = row.get("score")
                if score is None:
                    continue
                try:
                    s = float(score)
                except (TypeError, ValueError):
                    continue
                scores_by_task.setdefault(task, []).append(s)
    except Exception as e:
        print(f"[parse_per_task] error reading {predictions_path}: {e}", file=sys.stderr)
        return out

    for task, scores in scores_by_task.items():
        if not scores:
            continue
        mean_pct = 100.0 * statistics.mean(scores)
        out[task] = f"{mean_pct:.2f}"

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--predictions", default="",
                    help="Optional explicit path to predictions.jsonl; auto-detect if omitted")
    args = ap.parse_args()

    log_path = Path(args.log)
    csv_path = Path(args.csv)
    repo_root = Path(__file__).resolve().parents[2]

    aggregate = parse_aggregate(log_path)

    # Find + parse predictions.jsonl
    predictions_path: Path | None = None
    if args.predictions:
        pp = Path(args.predictions)
        if pp.exists():
            predictions_path = pp
    if predictions_path is None:
        predictions_path = find_predictions_jsonl(repo_root, log_path)

    per_task = parse_per_task(predictions_path) if predictions_path else {k: "" for k in ("niah", "cwe", "fwe", "qa", "mcq")}

    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp": args.exp,
        "desc": args.desc,
        "log_path": str(log_path.relative_to(Path.cwd())) if log_path.is_absolute() else str(log_path),
        "predictions_path": str(predictions_path) if predictions_path else "",
        **aggregate,
        **per_task,
    }

    # If CSV exists with old header (without predictions_path), upgrade it
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    if not is_new:
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            existing_cols = first.split(",")
            # If the existing CSV doesn't have predictions_path, write rows
            # in the OLD layout to preserve compat (skip the new col)
            if "predictions_path" not in existing_cols:
                row_to_write = {k: row.get(k, "") for k in existing_cols}
                with csv_path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=existing_cols)
                    writer.writerow(row_to_write)
                _print_summary(args.exp, aggregate, per_task, predictions_path)
                return 0
        except Exception:
            pass

    # Either new CSV (write our HEADER) or existing matches our HEADER
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in HEADER})

    _print_summary(args.exp, aggregate, per_task, predictions_path)
    return 0


def _print_summary(exp: str, aggregate: dict[str, str], per_task: dict[str, str], predictions_path: Path | None) -> None:
    """Echo what was extracted (for the master.log)."""
    parts = []
    for k in ("acc", "acc_ori", "duration_s"):
        if aggregate.get(k):
            parts.append(f"{k}={aggregate[k]}")
    for k in ("niah", "cwe", "fwe", "qa", "mcq"):
        if per_task.get(k):
            parts.append(f"{k}={per_task[k]}")
    if not parts:
        parts.append("(no metrics extracted)")
    summary = ", ".join(parts)
    print(f"[parse_results] {exp}: {summary}", file=sys.stderr)
    if predictions_path:
        print(f"[parse_results] predictions: {predictions_path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
