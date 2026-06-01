#!/usr/bin/env python3
"""Summarize FP8 KV scalar-scale waste from kv_calibrate.py reports.

`scripts/kv_calibrate.py` writes scalar SGLang-compatible k/v scales, but it can
also record optional per-head max-abs diagnostics:

    layers.<layer>.k_per_head.median_head_utilization_under_scalar
    layers.<layer>.v_per_head.median_head_utilization_under_scalar

Those values quantify how much of the scalar FP8 range the median head uses
when one layer-wide scale is chosen by the worst outlier head.  Lower is worse:

    1.00 -> median head uses the same range as the worst head
    0.25 -> median head effectively uses only 25% of the available FP8 codes

This tool ranks the worst layers/projections and gives a concrete go/no-go
signal for runtime per-head/per-block scale work.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SEVERE_UTIL = 0.50
DEFAULT_WARN_UTIL = 0.70


@dataclass(frozen=True)
class ProjectionWaste:
    layer: int
    projection: str
    median_util: float
    scalar_over_median: float | None
    max_over_min: float | None
    scalar_scale: float | None
    n_samples: int | None


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_projection_waste(report: dict[str, Any]) -> list[ProjectionWaste]:
    layers = report.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("report has no layers object")

    rows: list[ProjectionWaste] = []
    for layer_key, layer in layers.items():
        if not isinstance(layer, dict):
            continue
        try:
            layer_id = int(layer_key)
        except ValueError:
            continue

        for projection in ("k", "v"):
            per_head = layer.get(f"{projection}_per_head")
            if not isinstance(per_head, dict):
                continue
            median_util = _to_float(per_head.get("median_head_utilization_under_scalar"))
            if median_util is None:
                continue
            rows.append(
                ProjectionWaste(
                    layer=layer_id,
                    projection=projection,
                    median_util=median_util,
                    scalar_over_median=_to_float(
                        per_head.get("scalar_over_median_head_scale")
                    ),
                    max_over_min=_to_float(per_head.get("max_over_min_nonzero_abs")),
                    scalar_scale=_to_float(layer.get(f"{projection}_scale")),
                    n_samples=_to_int(layer.get("n_samples")),
                )
            )

    rows.sort(key=lambda row: (row.median_util, row.layer, row.projection))
    return rows


def classify(rows: list[ProjectionWaste], severe_util: float) -> str:
    if not rows:
        return "missing_per_head_stats"
    severe = [row for row in rows if row.median_util < severe_util]
    if severe:
        return "runtime_scale_work_worth_testing"
    return "scalar_scale_probably_not_primary_limit"


def format_table(rows: list[ProjectionWaste], limit: int) -> str:
    if not rows:
        return "no per-head stats found"
    lines = [
        f"{'layer':>5} {'proj':>4} {'median_util':>11} {'scalar/median':>13} {'max/min':>9} {'scale':>10} {'samples':>7}"
    ]
    for row in rows[:limit]:
        lines.append(
            f"{row.layer:>5} {row.projection:>4} {row.median_util:>11.3f} "
            f"{row.scalar_over_median if row.scalar_over_median is not None else float('nan'):>13.3f} "
            f"{row.max_over_min if row.max_over_min is not None else float('nan'):>9.3f} "
            f"{row.scalar_scale if row.scalar_scale is not None else float('nan'):>10.4f} "
            f"{row.n_samples if row.n_samples is not None else -1:>7}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path, help="fp8kv_scales_report.json")
    ap.add_argument("--limit", type=int, default=16, help="Rows to print.")
    ap.add_argument(
        "--severe-util",
        type=float,
        default=DEFAULT_SEVERE_UTIL,
        help="Median utilization below this value is severe.",
    )
    ap.add_argument(
        "--warn-util",
        type=float,
        default=DEFAULT_WARN_UTIL,
        help="Median utilization below this value is shown as warning count.",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON summary.")
    args = ap.parse_args()

    report = load_report(args.report)
    rows = collect_projection_waste(report)
    severe = [row for row in rows if row.median_util < args.severe_util]
    warn = [row for row in rows if row.median_util < args.warn_util]
    decision = classify(rows, args.severe_util)

    if args.json:
        print(
            json.dumps(
                {
                    "report": str(args.report),
                    "num_projection_rows": len(rows),
                    "severe_util_threshold": args.severe_util,
                    "warn_util_threshold": args.warn_util,
                    "num_severe": len(severe),
                    "num_warn": len(warn),
                    "decision": decision,
                    "worst": [row.__dict__ for row in rows[: args.limit]],
                },
                indent=2,
            )
        )
        return 0

    print(f"report: {args.report}")
    print(f"projection rows with per-head stats: {len(rows)}")
    print(f"warn rows median_util < {args.warn_util}: {len(warn)}")
    print(f"severe rows median_util < {args.severe_util}: {len(severe)}")
    print(f"decision: {decision}")
    print()
    print(format_table(rows, args.limit))

    if decision == "runtime_scale_work_worth_testing":
        print()
        print(
            "interpretation: scalar layer-wide FP8 KV scales waste enough per-head "
            "range that per-head/per-block runtime scale work is a plausible next "
            "accuracy lever."
        )
    elif decision == "scalar_scale_probably_not_primary_limit":
        print()
        print(
            "interpretation: per-head imbalance is not severe under this report; "
            "prioritize post-GPTQ scale-source alignment or non-KV quantization loss."
        )
    else:
        print()
        print(
            "interpretation: rerun kv_calibrate.py from the root script version that "
            "records k_per_head/v_per_head diagnostics."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
