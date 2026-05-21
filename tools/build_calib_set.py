#!/usr/bin/env python3
"""Build calibration sets for GPTQ from a source prompt corpus.

Bucketed by full token length, then emit per-bucket and mixed
calibration JSONL files plus a markdown report.

Use cases:
    - Tomorrow's Exp C-E: cycle through {short, medium, long, mixed} calib
      sets to test which length regime drives quant quality on SALA.
    - Holdout: 10% of source held out for later perplexity validation.

Bucket policy (defaults match perf_public_set distribution: median ~30K
tokens, max ~1M):
    short:  N <= 8192            — fits in one window
    medium: 8192 < N <= 32768
    long:   32768 < N <= 102400
    super:  N > 102400

Output files:
    calib_short.jsonl    cycled if needed to reach --per-set-size
    calib_medium.jsonl   same
    calib_long.jsonl     same
    calib_super.jsonl    (often empty for perf_public_set)
    calib_mixed.jsonl    uniform sample across non-empty buckets
    calib_holdout.jsonl  held-out rows, NOT used in any of the above
    calib_report.md      stats, distributions, source breakdown

CPU-only. Pass --tokenizer mock for development without an HF model on
disk (uses 1.3-tokens-per-word approximation).

Usage:
    # Local dev / unit testing (no HF needed)
    python3 tools/build_calib_set.py \\
        --input submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \\
        --tokenizer mock \\
        --output-dir /tmp/calib_sets

    # Server with real tokenizer
    python3 tools/build_calib_set.py \\
        --input /root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl \\
        --tokenizer /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\
        --output-dir /root/autodl-fs/zyn/calib_sets \\
        --per-set-size 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------
def iter_source(path: Path) -> Iterable[dict]:
    """Yield rows from a .jsonl (one JSON per line) or .txt (one line per row)."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[load] line {i}: skip bad json: {exc}", file=sys.stderr)
    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield {"question": line, "task": "txt"}
    else:
        raise ValueError(f"unsupported source extension: {suffix}")


def extract_text(row: dict) -> str:
    """Pull the prompt text from a row using common field names."""
    for key in ("question", "prompt", "text", "input"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    raise KeyError(f"no text field in row keys: {list(row.keys())}")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tokenization (HF or mock)
# ---------------------------------------------------------------------------
class TokenCounter:
    """Counts tokens for a single string using HF tokenizer or word-count mock."""

    def __init__(self, tokenizer_path: str):
        if tokenizer_path == "mock":
            self._mock = True
            self._tk = None
            print("[tokenizer] MOCK MODE: word count × 1.3 (approx)", file=sys.stderr)
        else:
            try:
                from transformers import AutoTokenizer  # noqa: F401
            except ImportError as exc:
                raise SystemExit(
                    f"transformers not installed; pass --tokenizer mock for local dev. ({exc})"
                )
            self._mock = False
            self._tk = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
            print(f"[tokenizer] loaded {tokenizer_path}", file=sys.stderr)

    def count(self, text: str) -> int:
        if self._mock:
            # Crude word-count to token estimate. Decent for English.
            return max(1, int(len(text.split()) * 1.3))
        return len(self._tk.encode(text, add_special_tokens=True))


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------
BUCKET_NAMES = ("short", "medium", "long", "super")


def bucket_for(n_tokens: int, bins: list[int]) -> str:
    """Classify n_tokens into a bucket name based on monotonic bin edges.

    bins = [B1, B2, B3] => short<=B1, medium<=B2, long<=B3, super>B3.
    """
    for name, edge in zip(BUCKET_NAMES[:-1], bins):
        if n_tokens <= edge:
            return name
    return BUCKET_NAMES[-1]


def sample_uniform_mix(
    buckets: dict[str, list[dict]], k: int, rng: random.Random
) -> list[dict]:
    """Sample ~k rows total, uniformly across non-empty buckets."""
    nonempty = {b: rows for b, rows in buckets.items() if rows}
    if not nonempty:
        return []
    per_bucket = max(1, k // len(nonempty))
    out: list[dict] = []
    for b, rows in nonempty.items():
        out.extend(rng.sample(rows, min(per_bucket, len(rows))))
    rng.shuffle(out)
    return out[:k]


def cycle_to_size(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Cycle a list deterministically (shuffled) until it reaches `target` size."""
    if not rows:
        return []
    if len(rows) >= target:
        return rows[:target]
    pool: list[dict] = []
    base = list(rows)
    rng.shuffle(base)
    while len(pool) < target:
        pool.extend(base)
    return pool[:target]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def percentiles(vals: list[int], ps: tuple[float, ...]) -> list[int | None]:
    if not vals:
        return [None] * len(ps)
    s = sorted(vals)
    return [s[min(int(p * (len(s) - 1)), len(s) - 1)] for p in ps]


def write_report(
    path: Path,
    dedup: list[dict],
    buckets: dict[str, list[dict]],
    sets: dict[str, list[dict]],
    bins: list[int],
    args: argparse.Namespace,
) -> None:
    n_tokens = [r["_n_tokens"] for r in dedup]
    p_min, p25, p50, p75, p90, p99, p_max = percentiles(
        n_tokens, (0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    )

    lines: list[str] = [
        "# Calibration set report\n",
        f"- source: `{args.input}`",
        f"- tokenizer: `{args.tokenizer}`",
        f"- seed: {args.seed}",
        f"- bucket bins: {bins}  (short<=B1, medium<=B2, long<=B3, super>B3)",
        f"- per-set size target: {args.per_set_size}",
        f"- holdout fraction: {args.holdout_fraction}\n",
        "## Token length distribution (after dedup)\n",
        f"- rows: {len(dedup)}",
        f"- min: {p_min:,}",
        f"- p25: {p25:,}",
        f"- p50 (median): {p50:,}",
        f"- p75: {p75:,}",
        f"- p90: {p90:,}",
        f"- p99: {p99:,}",
        f"- max: {p_max:,}\n",
        "## Bucket sizes (after dedup, before holdout split)\n",
        "| bucket | n rows |",
        "|---|---|",
    ]
    for b in BUCKET_NAMES:
        lines.append(f"| {b} | {len(buckets.get(b, []))} |")

    lines.append("\n## Emitted sets\n")
    lines.append("| set | n rows | min tokens | median tokens | max tokens |")
    lines.append("|---|---|---|---|---|")
    for name, rows in sets.items():
        if not rows:
            lines.append(f"| {name} | 0 | - | - | - |")
            continue
        ns = sorted(r.get("_n_tokens", 0) for r in rows)
        med = ns[len(ns) // 2]
        lines.append(f"| {name} | {len(rows)} | {ns[0]:,} | {med:,} | {ns[-1]:,} |")

    has_task = any("task" in r for r in dedup)
    if has_task:
        lines.append("\n## Task distribution per set\n")
        for name, rows in sets.items():
            if not rows:
                continue
            counts = Counter(r.get("task", "?") for r in rows)
            lines.append(f"- **{name}**: {dict(sorted(counts.items()))}")

    lines.append("\n## How to use\n")
    lines.append(
        "Pass the desired set as `CALIB_JSONL` to prepare_model.sh:\n```bash\n"
        "CALIB_JSONL=$(realpath calib_long.jsonl) \\\n"
        "    bash submission_gptqmodel_calib_w4a16/prepare_model.sh \\\n"
        "      --input /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\\n"
        "      --output /root/autodl-fs/zyn/models/sweep_long/\n"
        "```"
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Source .jsonl or .txt")
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer path/repo or 'mock' (CPU-only word-count estimator)",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--bins",
        default="8192,32768,102400",
        help="Three comma-separated upper bounds for short/medium/long",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--per-set-size",
        type=int,
        default=256,
        help="Target rows per emitted set; under-filled buckets cycle deterministically",
    )
    ap.add_argument("--holdout-fraction", type=float, default=0.1)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="(debug) only ingest first N rows from source",
    )
    args = ap.parse_args()

    if not 0 <= args.holdout_fraction < 1.0:
        return _exit(2, "--holdout-fraction must be in [0, 1)")
    try:
        bins = [int(x) for x in args.bins.split(",")]
    except ValueError:
        return _exit(2, f"--bins must be ints: {args.bins!r}")
    if len(bins) != 3:
        return _exit(2, "--bins must have exactly 3 values")
    if bins != sorted(bins) or len(set(bins)) != 3:
        return _exit(2, "--bins must be strictly increasing")

    input_path = Path(args.input)
    if not input_path.is_file():
        return _exit(2, f"input not found: {input_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] loading {input_path}", file=sys.stderr)
    rows = list(iter_source(input_path))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[build] loaded {len(rows)} rows", file=sys.stderr)
    if not rows:
        return _exit(2, "no rows loaded")

    # Dedup by SHA of text field
    seen: set[str] = set()
    dedup: list[dict] = []
    for r in rows:
        try:
            t = extract_text(r)
        except KeyError as exc:
            print(f"[build] skip malformed row: {exc}", file=sys.stderr)
            continue
        h = hash_text(t)
        if h in seen:
            continue
        seen.add(h)
        r["_text"] = t
        r["_hash"] = h
        dedup.append(r)
    print(f"[build] dedup: {len(rows)} -> {len(dedup)}", file=sys.stderr)

    # Tokenize (the slow step on real tokenizer)
    tk = TokenCounter(args.tokenizer)
    for i, r in enumerate(dedup):
        r["_n_tokens"] = tk.count(r["_text"])
        if (i + 1) % 50 == 0:
            print(f"[build] tokenized {i + 1}/{len(dedup)}", file=sys.stderr)

    # Bucket
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in dedup:
        buckets[bucket_for(r["_n_tokens"], bins)].append(r)
    print(
        f"[build] buckets (full): "
        + ", ".join(f"{b}={len(buckets[b])}" for b in BUCKET_NAMES),
        file=sys.stderr,
    )

    rng = random.Random(args.seed)

    # Carve holdout from each bucket proportionally (then never touch again)
    holdout: list[dict] = []
    for b in list(buckets):
        rows_b = buckets[b]
        rng.shuffle(rows_b)
        k = max(1, int(len(rows_b) * args.holdout_fraction)) if args.holdout_fraction > 0 else 0
        k = min(k, len(rows_b) - 1) if len(rows_b) > 1 else 0
        if k:
            holdout.extend(rows_b[:k])
            buckets[b] = rows_b[k:]
    print(f"[build] holdout size: {len(holdout)}", file=sys.stderr)

    # Per-bucket emitted sets (cycled to per-set-size)
    sets: dict[str, list[dict]] = {}
    for b in BUCKET_NAMES:
        sets[b] = cycle_to_size(buckets[b], args.per_set_size, rng)
    sets["mixed"] = sample_uniform_mix(buckets, args.per_set_size, rng)
    sets["holdout"] = holdout

    for name, set_rows in sets.items():
        path = out_dir / f"calib_{name}.jsonl"
        write_jsonl(set_rows, path)
        print(f"[build] wrote {path} ({len(set_rows)} rows)", file=sys.stderr)

    report_path = out_dir / "calib_report.md"
    write_report(report_path, dedup, buckets, sets, bins, args)
    print(f"[build] wrote {report_path}", file=sys.stderr)
    return 0


def _exit(code: int, msg: str) -> int:
    print(msg, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
