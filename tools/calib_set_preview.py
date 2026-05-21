#!/usr/bin/env python3
"""Preview what GPTQ would actually see, without running GPTQ.

Given a calibration JSONL + max_len + window_mode, tokenizes each prompt
(or estimates with --tokenizer mock), runs the same window-slicing logic
that quantize_gptqmodel_w4a16.py uses, and dumps per-window head/tail
tokens so you can visually verify the calib choice before committing
to a 14-min quant run.

This mirrors `_slice_windows_for_prompt()` from the live quantize script.
If you change the policy there, update this file too.

Usage:
    # mock tokenizer, local dev
    python3 tools/calib_set_preview.py \\
        --input submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \\
        --tokenizer mock \\
        --max-len 8192 \\
        --window-mode multi-adaptive \\
        --num-show 5

    # real tokenizer, server
    python3 tools/calib_set_preview.py \\
        --input /root/autodl-fs/zyn/calib_sets/calib_long.jsonl \\
        --tokenizer /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\
        --max-len 8192 \\
        --window-mode multi-adaptive
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Window slicing — must match quantize_gptqmodel_w4a16.py exactly.
# Single source of truth: change both files together.
# ---------------------------------------------------------------------------
def slice_windows_for_prompt(
    n_tokens: int, max_len: int, rng: random.Random
) -> list[tuple[int, int]]:
    """Return (start, end) windows under the multi-adaptive policy.

    Policy:
        n <= L:             [(0, n)]
        L < n <= 4L:        [(n-L, n)]                       — tail only
        4L < n <= 12.5L:    tail + centered mid              — 2 windows
        n > 12.5L:          tail + mid + 1 random window     — 3 windows

    The random window is drawn from regions that do NOT overlap the tail
    or the centered mid (the naive "anywhere in [L, n-2L]" would let it
    collide with the centered mid; we exclude that span explicitly).
    """
    L = max_len
    if n_tokens <= L:
        return [(0, n_tokens)]
    if n_tokens <= 4 * L:
        return [(n_tokens - L, n_tokens)]
    windows = [(n_tokens - L, n_tokens)]
    mid_start = (n_tokens - L) // 2
    mid_end = mid_start + L
    windows.append((mid_start, mid_end))
    if n_tokens > int(12.5 * L):
        # Random window placed in a region disjoint from tail and centered mid.
        #   region A (before mid): start in [L, mid_start - L]
        #   region B (after mid):  start in [mid_end, (n - L) - L]
        candidates: list[tuple[int, int]] = []
        if mid_start - L >= L:
            candidates.append((L, mid_start - L))
        if (n_tokens - 2 * L) >= mid_end:
            candidates.append((mid_end, n_tokens - 2 * L))
        if candidates:
            lo, hi = rng.choice(candidates)
            start = rng.randint(lo, hi)
            windows.append((start, start + L))
    return windows


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------
class Tokenizer:
    def __init__(self, path: str):
        if path == "mock":
            self._mock = True
            self._tk = None
        else:
            from transformers import AutoTokenizer  # noqa
            self._mock = False
            self._tk = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    def encode(self, text: str) -> list[int]:
        if self._mock:
            # Make up integer "tokens": one per word, plus 30% extra to mimic
            # subword splitting. Not real tokens but length-faithful.
            words = text.split()
            ids: list[int] = []
            for w in words:
                ids.append(hash(w) & 0xFFFF)
                if len(w) > 6:
                    ids.append(hash(w + "_subword") & 0xFFFF)
            return ids
        return self._tk.encode(text, add_special_tokens=True)

    def decode(self, ids: list[int]) -> str:
        if self._mock:
            return "<mock>" + " ".join(str(i) for i in ids[:8]) + "..."
        return self._tk.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[load] line {i}: skip bad json: {exc}", file=sys.stderr)


def extract_text(row: dict) -> str:
    for k in ("question", "prompt", "text", "input"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    raise KeyError(f"no text in row keys: {list(row.keys())}")


def render_with_chat(text: str, tokenizer, disable: bool) -> str:
    """Mirror the production chat-template handling."""
    if disable or getattr(tokenizer, "_mock", True):
        return text
    tk = tokenizer._tk
    if not (hasattr(tk, "apply_chat_template") and getattr(tk, "chat_template", None)):
        return text
    try:
        return tk.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="calib JSONL")
    ap.add_argument("--tokenizer", required=True, help="HF path or 'mock'")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument(
        "--window-mode",
        choices=["tail", "multi-adaptive"],
        default="multi-adaptive",
    )
    ap.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Skip apply_chat_template() (matches --no-chat-template in quant script)",
    )
    ap.add_argument(
        "--num-show",
        type=int,
        default=3,
        help="Show window dumps for the first N rows; others counted only",
    )
    ap.add_argument(
        "--peek-tokens",
        type=int,
        default=12,
        help="Tokens to decode at the head/tail of each window for preview",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        print(f"input not found: {src}", file=sys.stderr)
        return 2

    print(f"[preview] tokenizer={args.tokenizer}  max_len={args.max_len}  "
          f"mode={args.window_mode}  chat_template_off={args.disable_chat_template}",
          file=sys.stderr)
    tk = Tokenizer(args.tokenizer)

    rng = random.Random(args.seed)
    buckets = Counter()
    total_windows = 0
    n_rows = 0
    shown = 0
    for row in iter_jsonl(src):
        n_rows += 1
        try:
            text = extract_text(row)
        except KeyError as exc:
            print(f"[preview] skip: {exc}", file=sys.stderr)
            continue
        rendered = render_with_chat(text, tk, args.disable_chat_template)
        ids = tk.encode(rendered)
        n = len(ids)

        if args.window_mode == "tail":
            slices = [(max(0, n - args.max_len), n)]
        else:
            slices = slice_windows_for_prompt(n, args.max_len, rng)
        total_windows += len(slices)

        # Bucket label for stats
        L = args.max_len
        if n <= L:
            bucket = "short(<=L)"
        elif n <= 4 * L:
            bucket = "tail-only(<=4L)"
        elif n <= int(12.5 * L):
            bucket = "tail+mid(<=12.5L)"
        else:
            bucket = "tail+mid+rand(>12.5L)"
        buckets[bucket] += 1

        if shown < args.num_show:
            task = row.get("task", "?")
            idx = row.get("index", n_rows - 1)
            ratio = n / args.max_len
            print(
                f"\n--- row {idx} task={task} n_tokens={n:,} "
                f"({ratio:.1f}L) → {len(slices)} window(s)"
            )
            for i, (s, e) in enumerate(slices):
                head = tk.decode(ids[s:s + args.peek_tokens])
                tail = tk.decode(ids[max(s, e - args.peek_tokens):e])
                print(
                    f"  w{i}: [{s:,}:{e:,}] (size {e-s})\n"
                    f"    head: {head!r}\n"
                    f"    tail: {tail!r}"
                )
            shown += 1

    avg_w = total_windows / max(n_rows, 1)
    print()
    print(f"[preview] rows={n_rows}  windows={total_windows}  avg={avg_w:.2f}/row")
    for b in (
        "short(<=L)",
        "tail-only(<=4L)",
        "tail+mid(<=12.5L)",
        "tail+mid+rand(>12.5L)",
    ):
        print(f"  {b}: {buckets[b]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
