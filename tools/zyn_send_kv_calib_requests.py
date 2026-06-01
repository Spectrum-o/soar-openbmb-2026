#!/usr/bin/env python3
"""Send calibration prompts to a running OpenAI-compatible SGLang server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_DATA_PATH = Path("/autodl-fs/data/zyn/calib_sets/w4_kv_selected_20260601.jsonl")


def log(message: str) -> None:
    print(f"[kv-send {time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_text(row: dict[str, Any]) -> str:
    for key in ("question", "prompt", "text", "input"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(f"no prompt field in row keys={sorted(row)}")


def detect_model(session: requests.Session, api_base: str, timeout: int) -> str:
    resp = session.get(f"{api_base.rstrip('/')}/v1/models", timeout=timeout)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--api-base", default="http://127.0.0.1:31111")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--sleep-secs", type=float, default=0.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_path.is_file():
        print(f"data path not found: {args.data_path}", file=sys.stderr)
        return 2

    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    session = requests.Session()
    session.trust_env = False

    model_name = args.model_name or detect_model(session, args.api_base, args.timeout)
    rows = load_jsonl(args.data_path)
    end = len(rows) if args.limit is None else min(len(rows), args.start + args.limit)
    selected = rows[args.start:end]
    log(
        f"api={args.api_base} model={model_name} rows={len(selected)} "
        f"max_tokens={args.max_tokens}"
    )

    out_f = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_f = args.output.open("w", encoding="utf-8")

    try:
        for pos, row in enumerate(selected, args.start):
            text = extract_text(row)
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": text}],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
            }
            started = time.perf_counter()
            resp = session.post(
                f"{args.api_base.rstrip('/')}/v1/chat/completions",
                json=payload,
                timeout=args.timeout,
            )
            elapsed = time.perf_counter() - started
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            result = {
                "index": row.get("index", pos),
                "task": row.get("task"),
                "elapsed_s": round(elapsed, 3),
                "usage": usage,
                "state": "ok",
            }
            if out_f is not None:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            log(
                f"row={pos} index={result['index']} task={result['task']} "
                f"elapsed={elapsed:.2f}s usage={usage}"
            )
            if args.sleep_secs > 0:
                time.sleep(args.sleep_secs)
    finally:
        if out_f is not None:
            out_f.close()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
