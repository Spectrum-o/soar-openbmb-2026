#!/usr/bin/env python3
"""Small local smoke test for the FP8 KV SGLang server.

Runs a few SOAR public-set prompts against an OpenAI-compatible endpoint and
reports latency plus a simple gold-substring hit check. This is intentionally
small: use it before packing/pushing to verify the server is alive, long-context
prefill works, and decode returns non-empty text.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "/root/autodl-fs/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_dequant-quantized"
DEFAULT_DATA = "/root/autodl-tmp/zyn/sglang/submission_gptqmodel_calib_w4a16/perf_public_set.jsonl"


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iter_records(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:31111/v1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"data file not found: {data_path}")

    url = args.base_url.rstrip("/") + "/chat/completions"
    ok = 0
    hits = 0
    started = time.time()

    for n, rec in enumerate(iter_records(data_path, args.limit), start=1):
        question = rec.get("question") or rec.get("prompt") or rec.get("input")
        if not isinstance(question, str):
            print(f"[{n}] SKIP: no question/prompt/input field")
            continue
        gold = rec.get("gold") or rec.get("answers") or []
        if isinstance(gold, str):
            gold = [gold]

        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        t0 = time.time()
        try:
            out = post_json(url, payload, args.timeout)
        except urllib.error.HTTPError as exc:
            print(f"[{n}] HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}")
            return 2
        except Exception as exc:
            print(f"[{n}] ERROR: {exc}")
            return 2

        text = out.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = out.get("usage", {})
        elapsed = time.time() - t0
        hit = any(str(g).lower() in text.lower() for g in gold if str(g).strip())
        ok += bool(text.strip())
        hits += hit
        print(
            f"[{n}] ok={bool(text.strip())} gold_hit={hit} "
            f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')} "
            f"latency={elapsed:.2f}s"
        )
        preview = " ".join(text.split())[:220]
        print(f"    output: {preview}")

    total = time.time() - started
    print(f"summary: non_empty={ok}/{args.limit}, gold_hits={hits}/{args.limit}, elapsed={total:.2f}s")
    return 0 if ok == args.limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
