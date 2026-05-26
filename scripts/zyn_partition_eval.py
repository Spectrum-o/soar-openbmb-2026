#!/usr/bin/env python3
"""Resumable partitioned evaluator for zyn local SGLang runs.

This keeps the platform-like generation settings from soar_toolkit/eval_model.py
but writes each completed sample immediately so long runs can resume after a
server or session interruption.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer, GenerationConfig


MCQ_TASKS = {"mcq"}
LONG_CONTEXT_TASKS = {"niah", "cwe", "fwe", "qa", "lcx"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def completed_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("state") == "ok":
                done.add(int(row["index"]))
    return done


def extract_final_answer(pred: str) -> str:
    parts = pred.split("</think>")
    return parts[-1].strip() if len(parts) > 1 else pred


def extract_mcq_answer(pred: str) -> str | None:
    match = re.search(r"(?i)ANSWER\s*:\s*([A-D])", pred)
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{\\text\{([A-D])\}\}", pred)
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{([A-D])\}", pred)
    if match:
        return match.group(1).upper()
    return None


def score_mcq(pred: str, gold: str) -> tuple[float, str | None]:
    if not pred or not gold:
        return 0.0, None
    extracted = extract_mcq_answer(extract_final_answer(pred))
    return (1.0 if extracted and extracted.upper() == gold.upper() else 0.0), extracted


def score_exact_match(pred: str, gold: Any, task: str) -> float:
    if not pred or not gold:
        return 0.0
    final = extract_final_answer(pred)
    golds = gold if isinstance(gold, list) else [gold]
    if task in {"qa", "niah", "lcx"}:
        return 1.0 if any(str(g).lower() in final.lower() for g in golds) else 0.0
    hits = sum(1.0 for g in golds if str(g).lower() in final.lower())
    return hits / len(golds) if golds else 0.0


def score_prediction(pred: str, gold: Any, task: str) -> tuple[float, str | None]:
    if task in MCQ_TASKS:
        return score_mcq(pred, gold)
    if task in LONG_CONTEXT_TASKS:
        return score_exact_match(pred, gold, task), None
    if isinstance(gold, str) and gold.lower() in pred.lower():
        return 1.0, None
    return 0.0, None


class Client:
    def __init__(self, api_base: str, model_name: str, model_path: Path, timeout: int):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.stop_words = self._stop_words(model_path)

    def _stop_words(self, model_path: Path) -> list[str]:
        words: list[str] = []
        try:
            cfg = GenerationConfig.from_pretrained(model_path)
            eos = getattr(cfg, "eos_token_id", None)
            ids = [eos] if isinstance(eos, int) else (eos or [])
            for tid in ids:
                word = self.tokenizer.decode(tid)
                if word:
                    words.append(word)
        except Exception:
            pass
        if self.tokenizer.eos_token:
            words.append(self.tokenizer.eos_token)
        return sorted(set(w for w in words if w))

    def generate(self, prompt: str, max_out_len: int) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_out_len,
            "stop": self.stop_words,
        }
        resp = self.session.post(
            f"{self.api_base}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"], data.get("usage", {})

    def token_len(self, text: str) -> int:
        message = [{"role": "user", "content": text}]
        encoded = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_dict=True
        )
        return len(encoded["input_ids"])


def detect_model(api_base: str, timeout: int) -> str:
    session = requests.Session()
    session.trust_env = False
    resp = session.get(f"{api_base.rstrip('/')}/v1/models", timeout=timeout)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def summarize(rows: list[dict[str, Any]], started_at: float) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("state") == "ok"]
    score = sum(float(r.get("score", 0.0)) for r in ok_rows)
    total_out = sum(int(r.get("output_tokens", 0)) for r in ok_rows)
    duration = time.time() - started_at
    acc = (score / len(ok_rows) * 100.0) if ok_rows else 0.0
    return {
        "completed": len(ok_rows),
        "ori_accuracy": round(acc, 2),
        "overall_accuracy": min(round(acc / 80 * 100, 2), 100),
        "total_output_tokens": total_out,
        "duration": round(duration, 2),
        "tps": round(total_out / duration, 2) if duration > 0 else 0.0,
    }


def read_success_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    latest: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state") == "ok":
                latest[int(row["index"])] = row
    return [latest[i] for i in sorted(latest)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--api-base", default="http://127.0.0.1:31111")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=262144)
    parser.add_argument("--max-out-len", type=int, default=65536)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--part-size", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--max-failures", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"

    args.run_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.run_dir / "predictions.jsonl"
    error_path = args.run_dir / "errors.jsonl"
    model_name = args.model_name or detect_model(args.api_base, args.timeout)
    data = load_jsonl(args.data_path)
    end_index = args.end_index if args.end_index is not None else len(data)
    target_indices = list(range(args.start_index, min(end_index, len(data))))
    done = completed_indices(result_path)
    client = Client(args.api_base, model_name, args.model_path, args.timeout)

    write_json(
        args.run_dir / "run_config.json",
        {
            "model_path": str(args.model_path),
            "api_base": args.api_base,
            "model_name": model_name,
            "data_path": str(args.data_path),
            "max_seq_len": args.max_seq_len,
            "max_out_len": args.max_out_len,
            "concurrency": args.concurrency,
            "part_size": args.part_size,
            "start_index": args.start_index,
            "end_index": end_index,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    started_at = time.time()
    failures = 0
    for part_start in range(args.start_index, end_index, args.part_size):
        part_end = min(part_start + args.part_size, end_index)
        part_indices = [i for i in range(part_start, part_end) if i in target_indices]
        pending = [i for i in part_indices if i not in done]
        part_file = args.run_dir / f"part_{part_start:05d}_{part_end:05d}.jsonl"
        if not pending:
            write_json(
                args.run_dir / f"summary_part_{part_start:05d}_{part_end:05d}.json",
                {"part_start": part_start, "part_end": part_end, "skipped": True},
            )
            continue

        print(
            f"[partition] {part_start}:{part_end} pending={len(pending)} "
            f"completed={len(done)}",
            flush=True,
        )

        def infer(index: int) -> dict[str, Any]:
            item = data[index]
            pred, usage = client.generate(item["question"], args.max_out_len)
            score, extracted = score_prediction(pred, item.get("gold"), item.get("task", "unknown"))
            row = {
                "state": "ok",
                "index": index,
                "task": item.get("task", "unknown"),
                "gold": item.get("gold"),
                "prediction": pred,
                "score": score,
                "extracted": extracted,
                "usage": usage,
                "input_tokens": client.token_len(item["question"]),
                "output_tokens": client.token_len(pred),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            return row

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(infer, i): i for i in pending}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    failures += 1
                    append_jsonl(
                        error_path,
                        {
                            "state": "error",
                            "index": index,
                            "error": repr(exc),
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    )
                    print(f"[error] index={index} {exc!r}", flush=True)
                    if failures >= args.max_failures:
                        rows = read_success_rows(result_path)
                        write_json(args.run_dir / "summary.json", summarize(rows, started_at))
                        return 2
                    continue

                append_jsonl(result_path, row)
                append_jsonl(part_file, row)
                done.add(index)
                rows = read_success_rows(result_path)
                write_json(
                    args.run_dir / "progress.json",
                    {
                        "completed": len(done),
                        "last_index": index,
                        "target_total": len(target_indices),
                        "latest_accuracy": summarize(rows, started_at)["ori_accuracy"],
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                print(
                    f"[ok] index={index} score={row['score']} "
                    f"completed={len(done)}/{len(target_indices)}",
                    flush=True,
                )

        part_rows = read_success_rows(part_file)
        write_json(
            args.run_dir / f"summary_part_{part_start:05d}_{part_end:05d}.json",
            {
                "part_start": part_start,
                "part_end": part_end,
                **summarize(part_rows, started_at),
            },
        )
        write_json(args.run_dir / "summary.json", summarize(read_success_rows(result_path), started_at))

    final_rows = read_success_rows(result_path)
    write_json(args.run_dir / "summary.json", summarize(final_rows, started_at))
    print(json.dumps(summarize(final_rows, started_at), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
