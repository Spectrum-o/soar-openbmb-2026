#!/usr/bin/env python3
"""Collect FP8 KV-cache scale factors for MiniCPM-SALA on the selected W4 model.

The output JSON matches SGLang's --quantization-param-path schema. It must be
collected on the same W4A16 artifact that will be served, because the hidden
states feeding K/V projections change after W4 quantization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path(
    "/autodl-fs/data/zyn/models/"
    "submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized"
)
DEFAULT_DATA_PATH = Path("/autodl-fs/data/zyn/soar_toolkit/perf_public_set.jsonl")
DEFAULT_OUTPUT_DIR = Path("/autodl-fs/data/zyn/kv_scales")
FP8_E4M3_MAX = 240.0
BUCKET_EDGES = (8192, 32768, 102400)
BUCKET_ORDER = ("super", "long", "medium", "short")


def log(msg: str) -> None:
    print(f"[kv-calib {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log(f"skip bad json line={line_no}: {exc}")
    return rows


def extract_text(row: dict[str, Any]) -> str:
    for key in ("question", "prompt", "text", "input"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    raise KeyError(f"no prompt field in row keys={sorted(row.keys())}")


def prompt_len_hint(row: dict[str, Any]) -> int:
    val = row.get("prompt_tokens")
    if isinstance(val, int) and val > 0:
        return val
    try:
        return max(1, int(len(extract_text(row).split()) * 1.3))
    except KeyError:
        return 0


def row_uid(row: dict[str, Any]) -> str:
    if "index" in row:
        return str(row["index"])
    text = extract_text(row)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def bucket_for(n_tokens: int) -> str:
    if n_tokens <= BUCKET_EDGES[0]:
        return "short"
    if n_tokens <= BUCKET_EDGES[1]:
        return "medium"
    if n_tokens <= BUCKET_EDGES[2]:
        return "long"
    return "super"


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            uid = row_uid(row)
        except KeyError:
            continue
        if uid in seen:
            continue
        seen.add(uid)
        out.append(row)
    return out


def select_representative_rows(
    rows: list[dict[str, Any]], num_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Prefer long-context rows, while keeping task diversity."""
    rng = random.Random(seed)
    rows = dedupe_rows(rows)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[bucket_for(prompt_len_hint(row))].append(row)

    selected: list[dict[str, Any]] = []
    selected_uids: set[str] = set()
    nonempty = [b for b in BUCKET_ORDER if by_bucket.get(b)]
    per_bucket = max(1, math.ceil(num_samples / max(1, len(nonempty))))

    for bucket in nonempty:
        task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_bucket[bucket]:
            task_groups[str(row.get("task", "?"))].append(row)
        for task, task_rows in task_groups.items():
            task_rows.sort(key=prompt_len_hint, reverse=True)
            head = task_rows[: min(3, len(task_rows))]
            rng.shuffle(head)
            task_groups[task] = head + task_rows[len(head) :]

        task_names = sorted(task_groups)
        while (
            len([r for r in selected if bucket_for(prompt_len_hint(r)) == bucket])
            < per_bucket
        ):
            made_progress = False
            for task in task_names:
                if not task_groups[task]:
                    continue
                row = task_groups[task].pop(0)
                uid = row_uid(row)
                if uid in selected_uids:
                    continue
                selected.append(row)
                selected_uids.add(uid)
                made_progress = True
                if len(selected) >= num_samples:
                    return selected
                if (
                    len(
                        [
                            r
                            for r in selected
                            if bucket_for(prompt_len_hint(r)) == bucket
                        ]
                    )
                    >= per_bucket
                ):
                    break
            if not made_progress:
                break

    remaining = [r for r in rows if row_uid(r) not in selected_uids]
    remaining.sort(key=prompt_len_hint, reverse=True)
    for row in remaining:
        selected.append(row)
        if len(selected) >= num_samples:
            break
    return selected[:num_samples]


def plan_windows(
    n_tokens: int, max_window_len: int, max_windows_per_sample: int, rng: random.Random
) -> list[tuple[int, int, str]]:
    if n_tokens <= max_window_len:
        return [(0, n_tokens, "full")]

    windows: list[tuple[int, int, str]] = [
        (n_tokens - max_window_len, n_tokens, "tail")
    ]
    if max_windows_per_sample <= 1:
        return windows

    mid_start = max(0, (n_tokens - max_window_len) // 2)
    windows.append((mid_start, mid_start + max_window_len, "middle"))
    if max_windows_per_sample <= 2:
        return windows

    candidates: list[tuple[int, int]] = []
    if mid_start - max_window_len >= 0:
        candidates.append((0, mid_start - max_window_len))
    mid_end = mid_start + max_window_len
    if n_tokens - max_window_len >= mid_end:
        candidates.append((mid_end, n_tokens - max_window_len))
    if candidates:
        lo, hi = rng.choice(candidates)
        start = rng.randint(lo, hi)
        windows.append((start, start + max_window_len, "random"))
    return windows[:max_windows_per_sample]


def read_config(model_path: Path) -> dict[str, Any]:
    with (model_path / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def default_output_path(model_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{model_path.name}_fp8_e4m3_kv_scales.json"


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def dtype_from_name(torch_mod: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch_mod.bfloat16
    if name == "float16":
        return torch_mod.float16
    if name == "float32":
        return torch_mod.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_hf_config(model_path: Path) -> Any:
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except ValueError as exc:
        if "minicpm_sala" not in str(exc):
            raise

    sys.path.insert(0, str(model_path))
    try:
        from configuration_minicpm_sala import MiniCPMSALAConfig

        return MiniCPMSALAConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
    finally:
        try:
            sys.path.remove(str(model_path))
        except ValueError:
            pass


def get_model_device(model: Any, fallback: str) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def tensor_from_hook_output(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output


def token_count_from_output(output: Any) -> int:
    if output.ndim >= 3:
        return int(output.shape[0] * output.shape[1])
    if output.ndim >= 2:
        return int(output.shape[0])
    return 0


def update_amax(
    stats: dict[int, dict[str, Any]], layer_idx: int, name: str, output: Any
) -> None:
    tensor = tensor_from_hook_output(output)
    if tensor is None:
        return
    detached = tensor.detach()
    amax = float(detached.abs().float().amax().item())
    stats[layer_idx][f"{name}_amax"] = max(stats[layer_idx][f"{name}_amax"], amax)
    stats[layer_idx][f"{name}_tokens"] += token_count_from_output(detached)
    stats[layer_idx][f"{name}_dtype"] = str(detached.dtype)


def register_kv_hooks(
    model: Any, mixer_types: list[str]
) -> tuple[list[Any], dict[int, dict[str, Any]]]:
    stats: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "k_amax": 0.0,
            "v_amax": 0.0,
            "k_tokens": 0,
            "v_tokens": 0,
            "k_dtype": "",
            "v_dtype": "",
        }
    )
    handles: list[Any] = []
    layers = getattr(getattr(model, "model", model), "layers")

    for layer_idx, layer in enumerate(layers):
        if mixer_types and mixer_types[layer_idx] != "minicpm4":
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue

        if hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
            handles.append(
                attn.k_proj.register_forward_hook(
                    lambda _m, _inp, out, idx=layer_idx: update_amax(
                        stats, idx, "k", out
                    )
                )
            )
            handles.append(
                attn.v_proj.register_forward_hook(
                    lambda _m, _inp, out, idx=layer_idx: update_amax(
                        stats, idx, "v", out
                    )
                )
            )
            continue

        if hasattr(attn, "qkv_proj"):
            q_size = int(getattr(attn, "q_size"))
            kv_size = int(getattr(attn, "kv_size"))

            def qkv_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                idx: int = layer_idx,
                q_dim: int = q_size,
                kv_dim: int = kv_size,
            ) -> None:
                qkv = tensor_from_hook_output(output)
                if qkv is None:
                    return
                _q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
                update_amax(stats, idx, "k", k)
                update_amax(stats, idx, "v", v)

            handles.append(attn.qkv_proj.register_forward_hook(qkv_hook))

    return handles, stats


def encode_prompt(tokenizer: Any, text: str, disable_chat_template: bool) -> list[int]:
    if (
        not disable_chat_template
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return list(ids)
    return list(tokenizer.encode(text, add_special_tokens=True))


def gpu_snapshot(torch_mod: Any) -> dict[str, Any]:
    if not torch_mod.cuda.is_available():
        return {"cuda": False}
    free, total = torch_mod.cuda.mem_get_info()
    return {
        "cuda": True,
        "device": torch_mod.cuda.get_device_name(0),
        "free_gb": round(free / 1024**3, 2),
        "total_gb": round(total / 1024**3, 2),
        "allocated_gb": round(torch_mod.cuda.memory_allocated() / 1024**3, 2),
        "reserved_gb": round(torch_mod.cuda.memory_reserved() / 1024**3, 2),
        "max_allocated_gb": round(torch_mod.cuda.max_memory_allocated() / 1024**3, 2),
    }


def run_forward_window(
    model: Any,
    torch_mod: Any,
    ids: list[int],
    device: Any,
    start: int,
    end: int,
    use_attention_mask: bool,
) -> None:
    input_ids = torch_mod.tensor([ids[start:end]], dtype=torch_mod.long, device=device)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "use_cache": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
    }
    if use_attention_mask:
        kwargs["attention_mask"] = torch_mod.ones_like(input_ids)
    with torch_mod.inference_mode():
        _ = model(**kwargs)


def build_quant_param_json(
    config: dict[str, Any],
    stats: dict[int, dict[str, Any]],
    safety_margin: float,
    min_scale: float,
    tp_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    num_hidden_layers = int(config["num_hidden_layers"])
    mixer_types = config.get("mixer_types") or ["minicpm4"] * num_hidden_layers
    layer_map: dict[str, float] = {}
    report_layers: dict[str, dict[str, Any]] = {}

    for layer_idx in range(num_hidden_layers):
        if mixer_types[layer_idx] != "minicpm4":
            layer_map[str(layer_idx)] = 1.0
            report_layers[str(layer_idx)] = {
                "mixer_type": mixer_types[layer_idx],
                "scale": 1.0,
            }
            continue

        layer_stats = stats.get(layer_idx)
        if not layer_stats:
            raise RuntimeError(
                f"no KV activation stats collected for minicpm4 layer {layer_idx}"
            )
        amax = max(float(layer_stats["k_amax"]), float(layer_stats["v_amax"]))
        scale = max(min_scale, (amax / FP8_E4M3_MAX) * safety_margin)
        layer_map[str(layer_idx)] = float(scale)
        report_layers[str(layer_idx)] = {
            "mixer_type": mixer_types[layer_idx],
            "k_amax": layer_stats["k_amax"],
            "v_amax": layer_stats["v_amax"],
            "amax": amax,
            "scale": scale,
            "k_tokens": layer_stats["k_tokens"],
            "v_tokens": layer_stats["v_tokens"],
            "k_dtype": layer_stats["k_dtype"],
            "v_dtype": layer_stats["v_dtype"],
        }

    scaling_factor = {str(rank): dict(layer_map) for rank in range(tp_size)}
    quant_param = {
        "model_type": config.get("model_type", "minicpm_sala"),
        "kv_cache": {
            "dtype": "float8_e4m3fn",
            "scaling_factor": scaling_factor,
        },
    }
    return quant_param, report_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--selected-jsonl",
        type=Path,
        default=None,
        help="Optional path to write the selected W4/KV calibration rows.",
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-window-len", type=int, default=8192)
    parser.add_argument("--max-windows-per-sample", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument(
        "--attention-mask",
        action="store_true",
        help="Pass an all-ones attention_mask. Default omits it for no-padding windows.",
    )
    parser.add_argument("--safety-margin", type=float, default=1.05)
    parser.add_argument("--min-scale", type=float, default=1e-8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip an OOM window and continue collecting remaining windows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected samples and approximate window plan; no model load.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_path.is_dir():
        print(f"model path not found: {args.model_path}", file=sys.stderr)
        return 2
    if not args.data_path.is_file():
        print(f"data path not found: {args.data_path}", file=sys.stderr)
        return 2
    if args.num_samples <= 0:
        print("--num-samples must be positive", file=sys.stderr)
        return 2
    if args.max_window_len <= 0:
        print("--max-window-len must be positive", file=sys.stderr)
        return 2
    if args.tp_size <= 0:
        print("--tp-size must be positive", file=sys.stderr)
        return 2

    output_path = args.output or default_output_path(args.model_path)
    if output_path.exists() and not args.force and not args.dry_run:
        print(
            f"output exists, pass --force to overwrite: {output_path}",
            file=sys.stderr,
        )
        return 2

    started = time.perf_counter()
    config = read_config(args.model_path)
    mixer_types = config.get("mixer_types") or []
    radix_layers = [i for i, t in enumerate(mixer_types) if t == "minicpm4"]
    log(f"model={args.model_path}")
    log(f"data={args.data_path}")
    log(f"output={output_path}")
    log(
        f"model_type={config.get('model_type')} "
        f"layers={config.get('num_hidden_layers')} radix_layers={radix_layers}"
    )
    log(
        "collection knobs: "
        f"num_samples={args.num_samples} max_window_len={args.max_window_len} "
        f"max_windows_per_sample={args.max_windows_per_sample} dtype={args.dtype} "
        f"safety_margin={args.safety_margin}"
    )

    rows = load_jsonl(args.data_path)
    selected = select_representative_rows(rows, args.num_samples, args.seed)
    selected_counts = Counter(str(r.get("task", "?")) for r in selected)
    bucket_counts = Counter(bucket_for(prompt_len_hint(r)) for r in selected)
    log(
        f"selected_rows={len(selected)} task_counts={dict(selected_counts)} "
        f"bucket_counts={dict(bucket_counts)}"
    )
    if args.selected_jsonl is not None:
        selected_out: list[dict[str, Any]] = []
        for row in selected:
            clean = dict(row)
            n_hint = prompt_len_hint(row)
            clean["kv_calib_bucket"] = bucket_for(n_hint)
            clean["kv_calib_prompt_tokens_hint"] = n_hint
            clean["kv_calib_model_path"] = str(args.model_path)
            selected_out.append(clean)
        atomic_write_jsonl(args.selected_jsonl, selected_out)
        log(f"wrote selected calibration rows={args.selected_jsonl}")

    if args.dry_run:
        rng = random.Random(args.seed)
        for pos, row in enumerate(selected):
            n = prompt_len_hint(row)
            windows = plan_windows(
                n, args.max_window_len, args.max_windows_per_sample, rng
            )
            log(
                f"dry row={pos} index={row.get('index')} task={row.get('task')} "
                f"prompt_tokens_hint={n} windows={windows}"
            )
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_config = load_hf_config(args.model_path)
    dtype = dtype_from_name(torch, args.dtype)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "config": model_config,
    }
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = "auto"
    if args.device_map == "auto":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": args.device}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    except Exception as exc:
        msg = str(exc)
        if (
            "GPTQ" in msg
            or "QuantizeConfig" in msg
            or "gptqmodel" in msg.lower()
            or "auto-gptq" in msg.lower()
        ):
            raise RuntimeError(
                "Loading the selected W4 GPTQ model requires a compatible "
                "GPTQ runtime, such as the original quantization venv with "
                "optimum plus gptqmodel/auto-gptq. This collector must run "
                "in that environment because scales must be measured on the "
                "final W4 artifact."
            ) from exc
        raise
    model.eval()
    device = get_model_device(model, args.device)
    load_elapsed = time.perf_counter() - load_t0
    log(f"loaded tokenizer+model in {load_elapsed:.2f}s gpu={gpu_snapshot(torch)}")

    handles, stats = register_kv_hooks(model, mixer_types)
    if not handles:
        raise RuntimeError("no K/V projection hooks registered; check model class")
    log(f"registered_hooks={len(handles)}")

    rng = random.Random(args.seed)
    sample_reports: list[dict[str, Any]] = []
    ok_windows = 0
    failed_windows = 0

    for sample_pos, row in enumerate(selected):
        token_t0 = time.perf_counter()
        text = extract_text(row)
        ids = encode_prompt(tokenizer, text, args.disable_chat_template)
        token_elapsed = time.perf_counter() - token_t0
        windows = plan_windows(
            len(ids), args.max_window_len, args.max_windows_per_sample, rng
        )
        log(
            f"sample={sample_pos + 1}/{len(selected)} index={row.get('index')} "
            f"task={row.get('task')} tokens={len(ids)} windows={len(windows)} "
            f"tokenize_s={token_elapsed:.2f}"
        )
        row_report = {
            "sample_pos": sample_pos,
            "index": row.get("index"),
            "task": row.get("task"),
            "tokens": len(ids),
            "windows": [],
        }
        for start, end, label in windows:
            win_t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                run_forward_window(
                    model,
                    torch,
                    ids,
                    device,
                    start,
                    end,
                    use_attention_mask=args.attention_mask,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - win_t0
                ok_windows += 1
                snapshot = gpu_snapshot(torch)
                log(
                    f"  window={label} span={start}:{end} len={end - start} "
                    f"forward_s={elapsed:.2f} gpu={snapshot}"
                )
                row_report["windows"].append(
                    {
                        "label": label,
                        "start": start,
                        "end": end,
                        "tokens": end - start,
                        "elapsed_s": round(elapsed, 3),
                        "state": "ok",
                        "gpu": snapshot,
                    }
                )
            except RuntimeError as exc:
                failed_windows += 1
                elapsed = time.perf_counter() - win_t0
                msg = repr(exc)
                log(
                    f"  window={label} span={start}:{end} len={end - start} "
                    f"FAILED after {elapsed:.2f}s: {msg[:240]}"
                )
                row_report["windows"].append(
                    {
                        "label": label,
                        "start": start,
                        "end": end,
                        "tokens": end - start,
                        "elapsed_s": round(elapsed, 3),
                        "state": "error",
                        "error": msg,
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (not args.skip_oom) or "out of memory" not in msg.lower():
                    raise
        sample_reports.append(row_report)

    for handle in handles:
        handle.remove()

    quant_param, report_layers = build_quant_param_json(
        config,
        stats,
        args.safety_margin,
        args.min_scale,
        args.tp_size,
    )
    atomic_write_json(output_path, quant_param)
    args_report = {
        key: str(val) if isinstance(val, Path) else val
        for key, val in vars(args).items()
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "output_path": str(output_path),
        "selected_w4_baseline": args.model_path.name,
        "fp8_max": FP8_E4M3_MAX,
        "args": args_report | {"output": str(output_path)},
        "radix_layers": radix_layers,
        "ok_windows": ok_windows,
        "failed_windows": failed_windows,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "layers": report_layers,
        "samples": sample_reports,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    atomic_write_json(report_path, report)
    log(f"wrote scales={output_path}")
    log(f"wrote report={report_path}")
    log(f"run_sala env: QUANTIZATION_PARAM_PATH={output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
