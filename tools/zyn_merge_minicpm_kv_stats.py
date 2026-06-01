#!/usr/bin/env python3
"""Merge MiniCPM SGLang KV calibration stats into an FP8 scale JSON."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


FP8_E4M3_MAX = 240.0
DEFAULT_MODEL_PATH = Path(
    "/autodl-fs/data/zyn/models/"
    "submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized"
)
DEFAULT_STATS_DIR = Path("/autodl-fs/data/zyn/kv_calib_stats")
DEFAULT_OUTPUT_DIR = Path("/autodl-fs/data/zyn/kv_scales")


def log(message: str) -> None:
    print(f"[kv-merge {time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_config(model_path: Path) -> dict[str, Any]:
    with (model_path / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def default_output_path(model_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{model_path.name}_fp8_e4m3_kv_scales.json"


def merge_stats_files(stats_dir: Path) -> tuple[dict[int, dict[int, dict[str, Any]]], int]:
    files = sorted(glob.glob(str(stats_dir / "minicpm_kv_stats_rank*_pid*.json")))
    if not files:
        raise FileNotFoundError(f"no MiniCPM KV stats files found in {stats_dir}")

    merged: dict[int, dict[int, dict[str, Any]]] = {}
    inferred_tp_size = 1
    for filename in files:
        with open(filename, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rank = int(payload.get("tp_rank", 0))
        inferred_tp_size = max(inferred_tp_size, int(payload.get("tp_size", 1)))
        rank_stats = merged.setdefault(rank, {})
        for layer_key, layer_payload in payload.get("layers", {}).items():
            layer_idx = int(layer_key)
            stats = rank_stats.setdefault(
                layer_idx,
                {
                    "k_amax": 0.0,
                    "v_amax": 0.0,
                    "tokens": 0,
                    "updates": 0,
                    "sources": [],
                },
            )
            stats["k_amax"] = max(
                float(stats["k_amax"]), float(layer_payload.get("k_amax", 0.0))
            )
            stats["v_amax"] = max(
                float(stats["v_amax"]), float(layer_payload.get("v_amax", 0.0))
            )
            stats["tokens"] = int(stats["tokens"]) + int(
                layer_payload.get("tokens", 0)
            )
            stats["updates"] = int(stats["updates"]) + int(
                layer_payload.get("updates", 0)
            )
            stats["sources"].append(filename)
    return merged, inferred_tp_size


def build_layer_map(
    config: dict[str, Any],
    rank_stats: dict[int, dict[str, Any]],
    safety_margin: float,
    min_scale: float,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    num_hidden_layers = int(config["num_hidden_layers"])
    mixer_types = config.get("mixer_types") or ["minicpm4"] * num_hidden_layers
    layer_map: dict[str, float] = {}
    layer_report: dict[str, dict[str, Any]] = {}

    for layer_idx in range(num_hidden_layers):
        mixer_type = mixer_types[layer_idx]
        if mixer_type != "minicpm4":
            layer_map[str(layer_idx)] = 1.0
            layer_report[str(layer_idx)] = {
                "mixer_type": mixer_type,
                "scale": 1.0,
                "skipped": True,
            }
            continue

        stats = rank_stats.get(layer_idx)
        if stats is None:
            raise RuntimeError(
                f"missing KV stats for minicpm4 layer {layer_idx}; "
                "run calibration prompts through the SGLang server first"
            )
        amax = max(float(stats["k_amax"]), float(stats["v_amax"]))
        if not math.isfinite(amax) or amax <= 0.0:
            raise RuntimeError(f"bad amax for layer {layer_idx}: {amax}")
        scale = max(min_scale, (amax / FP8_E4M3_MAX) * safety_margin)
        layer_map[str(layer_idx)] = float(scale)
        layer_report[str(layer_idx)] = {
            "mixer_type": mixer_type,
            "k_amax": float(stats["k_amax"]),
            "v_amax": float(stats["v_amax"]),
            "amax": amax,
            "scale": scale,
            "tokens": int(stats["tokens"]),
            "updates": int(stats["updates"]),
            "source_count": len(stats.get("sources", [])),
        }

    return layer_map, layer_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--safety-margin", type=float, default=1.05)
    parser.add_argument("--min-scale", type=float, default=1e-8)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument(
        "--fill-missing-ranks",
        action="store_true",
        help="Copy global max stats into missing TP ranks instead of failing.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_path.is_dir():
        print(f"model path not found: {args.model_path}", file=sys.stderr)
        return 2
    if not args.stats_dir.is_dir():
        print(f"stats dir not found: {args.stats_dir}", file=sys.stderr)
        return 2

    output_path = args.output or default_output_path(args.model_path)
    if output_path.exists() and not args.force:
        print(f"output exists, pass --force to overwrite: {output_path}", file=sys.stderr)
        return 2

    config = read_config(args.model_path)
    merged, inferred_tp_size = merge_stats_files(args.stats_dir)
    tp_size = args.tp_size or inferred_tp_size
    if tp_size <= 0:
        print("--tp-size must be positive", file=sys.stderr)
        return 2

    global_stats: dict[int, dict[str, Any]] = {}
    for rank_stats in merged.values():
        for layer_idx, layer_stats in rank_stats.items():
            stats = global_stats.setdefault(
                layer_idx,
                {"k_amax": 0.0, "v_amax": 0.0, "tokens": 0, "updates": 0, "sources": []},
            )
            stats["k_amax"] = max(float(stats["k_amax"]), float(layer_stats["k_amax"]))
            stats["v_amax"] = max(float(stats["v_amax"]), float(layer_stats["v_amax"]))
            stats["tokens"] = int(stats["tokens"]) + int(layer_stats["tokens"])
            stats["updates"] = int(stats["updates"]) + int(layer_stats["updates"])
            stats["sources"].extend(layer_stats.get("sources", []))

    scaling_factor: dict[str, dict[str, float]] = {}
    report_layers: dict[str, Any] = {}
    for rank in range(tp_size):
        rank_stats = merged.get(rank)
        if rank_stats is None:
            if not args.fill_missing_ranks:
                raise RuntimeError(
                    f"missing stats for TP rank {rank}; available ranks={sorted(merged)}"
                )
            rank_stats = global_stats
        layer_map, layer_report = build_layer_map(
            config, rank_stats, args.safety_margin, args.min_scale
        )
        scaling_factor[str(rank)] = layer_map
        report_layers[str(rank)] = layer_report

    quant_param = {
        "model_type": config.get("model_type", "minicpm_sala"),
        "kv_cache": {
            "dtype": "float8_e4m3fn",
            "scaling_factor": scaling_factor,
        },
    }
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(args.model_path),
        "stats_dir": str(args.stats_dir),
        "output_path": str(output_path),
        "tp_size": tp_size,
        "available_ranks": sorted(merged),
        "fp8_max": FP8_E4M3_MAX,
        "safety_margin": args.safety_margin,
        "layers": report_layers,
    }

    atomic_write_json(output_path, quant_param)
    atomic_write_json(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    log(f"wrote scales={output_path}")
    log(f"wrote report={output_path.with_suffix(output_path.suffix + '.report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
