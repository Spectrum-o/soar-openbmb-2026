#!/usr/bin/env python3
"""KV calibration tool for FP8 KV cache (SALA + SGLang).

Why: SGLang's BaseKVCacheMethod default is k_scale = v_scale = 1.0.
For SALA with scale_emb=12, raw KV magnitudes can hit 50-500 and the
default-1.0 scale causes fp8_e5m2 mantissa saturation in long-context
decode (manifested 2026-05-26 as CWE repetition loops at AutoDL smoke).

This tool runs a BF16 forward pass on calibration data (perf_public_set
or similar) with hooks on every MiniCPMAttention.qkv_proj. The SOAR
`perf_public_set.jsonl` prompt field is `question`; older copies of this script
missed that field and silently skipped every sample, writing all-1.0 scales.
Long SOAR prompts put the actual task near the tail, so this script calibrates
with tokenizer.truncation_side="left" by default. K and V slices of the output
are extracted and per-layer max-abs tracked. The scales are written as
standalone safetensors so codex can:
  (a) inspect the values before injection;
  (b) inject into the quantized model checkpoint via the companion
      `--inject` mode (rewrites model.safetensors.index.json).

Calibration assumes BF16 forward statistics ≈ quantized forward statistics
— this is the GPTQ design intent (weight-only quant preserves activation
distribution within ~1%). For 2-bit-mantissa fp8_e5m2 that 1% drift is
within mantissa precision.

Two modes:
  1. CALIBRATE (default) — load BF16 model, run hooks, write scales to
     standalone .safetensors + JSON report.
  2. INJECT (--inject) — read previously-saved scales, write into a
     quantized model's safetensors + update index.

Usage:
  # Step 1: calibrate on BF16 model
  python3 scripts/kv_calibrate.py \\
      --model /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\
      --calib-jsonl submission_.../perf_public_set.jsonl \\
      --num-samples 64 --max-len 8192 \\
      --output /tmp/kv_scales.safetensors \\
      --report /tmp/kv_scales_report.json

  # Step 2: inject scales into quantized model
  python3 scripts/kv_calibrate.py \\
      --inject \\
      --scales /tmp/kv_scales.safetensors \\
      --quantized-model /root/autodl-fs/zyn/models/...-quantized

Skipping lightning layers: MiniCPMLightningMixer doesn't use a KV cache,
so we skip those layers entirely. The skip is automatic via the model
introspection in `find_attention_layers`.

Calibration target for the current SALA FP8KV follow-up: fp8_e4m3fn safe
range ±224. PyTorch reports fp8_e4m3fn max as 448, so this keeps 2x headroom
while using 3.7x finer scale than the prior safe-max 60 package. Adjust via
--fp8-safe-max for follow-up A/B tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


# fp8 safe ranges:
#   fp8_e5m2: full ±57344, exp range ±15, mantissa 2 bit (eps ~0.25 of value).
#     It is range-friendly but less accurate for SALA KV than e4m3.
#   fp8_e4m3fn: torch.finfo max is ±448, mantissa 3 bit (eps ~0.125 of value).
#     safe ±60 was the 17:23 ATTNSCALE conservative setting and produced
#     acc_ori=77.24. HP224 maps max observed KV to half range for finer
#     small/medium-value resolution while retaining outlier headroom.
FP8_E5M2_SAFE_MAX = 224.0
FP8_E4M3_SAFE_MAX = 224.0
FP8_E4M3_TORCH_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
FP8_E5M2_TORCH_MAX = float(torch.finfo(torch.float8_e5m2).max)


def find_attention_layers(model) -> list[tuple[int, Any]]:
    """Return list of (layer_id, qkv_proj_module) for layers that use KV cache.

    SALA has MiniCPMAttention (dense, has KV) and MiniCPMLightningMixer
    (linear-state, no KV cache). Detect by class name to be HF-modeling
    agnostic (the trust_remote_code module class lives in the model dir).
    """
    found = []
    layers = None
    for attr in ("model", "transformer", "backbone"):
        body = getattr(model, attr, None)
        if body is not None:
            layers = getattr(body, "layers", None)
            if layers is not None:
                break
    if layers is None:
        raise RuntimeError("could not locate model.layers — model architecture unknown")

    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        cls_name = type(attn).__name__
        # Skip lightning / linear-state mixers (no KV cache to calibrate)
        if "Lightning" in cls_name or "Mixer" in cls_name or "Linear" in cls_name:
            print(f"  layer {idx:>3}: SKIP ({cls_name})")
            continue
        num_kv_heads = getattr(attn, "num_kv_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(attn, "num_key_value_heads", None)
        head_dim = getattr(attn, "head_dim", None)

        # Need qkv_proj for hook attachment + q_size/kv_size for slicing
        qkv = getattr(attn, "qkv_proj", None)
        if qkv is None:
            # Fall back: maybe separate q_proj/k_proj/v_proj (non-SGLang HF model)
            kp = getattr(attn, "k_proj", None)
            vp = getattr(attn, "v_proj", None)
            if kp is not None and vp is not None:
                found.append((idx, ("split", kp, vp, num_kv_heads, head_dim)))
                print(
                    f"  layer {idx:>3}: dense ({cls_name}, split q/k/v, "
                    f"num_kv_heads={num_kv_heads} head_dim={head_dim})"
                )
            else:
                print(f"  layer {idx:>3}: SKIP ({cls_name}, no kv proj found)", file=sys.stderr)
            continue
        # SGLang QKVParallelLinear path
        q_size = getattr(attn, "q_size", None)
        kv_size = getattr(attn, "kv_size", None)
        if q_size is None or kv_size is None:
            print(f"  layer {idx:>3}: SKIP ({cls_name}, missing q_size/kv_size)", file=sys.stderr)
            continue
        found.append((idx, ("qkv_merged", qkv, q_size, kv_size, num_kv_heads, head_dim)))
        print(
            f"  layer {idx:>3}: dense ({cls_name}, qkv_proj merged, "
            f"q_size={q_size} kv_size={kv_size} "
            f"num_kv_heads={num_kv_heads} head_dim={head_dim})"
        )
    return found


def compute_per_head_absmax(
    tensor: torch.Tensor,
    num_kv_heads: int | None,
    head_dim: int | None,
) -> list[float] | None:
    """Return per-KV-head max abs values when the tensor shape is inferable."""
    if not num_kv_heads or not head_dim:
        return None
    if tensor.numel() == 0:
        return None

    if tensor.ndim >= 3 and tensor.shape[-2] == num_kv_heads and tensor.shape[-1] == head_dim:
        by_head = tensor.reshape(-1, num_kv_heads, head_dim)
    elif tensor.shape[-1] == num_kv_heads * head_dim:
        by_head = tensor.reshape(-1, num_kv_heads, head_dim)
    else:
        return None

    return by_head.detach().abs().amax(dim=(0, 2)).float().cpu().tolist()


def update_projection_stats(
    layer_stats: dict,
    prefix: str,
    tensor: torch.Tensor,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> None:
    """Update scalar and optional per-head max-abs stats for K or V projection output."""
    max_key = f"{prefix}_max"
    head_key = f"{prefix}_head_max"

    max_abs = tensor.detach().abs().max().item()
    layer_stats[max_key] = max(layer_stats[max_key], max_abs)

    head_abs = compute_per_head_absmax(tensor, num_kv_heads, head_dim)
    if head_abs is None:
        return
    existing = layer_stats.get(head_key)
    if existing is None:
        layer_stats[head_key] = head_abs
    else:
        layer_stats[head_key] = [max(a, b) for a, b in zip(existing, head_abs)]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def build_head_report(head_max: list[float] | None, safe_max: float) -> dict[str, Any] | None:
    """Summarize how much resolution a scalar per-layer scale wastes across heads."""
    if not head_max:
        return None
    values = [float(x) for x in head_max]
    nonzero = [x for x in values if x > 0]
    if not nonzero:
        return None

    max_abs = max(nonzero)
    min_abs = min(nonzero)
    median_abs = _median(nonzero)
    assert median_abs is not None
    scales = [x / safe_max if x > 0 else 1.0 for x in values]
    scalar_scale = max_abs / safe_max
    median_scale = median_abs / safe_max

    return {
        "max_abs": values,
        "scale": scales,
        "max_over_min_nonzero_abs": max_abs / min_abs if min_abs > 0 else None,
        "scalar_over_median_head_scale": (
            scalar_scale / median_scale if median_scale > 0 else None
        ),
        "median_head_utilization_under_scalar": median_abs / max_abs if max_abs > 0 else None,
    }


def build_scale_tensor_values(
    layer_stats: dict,
    prefix: str,
    safe_max: float,
    emit_per_head: bool = False,
) -> tuple[list[float], dict[str, Any] | None, str]:
    """Return scale tensor values plus optional per-head report for one projection."""
    scalar_max = layer_stats[f"{prefix}_max"]
    scalar_scale = scalar_max / safe_max if scalar_max > 0 else 1.0
    head_report = build_head_report(layer_stats.get(f"{prefix}_head_max"), safe_max)
    if emit_per_head and head_report is not None:
        return list(head_report["scale"]), head_report, "per_head"
    return [scalar_scale], head_report, "scalar"


def attach_hooks(targets: list[tuple[int, Any]], stats: dict[int, dict]):
    """Register forward hooks on each target's qkv/kv projection.

    For qkv_merged: split output by (q_size, kv_size, kv_size).
    For split: hook k_proj and v_proj separately.
    """
    hook_handles = []

    for layer_id, target in targets:
        stats[layer_id] = {
            "k_max": 0.0,
            "v_max": 0.0,
            "k_head_max": None,
            "v_head_max": None,
            "n_samples": 0,
        }

        if target[0] == "qkv_merged":
            _, qkv_module, q_size, kv_size, num_kv_heads, head_dim = target

            def make_hook(lid: int, qs: int, ks: int, nkh: int | None, hd: int | None):
                def hook(_module, _input, output):
                    # QKVParallelLinear returns (tensor, bias) — unwrap if tuple
                    out = output[0] if isinstance(output, tuple) else output
                    # Last dim split: [q_size, kv_size, kv_size]
                    _q, k, v = out.split([qs, ks, ks], dim=-1)
                    update_projection_stats(stats[lid], "k", k, nkh, hd)
                    update_projection_stats(stats[lid], "v", v, nkh, hd)
                    stats[lid]["n_samples"] += 1
                return hook

            hook_handles.append(
                qkv_module.register_forward_hook(
                    make_hook(layer_id, q_size, kv_size, num_kv_heads, head_dim)
                )
            )

        elif target[0] == "split":
            _, k_proj, v_proj, num_kv_heads, head_dim = target

            def make_hook_k(lid: int, nkh: int | None, hd: int | None):
                def hook(_module, _input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    update_projection_stats(stats[lid], "k", out, nkh, hd)
                    stats[lid]["n_samples"] += 1
                return hook

            def make_hook_v(lid: int, nkh: int | None, hd: int | None):
                def hook(_module, _input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    update_projection_stats(stats[lid], "v", out, nkh, hd)
                return hook

            hook_handles.append(k_proj.register_forward_hook(make_hook_k(layer_id, num_kv_heads, head_dim)))
            hook_handles.append(v_proj.register_forward_hook(make_hook_v(layer_id, num_kv_heads, head_dim)))

    return hook_handles


def run_calibration(args) -> dict:
    """Run BF16 forward over calib data, return stats dict.

    Output dict shape: {layer_id (int): {"k_max": float, "v_max": float, "n_samples": int}}
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[calib] loading model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = args.truncation_side
    print(
        f"[calib] tokenizer truncation_side={tokenizer.truncation_side} "
        f"(was {original_truncation_side})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # SALA hard-asserts this
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    print("[calib] introspecting attention layers...")
    targets = find_attention_layers(model)
    print(f"[calib] found {len(targets)} dense attention layers needing KV calibration")
    if not targets:
        raise RuntimeError("no calibration targets found — model architecture mismatch?")

    stats: dict[int, dict] = {}
    handles = attach_hooks(targets, stats)

    print(f"[calib] reading calibration samples from {args.calib_jsonl}")
    samples = []
    with open(args.calib_jsonl) as f:
        for line_num, line in enumerate(f):
            if line_num >= args.num_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[calib] skip line {line_num}: {e}", file=sys.stderr)
                continue

    print(f"[calib] loaded {len(samples)} calibration samples")

    print(f"[calib] running forward (max_len={args.max_len})...")
    ran_forwards = 0
    skipped_no_prompt = 0
    forward_failures = 0
    for i, sample in enumerate(samples):
        # Support multiple JSONL formats
        prompt = (
            sample.get("instruction")
            or sample.get("input")
            or sample.get("question")
            or sample.get("prompt")
            or sample.get("text")
            or (sample.get("messages")[-1].get("content") if sample.get("messages") else None)
        )
        if not prompt:
            skipped_no_prompt += 1
            print(f"  [{i+1:>3}/{len(samples)}] SKIP (no prompt field)", file=sys.stderr)
            continue

        ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_len,
        ).input_ids
        if torch.cuda.is_available():
            ids = ids.cuda()

        with torch.no_grad():
            try:
                model(ids)
                ran_forwards += 1
            except Exception as e:
                forward_failures += 1
                print(f"  [{i+1:>3}/{len(samples)}] forward failed: {e}", file=sys.stderr)
                continue

        if (i + 1) % 8 == 0 or i + 1 == len(samples):
            print(f"  [{i+1:>3}/{len(samples)}] done", flush=True)

    for h in handles:
        h.remove()

    print(
        f"[calib] effective forwards={ran_forwards} "
        f"skipped_no_prompt={skipped_no_prompt} forward_failures={forward_failures}"
    )
    if ran_forwards == 0:
        raise RuntimeError(
            "no calibration sample was executed; check JSONL prompt fields. "
            "SOAR perf_public_set uses the `question` field."
        )

    unvisited = [lid for lid, s in stats.items() if s["n_samples"] == 0]
    if unvisited:
        raise RuntimeError(
            "calibration hooks did not fire for dense layers: "
            + ", ".join(map(str, unvisited[:16]))
            + (" ..." if len(unvisited) > 16 else "")
        )

    tokenizer.truncation_side = original_truncation_side
    return stats


def write_scales(args, stats: dict):
    """Write scales to standalone safetensors + JSON report.

    Safetensors layout (for codex inject step):
      model.layers.{layer_id}.self_attn.attn.k_scale -> tensor(scalar or per-head, fp32)
      model.layers.{layer_id}.self_attn.attn.v_scale -> tensor(scalar or per-head, fp32)

    Codex can then merge into the quantized model's safetensors via
    --inject mode (which rewrites model.safetensors.index.json).
    """
    from safetensors.torch import save_file

    safe_max = args.fp8_safe_max
    tensors = {}
    report = {
        "fp8_safe_max": safe_max,
        "fp8_finfo": {
            "e4m3fn_max": FP8_E4M3_TORCH_MAX,
            "e5m2_max": FP8_E5M2_TORCH_MAX,
        },
        "scale_formula": "cache stores kv / scale; attention receives scale for dequant",
        "output_tensor_granularity": (
            "per_head_when_available" if args.emit_per_head_scales else "scalar"
        ),
        "truncation_side": args.truncation_side,
        "headroom_vs_e4m3fn_max": FP8_E4M3_TORCH_MAX / safe_max if safe_max > 0 else None,
        "num_layers": len(stats),
        "layers": {},
    }
    for lid, s in sorted(stats.items()):
        k_tensor_values, k_head_report, k_granularity = build_scale_tensor_values(
            s, "k", safe_max, args.emit_per_head_scales
        )
        v_tensor_values, v_head_report, v_granularity = build_scale_tensor_values(
            s, "v", safe_max, args.emit_per_head_scales
        )
        k_scale = max(k_tensor_values)
        v_scale = max(v_tensor_values)
        # Inject under the REAL SGLang param path: the RadixAttention module is at
        # self_attn.attn, so its k_scale/v_scale params are named
        # model.layers.{i}.self_attn.attn.k_scale. Naming the tensors this way means
        # the plain `params_dict[name]` loader path finds them directly.
        tensors[f"model.layers.{lid}.self_attn.attn.k_scale"] = torch.tensor(k_tensor_values, dtype=torch.float32)
        tensors[f"model.layers.{lid}.self_attn.attn.v_scale"] = torch.tensor(v_tensor_values, dtype=torch.float32)
        report["layers"][str(lid)] = {
            "k_max_abs": s["k_max"],
            "v_max_abs": s["v_max"],
            "k_scale": k_scale,
            "v_scale": v_scale,
            "k_tensor_granularity": k_granularity,
            "v_tensor_granularity": v_granularity,
            "k_tensor_shape": [len(k_tensor_values)],
            "v_tensor_shape": [len(v_tensor_values)],
            "n_samples": s["n_samples"],
        }
        if k_head_report is not None:
            report["layers"][str(lid)]["k_per_head"] = k_head_report
        if v_head_report is not None:
            report["layers"][str(lid)]["v_per_head"] = v_head_report

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output)
    print(f"[calib] scales written to {args.output} ({len(tensors)} tensors)")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[calib] report written to {args.report}")

    # Print summary table to stdout
    print()
    print(
        f"{'layer':>6} {'k_max':>10} {'v_max':>10} "
        f"{'k_scale':>10} {'v_scale':>10} {'k_med_util':>10} {'v_med_util':>10}"
    )
    for lid in sorted(stats.keys()):
        s = stats[lid]
        k_tensor_values, k_head_report, _ = build_scale_tensor_values(
            s, "k", safe_max, args.emit_per_head_scales
        )
        v_tensor_values, v_head_report, _ = build_scale_tensor_values(
            s, "v", safe_max, args.emit_per_head_scales
        )
        k_scale = max(k_tensor_values)
        v_scale = max(v_tensor_values)
        k_med_util = (
            k_head_report["median_head_utilization_under_scalar"]
            if k_head_report is not None
            else None
        )
        v_med_util = (
            v_head_report["median_head_utilization_under_scalar"]
            if v_head_report is not None
            else None
        )
        print(
            f"{lid:>6} {s['k_max']:>10.3f} {s['v_max']:>10.3f} "
            f"{k_scale:>10.4f} {v_scale:>10.4f} "
            f"{k_med_util if k_med_util is not None else float('nan'):>10.3f} "
            f"{v_med_util if v_med_util is not None else float('nan'):>10.3f}"
        )


def inject_scales(args):
    """Inject pre-computed scales into a quantized model's safetensors.

    Rewrites model.safetensors.index.json to map new tensor names to the
    first shard. Idempotent (re-running with same scales is no-op).
    """
    from safetensors.torch import load_file, save_file

    scales_path = Path(args.scales)
    if not scales_path.is_file():
        print(f"[inject] scales file not found: {scales_path}", file=sys.stderr)
        return 2

    scales = load_file(scales_path)
    print(f"[inject] loaded {len(scales)} scale tensors from {scales_path}")

    quantized_dir = Path(args.quantized_model)
    if not quantized_dir.is_dir():
        print(f"[inject] quantized model dir not found: {quantized_dir}", file=sys.stderr)
        return 2

    shards = sorted(quantized_dir.glob("*.safetensors"))
    if not shards:
        print(f"[inject] no .safetensors found in {quantized_dir}", file=sys.stderr)
        return 2

    # Merge scales into the FIRST shard (simplest; alternative would be to
    # write a new shard, but index rewrite is easier for one shard).
    first_shard = shards[0]
    print(f"[inject] merging scales into {first_shard.name}")
    existing = load_file(first_shard)
    overlap = set(existing.keys()) & set(scales.keys())
    if overlap:
        print(f"[inject] WARNING: {len(overlap)} tensor names already present, overwriting:")
        for k in sorted(overlap)[:5]:
            print(f"    - {k}")
        if len(overlap) > 5:
            print(f"    ... and {len(overlap) - 5} more")
    existing.update(scales)
    save_file(existing, first_shard)
    print(f"[inject] {first_shard.name} now has {len(existing)} tensors")

    # Update index.json
    index_path = quantized_dir / "model.safetensors.index.json"
    if index_path.is_file():
        idx = json.loads(index_path.read_text())
        added = 0
        for k in scales:
            if k not in idx["weight_map"]:
                added += 1
            idx["weight_map"][k] = first_shard.name
        index_path.write_text(json.dumps(idx, indent=2))
        print(f"[inject] updated {index_path.name}: +{added} new tensor entries")
    else:
        print(f"[inject] no index.json at {index_path} — model is single-shard, OK")

    print("[inject] done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inject", action="store_true",
                    help="Inject mode: read previously-saved scales and merge into a quantized model checkpoint.")

    # Calibrate mode args
    ap.add_argument("--model", help="Path to BF16 model dir (for calibration).")
    ap.add_argument("--calib-jsonl", help="Path to calibration JSONL (perf_public_set or similar).")
    ap.add_argument("--num-samples", type=int, default=64, help="Number of calibration samples.")
    ap.add_argument("--max-len", type=int, default=8192, help="Max input length per sample.")
    ap.add_argument("--truncation-side", choices=("left", "right"), default="left",
                    help="Tokenizer truncation side for long prompts. Default left keeps the SOAR task tail.")
    ap.add_argument("--output", help="Where to write standalone scales .safetensors.")
    ap.add_argument("--report", help="Optional: where to write per-layer JSON report.")
    ap.add_argument("--fp8-safe-max", type=float, default=FP8_E4M3_SAFE_MAX,
                    help=f"Target fp8 safe-max value (default {FP8_E4M3_SAFE_MAX} for "
                         f"the e4m3 HP224 path; torch e4m3 max is "
                         f"{FP8_E4M3_TORCH_MAX}).")
    ap.add_argument("--emit-per-head-scales", action="store_true",
                    help="Write rank-1 per-head k/v scale tensors when head stats are available. "
                         "Requires a runtime that preserves per-head FP8KV scales.")

    # Inject mode args
    ap.add_argument("--scales", help="Inject mode: path to standalone scales .safetensors.")
    ap.add_argument("--quantized-model", help="Inject mode: path to quantized model dir.")

    args = ap.parse_args()

    if args.inject:
        if not args.scales or not args.quantized_model:
            ap.error("--inject requires --scales and --quantized-model")
        return inject_scales(args)
    else:
        required = ["model", "calib_jsonl", "output"]
        missing = [k for k in required if not getattr(args, k.replace("-", "_"))]
        if missing:
            ap.error(f"calibrate mode requires --{', --'.join(missing)}")
        stats = run_calibration(args)
        write_scales(args, stats)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
