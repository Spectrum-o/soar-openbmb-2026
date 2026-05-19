#!/usr/bin/env python3
"""Symmetric RTN W4A16 quantization for the SOAR MiniCPM-SALA submission.

This produces GPTQ tensors (`qweight`, `scales`, `qzeros`, `g_idx`) plus
`quantize_config.json` configured with `sym=true` so that SGLang's
`--quantization gptq_marlin` accepts the model. SGLang's GPTQ Marlin path
only supports (bits=4, sym=True) -> uint4b8, so the encoding here matches
that convention:

    stored_q (unsigned) = signed_q + 2**(bits-1)
                        = signed_q + 8

The quantization itself is symmetric round-to-nearest (no calibration set,
no external dependencies). For each (group, output_channel) we compute
    scale = max(abs(w)) / 2**(bits-1)
    signed_q = clamp(round(w / scale), -2**(bits-1), 2**(bits-1) - 1)
    stored_q = signed_q + 2**(bits-1)
    stored_zero = 2**(bits-1)              # bias-8, identical for all groups

This is enough to run through Marlin and pass the SOAR correctness gate;
heavier methods (GPTQ with Hessian, AWQ, etc.) can be layered on later.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


LINEAR_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".o_gate.weight",
    ".z_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


def quantize_weight_rtn_symmetric(weight: torch.Tensor, bits: int, group_size: int):
    """Symmetric RTN quantization compatible with GPTQ Marlin (uint4b8)."""
    out_features, in_features_orig = weight.shape
    pack_factor = 32 // bits
    half = 1 << (bits - 1)            # 8 for 4-bit (the bias)
    signed_min = -half                # -8
    signed_max = half - 1             #  7

    if group_size <= 0:
        group_size = in_features_orig

    num_groups = (in_features_orig + group_size - 1) // group_size
    in_features = num_groups * group_size
    if in_features != in_features_orig:
        weight = torch.nn.functional.pad(weight, (0, in_features - in_features_orig))

    # (out, in) -> (out, num_groups, group_size)
    w = weight.float().reshape(out_features, num_groups, group_size)

    # Symmetric scale per (out_channel, group)
    w_absmax = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-10)
    scales = w_absmax / half                  # shape (out, num_groups, 1)

    # Quantize -> signed -> shift to unsigned [0, 2^bits - 1]
    q_signed = torch.clamp(torch.round(w / scales), signed_min, signed_max)
    q = (q_signed + half).to(torch.int32)     # range [0, 2^bits - 1]

    # Repack as (in, out) and pack pack_factor unsigned values per int32 row.
    q = q.reshape(out_features, in_features).t().contiguous()
    qweight = torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32)
    for j in range(pack_factor):
        qweight |= q[j::pack_factor, :] << (bits * j)

    scales_out = scales.squeeze(2).t().contiguous().half()   # (num_groups, out)

    # Symmetric qzeros: every (group, out_channel) stores the bias value `half`.
    zeros_int = torch.full((num_groups, out_features), half, dtype=torch.int32)
    qzeros = torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32)
    for j in range(pack_factor):
        qzeros |= zeros_int[:, j::pack_factor] << (bits * j)

    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    return qweight, scales_out, qzeros, g_idx


def copy_metadata(src: Path, dst: Path) -> None:
    for path in src.iterdir():
        if path.suffix in {".json", ".py", ".model", ".txt"} or path.name == "tokenizer.json":
            if path.name in {"model.safetensors.index.json", "quantize_config.json"}:
                continue
            shutil.copy2(path, dst / path.name)


def shard_names(src: Path) -> list[str]:
    index = src / "model.safetensors.index.json"
    if index.exists():
        with index.open() as f:
            data = json.load(f)
        return sorted(set(data.get("weight_map", {}).values()))
    shards = sorted(p.name for p in src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors weights found under {src}")
    return shards


def convert_shard(src_shard: Path, dst_shard: Path, bits: int, group_size: int):
    print(f"[rtn] converting {src_shard.name}", flush=True)
    state = load_file(str(src_shard))
    new_state: dict[str, torch.Tensor] = {}

    for name, tensor in state.items():
        if not name.endswith(LINEAR_SUFFIXES):
            new_state[name] = tensor.contiguous().to(torch.float16) if tensor.dtype.is_floating_point else tensor
            continue

        weight = tensor.to(torch.float32)
        qweight, scales, qzeros, g_idx = quantize_weight_rtn_symmetric(weight, bits, group_size)

        base = name[: -len(".weight")]
        new_state[f"{base}.qweight"] = qweight.contiguous()
        new_state[f"{base}.scales"] = scales.contiguous()
        new_state[f"{base}.qzeros"] = qzeros.contiguous()
        new_state[f"{base}.g_idx"] = g_idx.contiguous()

    save_file(new_state, str(dst_shard))


def write_quant_configs(input_dir: Path, output_dir: Path, bits: int, group_size: int) -> None:
    quant_cfg = {
        "bits": bits,
        "group_size": group_size,
        "quant_method": "gptq",
        "desc_act": False,
        "sym": True,           # required by SGLang gptq_marlin (uint4b8)
        "lm_head": False,
        "dynamic": {},
    }

    # config.json: ensure torch_dtype=float16 and quantization_config block
    config_path_in = input_dir / "config.json"
    config_path_out = output_dir / "config.json"
    if config_path_in.exists():
        with config_path_in.open() as f:
            config = json.load(f)
    else:
        config = {}
    config["torch_dtype"] = "float16"
    config["quantization_config"] = quant_cfg
    with config_path_out.open("w") as f:
        json.dump(config, f, indent=2)

    # quantize_config.json
    with (output_dir / "quantize_config.json").open("w") as f:
        json.dump(quant_cfg, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Symmetric RTN W4A16 quantization for MiniCPM-SALA -> SGLang gptq_marlin"
    )
    parser.add_argument("--input", required=True, help="Original (BF16) model directory")
    parser.add_argument("--output", required=True, help="Quantized output directory")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bits not in (4, 8):
        raise SystemExit(f"unsupported bits={args.bits}; SGLang gptq_marlin needs 4 or 8")

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = shard_names(input_dir)
    weight_map: dict[str, str] = {}
    for shard in shards:
        src = input_dir / shard
        dst = output_dir / shard
        convert_shard(src, dst, args.bits, args.group_size)
        # rebuild weight_map by reading the just-saved file (cheap)
        for key in load_file(str(dst)).keys():
            weight_map[key] = shard

    index_out = output_dir / "model.safetensors.index.json"
    with index_out.open("w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2)

    copy_metadata(input_dir, output_dir)
    write_quant_configs(input_dir, output_dir, args.bits, args.group_size)
    print(f"[rtn] done. output dir: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
