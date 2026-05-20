"""Inspect GPTQ Marlin safetensors layout (qzeros / g_idx / scales / qweight)
for a quantized MiniCPM-SALA model directory.

Purpose
-------

Active hypothesis (handoff/README.md §5): GPTQModel-written safetensors
differ from the RTN-scalefix script's hand-written layout in a way SGLang's
gptq_marlin loader silently misreads. Most likely the `qzeros` encoding —
for symmetric uint4b8 the dequant formula is `(q_unsigned - qzero) * scale`,
so qzero MUST equal 8 (the bias). If GPTQModel writes 0 or 7, every
weight gets a global additive bias and the model outputs garbage despite
loading without errors (matching v17/v18/v19's acc=0 symptom).

Usage
-----

    python3 handoff/diagnose_qzeros.py <quantized_model_dir> [more dirs...]

Compares multiple dirs side-by-side if more than one given.

What to look for
----------------

For a quantized layer (e.g. model.layers.0.mlp.gate_proj), expect:

    qweight    int32      (in_features // 8, out_features)
    scales     float16    (num_groups, out_features)
    qzeros     int32      (num_groups, out_features // 8)
        unique values should contain 8 only (for sym uint4b8, packed)
        i.e. each 4-bit slot is the bit pattern 0b1000 = 8
        packed into int32: 0x88888888 = 2290649224
    g_idx      int32      (in_features,)
        first 128 values all 0, next 128 all 1, ... (group_size=128)

For RTN-scalefix the values match exactly. For GPTQModel they MAY not.

This script does not need a GPU — it only reads tensor headers and a
handful of values per layer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "this script needs `safetensors` (pip install safetensors)\n"
    )
    sys.exit(1)


def find_shard_for_key(root: Path, key: str) -> Path | None:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        rel = index.get("weight_map", {}).get(key)
        if rel:
            return root / rel
        return None
    shards = sorted(root.glob("model-*.safetensors"))
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            if key in set(f.keys()):
                return shard
    return None


def dump_module(root: Path, layer_idx: int, module: str) -> None:
    base = f"model.layers.{layer_idx}.{module}"
    suffixes = ("qweight", "scales", "qzeros", "g_idx", "weight")
    print(f"--- {root.name}  layer {layer_idx}  {module} ---")
    for suffix in suffixes:
        key = f"{base}.{suffix}"
        shard = find_shard_for_key(root, key)
        if shard is None:
            print(f"  {key:62s}  MISSING")
            continue
        with safe_open(str(shard), framework="pt") as f:
            tensor = f.get_tensor(key)
        extra = ""
        if suffix == "qzeros":
            unique = tensor.flatten().unique().tolist()
            unique = unique[:6]
            # 0x88888888 == 2290649224 means each 4-bit slot is 8
            extra = f"  unique[:6]={unique}  (sym uint4b8 expects 2290649224)"
        elif suffix == "g_idx":
            head = tensor[:5].tolist() if tensor.numel() >= 5 else tensor.tolist()
            tail = tensor[-5:].tolist() if tensor.numel() >= 5 else []
            extra = f"  head={head}  tail={tail}"
        elif suffix == "scales":
            extra = f"  abs.mean={tensor.float().abs().mean().item():.6e}"
        elif suffix == "qweight":
            extra = f"  abs.mean={tensor.float().abs().mean().item():.2e}"
        elif suffix == "weight":
            extra = f"  abs.mean={tensor.float().abs().mean().item():.4e}"
        print(f"  {key:62s}  {str(tensor.dtype):14s}  {tuple(tensor.shape)}{extra}")


def dump_root(root: Path) -> None:
    quant_cfg_path = root / "quantize_config.json"
    if quant_cfg_path.exists():
        with quant_cfg_path.open() as f:
            qc = json.load(f)
        print(f"=== {root}  (quantize_config) ===")
        for key in ("quant_method", "bits", "group_size", "sym", "desc_act",
                    "lm_head", "format", "version"):
            if key in qc:
                print(f"  {key} = {qc[key]}")
        if "dynamic" in qc:
            print(f"  dynamic = {qc['dynamic']}")
    else:
        cfg_path = root / "config.json"
        if cfg_path.exists():
            with cfg_path.open() as f:
                cfg = json.load(f)
            qc = cfg.get("quantization_config", {})
            print(f"=== {root}  (config.quantization_config) ===")
            for key in ("quant_method", "bits", "group_size", "sym",
                        "desc_act", "format", "version"):
                if key in qc:
                    print(f"  {key} = {qc[key]}")

    for module in ("mlp.gate_proj", "self_attn.q_proj"):
        dump_module(root, layer_idx=0, module=module)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in sys.argv[1:]:
        root = Path(arg)
        if not root.is_dir():
            print(f"skipping {arg!r}: not a directory", file=sys.stderr)
            continue
        dump_root(root)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
