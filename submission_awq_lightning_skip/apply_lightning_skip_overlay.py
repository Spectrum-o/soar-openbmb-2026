#!/usr/bin/env python3
"""tools/apply_lightning_skip_overlay.py

Post-quant surgery: convert a normally-quantized SALA artifact into a
mixed-precision artifact where Lightning Attention layers' MLPs are
preserved as BF16 instead of W4A16.

WHY
---
SALA has hybrid attention: some layers are minicpm4 dense attention, some
are Lightning Attention (Gated Delta Rule = linear recurrent). The
recurrent state h_t = g_t * h_{t-1} + ... accumulates errors across time
steps, so quantization errors in lightning-adjacent layers compound badly
on long-context tasks (cwe / niah). MLP-only quant still touches the MLP
that FEEDS into the recurrence, which is suspected of being the dominant
acc loss source.

This tool keeps the existing W4A16 quantization for DENSE layers' MLPs
but reverts LIGHTNING layers' MLPs to BF16, by:
  1. Reading the source BF16 model's lightning-layer MLP weights
  2. Writing those into a new "lightning-skip-overlay" safetensors shard
     in the quantized output directory
  3. Updating model.safetensors.index.json so SGLang loader finds .weight
     at the lightning-layer-mlp paths
  4. Updating quantize_config.json's dynamic field to mark lightning MLPs
     as "skip quantization" (load as BF16 / UnquantizedLinearMethod)

The original .qweight/.qzeros/.scales tensors for lightning MLPs remain in
the shard files (wasted disk space, ~30% of MLP portion) but they're never
read because SGLang's dynamic skip routes those modules through
UnquantizedLinearMethod which looks for .weight.

ALIGNMENT GUARANTEE
-------------------
Quant-time and load-time skip lists must match for this to work. After
running this tool, both the safetensors content (now has BF16 .weight for
lightning MLPs) and quantize_config.json's dynamic field (now skips
lightning MLPs) reference the same set of layer indices.

USAGE
-----
After running the normal 1849-style quant:

    python3 tools/apply_lightning_skip_overlay.py \\
        --quantized-dir /path/to/output-quantized \\
        --bf16-dir      /path/to/MiniCPM-SALA \\
        [--dry-run]

Idempotent: if the overlay shard already exists, it's overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_lightning_layer_indices(config_path: Path) -> tuple[list[int], int]:
    """Read config.json, return (lightning_indices, total_layers).

    SALA's config has `mixer_types`: a list of strings like
        ["minicpm4", "lightning-attn", "minicpm4", "lightning-attn", ...]
    of length num_hidden_layers. Lightning indices are positions where the
    value is the lightning marker (case-insensitive substring match).
    """
    if not config_path.exists():
        raise FileNotFoundError(f"config.json missing at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    mixer_types = cfg.get("mixer_types")
    total_layers = int(cfg.get("num_hidden_layers", 0))

    if not isinstance(mixer_types, list):
        raise RuntimeError(
            "config.json has no `mixer_types` list — cannot identify lightning layers"
        )

    if total_layers and len(mixer_types) != total_layers:
        print(
            f"[warn] mixer_types length {len(mixer_types)} != num_hidden_layers {total_layers}; "
            f"using min",
            file=sys.stderr,
        )

    indices: list[int] = []
    for i, t in enumerate(mixer_types):
        t_lower = str(t).lower()
        # Lightning attention markers we know about
        if "lightning" in t_lower or "gated_delta" in t_lower or "linear_attn" in t_lower:
            indices.append(i)

    return indices, len(mixer_types)


def collect_lightning_mlp_tensors(
    bf16_dir: Path, lightning_indices: list[int]
) -> dict[str, "any"]:  # noqa: F821 (forward ref to torch tensor)
    """Read BF16 source for lightning layer MLP weights.

    Returns a dict mapping `model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight`
    → torch.Tensor (bfloat16, on CPU).
    """
    from safetensors.torch import load_file

    # Read the safetensors index to know which shard has which tensor
    index_path = bf16_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-file safetensors case
        single_file = bf16_dir / "model.safetensors"
        if single_file.exists():
            weight_map = {"_single": str(single_file)}
            mapping: dict[str, str] = {}
        else:
            raise RuntimeError(f"no safetensors files found in {bf16_dir}")
    else:
        with index_path.open("r", encoding="utf-8") as f:
            idx = json.load(f)
        mapping = idx.get("weight_map", {})

    out: dict[str, "any"] = {}  # noqa: F821

    # Group needed tensors by source shard for efficient loading
    needed: list[str] = []
    for layer_idx in lightning_indices:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            needed.append(f"model.layers.{layer_idx}.mlp.{proj}.weight")

    shard_to_tensors: dict[str, list[str]] = {}
    for tname in needed:
        if not mapping:
            shard = "_single"
        else:
            shard = mapping.get(tname)
            if shard is None:
                raise RuntimeError(
                    f"tensor {tname} not in BF16 model.safetensors.index.json — "
                    "is mixer_types wrong, or is the source model not a SALA?"
                )
        shard_to_tensors.setdefault(shard, []).append(tname)

    for shard, tensor_names in shard_to_tensors.items():
        if shard == "_single":
            shard_path = bf16_dir / "model.safetensors"
        else:
            shard_path = bf16_dir / shard
        print(f"[collect] reading {len(tensor_names)} tensors from {shard_path.name}", flush=True)
        shard_tensors = load_file(str(shard_path), device="cpu")
        for tname in tensor_names:
            if tname not in shard_tensors:
                raise RuntimeError(f"{tname} not in {shard_path.name}")
            t = shard_tensors[tname]
            # Verify dtype — SALA source is bf16
            print(f"[collect]   {tname}: shape={tuple(t.shape)} dtype={t.dtype}", flush=True)
            out[tname] = t

    return out


def remove_quantized_lightning_tensors_from_index(
    quantized_dir: Path, lightning_indices: list[int]
) -> set[str]:
    """Remove the quantized tensor entries for lightning MLPs from
    model.safetensors.index.json. Returns the set of removed tensor names.

    The actual tensors stay in the safetensors files (we don't rewrite
    shards) — they just become unreferenced dead data. SGLang load uses
    only the index, so it won't see them.

    Handles BOTH formats:
      - gptq_marlin: .qweight, .qzeros, .scales, .g_idx
      - compressed-tensors (AWQ/W4A16_ASYM): .weight_packed, .weight_scale,
        .weight_zero_point, .weight_shape, .weight_g_idx
    """
    index_path = quantized_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise RuntimeError(
            f"model.safetensors.index.json missing in {quantized_dir} — "
            "quant probably wrote a single safetensors file. Lightning-skip overlay "
            "requires the sharded format. Add post-quant index regeneration."
        )

    with index_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})

    # Quantized-format tensor suffixes (both gptq_marlin and compressed-tensors)
    suffixes = (
        # gptq_marlin format
        ".qweight",
        ".qzeros",
        ".scales",
        ".g_idx",
        ".bias",
        # compressed-tensors format
        ".weight_packed",
        ".weight_scale",
        ".weight_zero_point",
        ".weight_shape",
        ".weight_g_idx",
    )
    projs = ("gate_proj", "up_proj", "down_proj")
    removed: set[str] = set()
    for layer_idx in lightning_indices:
        for proj in projs:
            for suf in suffixes:
                name = f"model.layers.{layer_idx}.mlp.{proj}{suf}"
                if name in weight_map:
                    del weight_map[name]
                    removed.add(name)
            # Also remove the plain .weight (rare but possible if quant chose
            # to retain unquantized weight alongside packed)
            name_weight = f"model.layers.{layer_idx}.mlp.{proj}.weight"
            if name_weight in weight_map:
                del weight_map[name_weight]
                removed.add(name_weight)

    idx["weight_map"] = weight_map
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)

    print(f"[index] removed {len(removed)} quantized tensors for lightning MLPs", flush=True)
    if removed:
        # Show first few for verification
        sample = sorted(removed)[:6]
        print(f"[index] sample removed: {sample}", flush=True)
    return removed


def write_overlay_shard(
    quantized_dir: Path, tensors: dict[str, "any"]  # noqa: F821
) -> str:
    """Write a new safetensors file containing the BF16 lightning MLP weights.

    Returns the filename (relative to quantized_dir) that the index should reference.
    """
    from safetensors.torch import save_file

    overlay_name = "model-lightning-skip-overlay.safetensors"
    overlay_path = quantized_dir / overlay_name

    print(f"[overlay] writing {len(tensors)} BF16 tensors to {overlay_name}", flush=True)
    # Convert to contiguous (safetensors requires it)
    cleaned = {name: t.contiguous() for name, t in tensors.items()}
    save_file(cleaned, str(overlay_path), metadata={"format": "pt"})

    size_mb = overlay_path.stat().st_size / (1024 * 1024)
    print(f"[overlay] wrote {overlay_path.name}, {size_mb:.1f} MB", flush=True)

    return overlay_name


def update_index_with_overlay(
    quantized_dir: Path, overlay_name: str, tensor_names: list[str]
) -> None:
    """Update model.safetensors.index.json to point lightning MLP .weight at the overlay shard."""
    index_path = quantized_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})

    for tname in tensor_names:
        weight_map[tname] = overlay_name

    idx["weight_map"] = weight_map
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    print(f"[index] added {len(tensor_names)} BF16 .weight pointers to {overlay_name}", flush=True)


def update_quantize_config_dynamic(
    quantized_dir: Path, lightning_indices: list[int]
) -> None:
    """Add per-layer-index MLP skip rules to quantize_config.json's dynamic field.

    SGLang's get_dynamic_override reads this and routes matching modules
    through UnquantizedLinearMethod (which looks for .weight in safetensors).
    """
    # Both quantize_config.json AND config.json's quantization_config block
    # are read by SGLang. Update both for safety.
    new_skip_rules: dict[str, bool] = {}
    for layer_idx in lightning_indices:
        new_skip_rules[f"-:model.layers.{layer_idx}.mlp.gate_proj$"] = True
        new_skip_rules[f"-:model.layers.{layer_idx}.mlp.up_proj$"] = True
        new_skip_rules[f"-:model.layers.{layer_idx}.mlp.down_proj$"] = True

    for fname, key_path in (
        ("quantize_config.json", ("dynamic",)),
        ("config.json", ("quantization_config", "dynamic")),
    ):
        path = quantized_dir / fname
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        # Navigate to the dynamic dict
        node = obj
        for k in key_path[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        last = key_path[-1]
        existing = node.get(last, {}) or {}
        existing.update(new_skip_rules)
        node[last] = existing

        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"[config] added {len(new_skip_rules)} skip rules to {fname}::{'.'.join(key_path)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    ap.add_argument("--quantized-dir", required=True, type=Path,
                    help="Quantized output dir (will be modified in-place)")
    ap.add_argument("--bf16-dir", required=True, type=Path,
                    help="Source BF16 model dir (read-only, for original weights)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and exit; don't modify any files")
    args = ap.parse_args()

    qdir = args.quantized_dir.resolve()
    bdir = args.bf16_dir.resolve()

    if not qdir.is_dir():
        print(f"[fatal] quantized-dir {qdir} not a directory", file=sys.stderr)
        return 2
    if not bdir.is_dir():
        print(f"[fatal] bf16-dir {bdir} not a directory", file=sys.stderr)
        return 2

    # 1. Read SALA config to find lightning layer indices
    print(f"[1/5] reading config from {bdir / 'config.json'}", flush=True)
    lightning_indices, total = find_lightning_layer_indices(bdir / "config.json")
    if not lightning_indices:
        print("[result] no lightning layers found in mixer_types; nothing to do", flush=True)
        return 0
    print(f"[result] lightning layers ({len(lightning_indices)} of {total}): {lightning_indices}",
          flush=True)

    if args.dry_run:
        n_tensors = len(lightning_indices) * 3
        est_bytes = n_tensors * 4096 * 4096 * 2  # rough BF16 estimate per MLP proj
        print(f"\n[dry-run] would write ~{n_tensors} tensors (~{est_bytes / 1e9:.1f} GB total)")
        print(f"[dry-run] would remove .qweight/.qzeros/.scales for layers {lightning_indices}")
        print(f"[dry-run] would add {len(lightning_indices) * 3} dynamic skip rules")
        return 0

    # Late imports (torch + safetensors only needed for actual work)
    try:
        import torch  # noqa: F401
        from safetensors.torch import load_file, save_file  # noqa: F401
    except ImportError as e:
        print(f"[fatal] missing torch or safetensors: {e}", file=sys.stderr)
        return 2

    # 2. Collect BF16 tensors from source
    print(f"\n[2/5] collecting BF16 lightning MLP tensors from {bdir.name}", flush=True)
    tensors = collect_lightning_mlp_tensors(bdir, lightning_indices)

    # 3. Remove quantized lightning MLP entries from index (frees the
    #    safetensors namespace for our BF16 versions)
    print(f"\n[3/5] removing quantized entries for lightning MLPs from index", flush=True)
    remove_quantized_lightning_tensors_from_index(qdir, lightning_indices)

    # 4. Write overlay shard containing BF16 lightning MLP tensors
    print(f"\n[4/5] writing overlay shard with {len(tensors)} BF16 tensors", flush=True)
    overlay_name = write_overlay_shard(qdir, tensors)

    # 5. Update index + quantize_config dynamic rules
    print(f"\n[5/5] updating index + quantize_config.json dynamic rules", flush=True)
    update_index_with_overlay(qdir, overlay_name, list(tensors.keys()))
    update_quantize_config_dynamic(qdir, lightning_indices)

    print(f"\n[done] Lightning-skip overlay applied to {qdir.name}")
    print(f"[done] {len(lightning_indices)} lightning layers' MLPs reverted to BF16")
    print(f"[done] Quant-time + load-time alignment: both reference layer indices {lightning_indices}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
