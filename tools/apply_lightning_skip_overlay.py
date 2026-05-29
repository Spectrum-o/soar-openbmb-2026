#!/usr/bin/env python3
"""tools/apply_lightning_skip_overlay.py

Post-quant surgery: convert a normally-quantized SALA artifact into a
mixed-precision artifact where selected SALA layer modules are preserved as
BF16 instead of W4A16.

WHY
---
SALA has hybrid attention: some layers are minicpm4 dense attention, some
are Lightning Attention (Gated Delta Rule = linear recurrent). The
recurrent state h_t = g_t * h_{t-1} + ... accumulates errors across time
steps, so quantization errors in lightning-adjacent layers compound badly
on long-context tasks (cwe / niah). MLP-only quant still touches the MLP
that FEEDS into the recurrence, which is suspected of being the dominant
acc loss source.

By default, this tool keeps the original lightning-skip behavior: revert
LIGHTNING layers' MLPs to BF16. Newer full-W4A16 recovery variants can pass
`--modules attn` to restore selected attention q/k/v/o projections instead.

This tool works by:
  1. Reading the source BF16 model's selected layer weights
  2. Writing those into a new "lightning-skip-overlay" safetensors shard
     in the quantized output directory
  3. Updating model.safetensors.index.json so SGLang loader finds BF16 .weight
     tensors at the selected paths
  4. Physically removing old selected quant tensors from their
     safetensors shards, because SGLang iterates physical shard keys rather
     than only index.json entries
  5. Updating quantize_config.json's dynamic field to mark selected modules
     as "skip quantization" (load as BF16 / UnquantizedLinearMethod)

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
        [--layers all-lightning|last4-lightning|23,27,29-31]
        [--modules mlp|attn|mlp,attn]
        [--dry-run]

Idempotent: if the overlay shard already exists, it's overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
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


def parse_layer_selector(
    selector: str, lightning_indices: list[int], total_layers: int
) -> list[int]:
    """Resolve a user layer selector into sorted layer indices.

    Supported selectors:
      - all-lightning / lightning: all lightning indices from config.json
      - lastN-lightning: last N lightning indices, e.g. last4-lightning
      - all: every layer index [0, total_layers)
      - explicit comma/range list: "23,27,29-31"
    """
    value = (selector or "all-lightning").strip().lower()
    if value in ("all-lightning", "lightning"):
        return list(lightning_indices)
    if value == "all":
        return list(range(total_layers))

    match = re.fullmatch(r"last(\d+)-lightning", value)
    if match:
        n = int(match.group(1))
        if n <= 0:
            raise ValueError(f"layer selector {selector!r} must request at least one layer")
        return list(lightning_indices[-n:])

    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid descending layer range {part!r}")
            out.update(range(start, end + 1))
        else:
            out.add(int(part))

    if not out:
        raise ValueError(f"layer selector {selector!r} resolved to no layers")
    bad = [i for i in sorted(out) if i < 0 or i >= total_layers]
    if bad:
        raise ValueError(
            f"layer selector {selector!r} has out-of-range layer(s) {bad}; "
            f"valid range is 0..{total_layers - 1}"
        )
    return sorted(out)


def parse_module_groups(value: str) -> tuple[str, ...]:
    """Parse --modules into a normalized tuple.

    `mlp` restores gate/up/down. `attn` restores q/k/v/o. The helper also
    accepts qkv/o_proj/down aliases for quick ablation variants.
    """
    aliases = {
        "mlp": "mlp",
        "attn": "attn",
        "attention": "attn",
        "qkv": "qkv",
        "o": "o_proj",
        "o_proj": "o_proj",
        "down": "down_proj",
        "down_proj": "down_proj",
    }
    groups: list[str] = []
    for raw in (value or "mlp").split(","):
        item = raw.strip().lower().replace("-", "_")
        if not item:
            continue
        if item not in aliases:
            raise ValueError(
                f"unknown module group {raw!r}; expected one of "
                "mlp, attn, qkv, o_proj, down_proj"
            )
        normalized = aliases[item]
        if normalized not in groups:
            groups.append(normalized)
    if not groups:
        raise ValueError("--modules resolved to an empty module set")
    return tuple(groups)


def tensor_bases_for_layer(layer_idx: int, modules: tuple[str, ...]) -> list[str]:
    bases: list[str] = []
    if "attn" in modules or "qkv" in modules:
        bases.extend(
            f"model.layers.{layer_idx}.self_attn.{proj}"
            for proj in ("q_proj", "k_proj", "v_proj")
        )
    if "attn" in modules or "o_proj" in modules:
        bases.append(f"model.layers.{layer_idx}.self_attn.o_proj")
    if "mlp" in modules:
        bases.extend(
            f"model.layers.{layer_idx}.mlp.{proj}"
            for proj in ("gate_proj", "up_proj", "down_proj")
        )
    elif "down_proj" in modules:
        bases.append(f"model.layers.{layer_idx}.mlp.down_proj")
    return bases


def sglang_dynamic_skip_rules_for_layer(
    layer_idx: int, modules: tuple[str, ...]
) -> dict[str, bool]:
    rules: dict[str, bool] = {}
    if "attn" in modules or "qkv" in modules:
        rules[f"-:model.layers.{layer_idx}.self_attn.qkv_proj$"] = True
    if "attn" in modules or "o_proj" in modules:
        rules[f"-:model.layers.{layer_idx}.self_attn.o_proj$"] = True
    if "mlp" in modules:
        rules[f"-:model.layers.{layer_idx}.mlp.gate_up_proj$"] = True
        rules[f"-:model.layers.{layer_idx}.mlp.down_proj$"] = True
    elif "down_proj" in modules:
        rules[f"-:model.layers.{layer_idx}.mlp.down_proj$"] = True
    return rules


def quantized_tensor_names_for_modules(
    layer_indices: list[int], modules: tuple[str, ...]
) -> set[str]:
    """Return all selected quantized/plain tensor names that must disappear.

    SGLang filters safetensors at the shard/file level using the index, then
    iterates every physical key in the selected shard. Therefore the physical
    cleanup must not depend only on names currently present in index.json: a
    stale key that is missing from the index can still be read by SGLang if its
    shard remains selected for any other tensor.
    """
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
        # plain fallback if a prior run or format left it behind
        ".weight",
    )
    names: set[str] = set()
    for layer_idx in layer_indices:
        for base in tensor_bases_for_layer(layer_idx, modules):
            for suffix in suffixes:
                names.add(f"{base}{suffix}")
    return names


def collect_bf16_tensors(
    bf16_dir: Path, layer_indices: list[int], modules: tuple[str, ...]
) -> dict[str, "any"]:  # noqa: F821 (forward ref to torch tensor)
    """Read BF16 source for selected layer weights.

    Returns a dict mapping selected `model.layers.*.*.weight` names to
    torch.Tensor (bfloat16, on CPU).
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
    for layer_idx in layer_indices:
        for base in tensor_bases_for_layer(layer_idx, modules):
            needed.append(f"{base}.weight")

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


def collect_lightning_mlp_tensors(
    bf16_dir: Path, lightning_indices: list[int]
) -> dict[str, "any"]:  # noqa: F821 (forward ref to torch tensor)
    """Backward-compatible wrapper for the original lightning MLP behavior."""
    return collect_bf16_tensors(bf16_dir, lightning_indices, ("mlp",))


def remove_quantized_tensors_from_index(
    quantized_dir: Path, layer_indices: list[int], modules: tuple[str, ...]
) -> set[str]:
    """Remove quantized tensor entries for selected modules from
    model.safetensors.index.json. Returns the set of removed tensor names.

    The caller must also remove these names from the physical safetensors
    shards. SGLang filters shard files via model.safetensors.index.json, but
    once a shard is selected it iterates every physical key in that shard.
    Leaving orphan .qweight/.qzeros/.scales tensors behind will still surface
    them to MiniCPM.load_weights and can trigger KeyError.

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

    removed: set[str] = set()
    for name in quantized_tensor_names_for_modules(layer_indices, modules):
        if name in weight_map:
            del weight_map[name]
            removed.add(name)

    idx["weight_map"] = weight_map
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)

    print(f"[index] removed {len(removed)} quantized tensors for selected modules", flush=True)
    if removed:
        # Show first few for verification
        sample = sorted(removed)[:6]
        print(f"[index] sample removed: {sample}", flush=True)
    return removed


def remove_quantized_lightning_tensors_from_index(
    quantized_dir: Path, lightning_indices: list[int]
) -> set[str]:
    """Backward-compatible wrapper for the original lightning MLP behavior."""
    return remove_quantized_tensors_from_index(quantized_dir, lightning_indices, ("mlp",))


def rewrite_safetensors_without_tensors(
    quantized_dir: Path, tensor_names: set[str]
) -> dict[str, int]:
    """Physically remove tensor_names from all safetensors shards.

    SGLang's safetensors iterator opens each selected shard and yields f.keys()
    directly. Editing only model.safetensors.index.json is therefore not
    enough: orphan quant tensors remain visible if they still exist inside the
    shard file. This rewrites each affected shard in place, one file at a time.
    """
    if not tensor_names:
        print("[rewrite] no tensor names to remove from physical shards", flush=True)
        return {}

    from safetensors import safe_open
    from safetensors.torch import save_file

    removed_by_shard: dict[str, int] = {}
    for shard_path in sorted(quantized_dir.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            to_remove = [name for name in keys if name in tensor_names]
            if not to_remove:
                continue
            kept = {
                name: f.get_tensor(name).contiguous()
                for name in keys
                if name not in tensor_names
            }
            metadata = f.metadata()

        tmp_path = shard_path.with_name(shard_path.name + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        save_file(kept, str(tmp_path), metadata=metadata)
        tmp_path.replace(shard_path)

        removed_by_shard[shard_path.name] = len(to_remove)
        print(
            f"[rewrite] {shard_path.name}: physically removed {len(to_remove)} tensor(s)",
            flush=True,
        )

    # We intentionally do not fail if a removed index entry was already absent
    # from physical shards; idempotent reruns can reach that state.
    total_removed = sum(removed_by_shard.values())
    if total_removed == 0:
        print(
            "[rewrite] no matching physical tensors found; assuming they were already removed",
            flush=True,
        )
    return removed_by_shard


def write_overlay_shard(
    quantized_dir: Path,
    tensors: dict[str, "any"],  # noqa: F821
    overlay_name: str = "model-lightning-skip-overlay.safetensors",
) -> str:
    """Write a new safetensors file containing the BF16 lightning MLP weights.

    Returns the filename (relative to quantized_dir) that the index should reference.
    """
    from safetensors.torch import save_file

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


def update_quantize_config_dynamic_for_modules(
    quantized_dir: Path, layer_indices: list[int], modules: tuple[str, ...]
) -> None:
    """Add per-layer skip rules to quantize_config.json's dynamic field.

    SGLang's get_dynamic_override reads this and routes matching modules
    through UnquantizedLinearMethod (which looks for .weight in safetensors).

    CRITICAL (2026-05-26 fix): SGLang's MiniCPM implementation uses a fused
    MergedColumnParallelLinear named `gate_up_proj` (minicpm.py:62), NOT
    separate `gate_proj` and `up_proj` Linears. The dynamic rule lookup
    happens at Linear construction time using the SGLang-side prefix, which
    will be `...mlp.gate_up_proj` — so rules ending in `gate_proj$` or
    `up_proj$` will NEVER match. (The pre-2026-05-26 version of this
    function wrote `gate_proj$` / `up_proj$` rules → SGLang fell back to
    GPTQMarlinLinearMethod → tried to load .qweight tensors that this
    overlay had already removed from the index → KeyError on model load.
    The 14 unit tests for this tool did not cover SGLang's fused-Linear
    routing, so the bug stayed latent. Discovered by audit-by-source-grep
    of `python/sglang/srt/models/minicpm.py:62` + `linear.py:463`
    MergedColumnParallelLinear.)

    Fix: write a single `gate_up_proj$` rule (matches SGLang's fused
    Linear) PLUS a `down_proj$` rule (matches RowParallelLinear). The
    safetensors index still has separate `gate_proj.weight` and
    `up_proj.weight` BF16 tensors; SGLang's stacked_params_mapping in
    MiniCPMSALAForCausalLM.load_weights (minicpm.py:611-618) maps both
    tensors to `gate_up_proj` and calls the MergedColumnParallelLinear
    weight_loader with shard_id 0/1 (linear.py:523) — exactly the path
    that's been working for base MLP-only quant via the `-:.*self_attn.*`
    rule (which matches `qkv_proj` via the substring).
    """
    new_skip_rules: dict[str, bool] = {}
    for layer_idx in layer_indices:
        new_skip_rules.update(sglang_dynamic_skip_rules_for_layer(layer_idx, modules))

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


def update_quantize_config_dynamic(
    quantized_dir: Path, lightning_indices: list[int]
) -> None:
    """Backward-compatible wrapper for the original lightning MLP behavior."""
    update_quantize_config_dynamic_for_modules(quantized_dir, lightning_indices, ("mlp",))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    ap.add_argument("--quantized-dir", required=True, type=Path,
                    help="Quantized output dir (will be modified in-place)")
    ap.add_argument("--bf16-dir", required=True, type=Path,
                    help="Source BF16 model dir (read-only, for original weights)")
    ap.add_argument("--layers", default="all-lightning",
                    help="Layer selector: all-lightning, last4-lightning, all, or explicit 23,27,29-31")
    ap.add_argument("--modules", default="mlp",
                    help="Module groups to recover as BF16: mlp, attn, qkv, o_proj, down_proj, or comma list")
    ap.add_argument("--overlay-name", default="model-lightning-skip-overlay.safetensors",
                    help="Overlay safetensors filename written under --quantized-dir")
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
    try:
        selected_indices = parse_layer_selector(args.layers, lightning_indices, total)
        modules = parse_module_groups(args.modules)
    except ValueError as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 2
    print(
        f"[result] selected layers ({len(selected_indices)}): {selected_indices}; "
        f"modules={','.join(modules)}",
        flush=True,
    )

    if args.dry_run:
        n_tensors = sum(
            len(tensor_bases_for_layer(layer_idx, modules))
            for layer_idx in selected_indices
        )
        est_bytes = n_tensors * 4096 * 4096 * 2  # rough BF16 estimate per MLP proj
        print(f"\n[dry-run] would write ~{n_tensors} tensors (~{est_bytes / 1e9:.1f} GB total)")
        print(f"[dry-run] would remove quant tensors for layers {selected_indices}")
        print("[dry-run] would physically rewrite affected safetensors shards")
        n_rules = sum(
            len(sglang_dynamic_skip_rules_for_layer(layer_idx, modules))
            for layer_idx in selected_indices
        )
        print(f"[dry-run] would add {n_rules} dynamic skip rules")
        return 0

    # Late imports (torch + safetensors only needed for actual work)
    try:
        import torch  # noqa: F401
        from safetensors.torch import load_file, save_file  # noqa: F401
    except ImportError as e:
        print(f"[fatal] missing torch or safetensors: {e}", file=sys.stderr)
        return 2

    # 2. Collect BF16 tensors from source
    print(f"\n[2/5] collecting BF16 selected tensors from {bdir.name}", flush=True)
    tensors = collect_bf16_tensors(bdir, selected_indices, modules)

    # 3. Remove quantized lightning MLP entries from index and from the physical
    #    safetensors shards. SGLang iterates physical shard keys after file-level
    #    filtering, so both steps are required.
    print(f"\n[3/5] removing quantized entries for selected modules", flush=True)
    expected_stale_names = quantized_tensor_names_for_modules(selected_indices, modules)
    removed = remove_quantized_tensors_from_index(qdir, selected_indices, modules)
    if expected_stale_names - removed:
        print(
            "[rewrite] also scanning for selected stale physical keys that "
            "were absent from index.json",
            flush=True,
        )
    rewrite_safetensors_without_tensors(qdir, expected_stale_names)

    # 4. Write overlay shard containing BF16 lightning MLP tensors
    print(f"\n[4/5] writing overlay shard with {len(tensors)} BF16 tensors", flush=True)
    overlay_name = write_overlay_shard(qdir, tensors, overlay_name=args.overlay_name)

    # 5. Update index + quantize_config dynamic rules
    print(f"\n[5/5] updating index + quantize_config.json dynamic rules", flush=True)
    update_index_with_overlay(qdir, overlay_name, list(tensors.keys()))
    update_quantize_config_dynamic_for_modules(qdir, selected_indices, modules)

    print(f"\n[done] selective BF16 overlay applied to {qdir.name}")
    print(f"[done] {len(selected_indices)} layers' {','.join(modules)} modules reverted to BF16")
    print(f"[done] Quant-time + load-time alignment: both reference layer indices {selected_indices}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
