#!/usr/bin/env python3
"""Verify a full-W4A16 GPTQ artifact before serving it with SGLang.

The check is intentionally structural and CPU-only:
  - every target Linear module is either GPTQ-packed or explicitly expected
    to stay BF16;
  - expected BF16 skips have `.weight` and no GPTQ tensor quartet;
  - packed modules have `.qweight/.qzeros/.scales/.g_idx` and no `.weight`;
  - `quantize_config.json` carries the same dynamic skip rule for mixed runs.

This catches the common platform-load failure where GPTQModel saved BF16
weights for a mixed-sensitive module but SGLang still expects GPTQ tensor
names, or the reverse.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from safetensors import safe_open


TARGET_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

GPTQ_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx")

MODULE_ALIASES = {
    "all": TARGET_MODULES,
    "attn": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
    "self_attn": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
    "qkv": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ),
    "q": ("self_attn.q_proj",),
    "k": ("self_attn.k_proj",),
    "v": ("self_attn.v_proj",),
    "o": ("self_attn.o_proj",),
    "attn_o": ("self_attn.o_proj",),
    "self_attn.o_proj": ("self_attn.o_proj",),
    "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "gate": ("mlp.gate_proj",),
    "up": ("mlp.up_proj",),
    "down": ("mlp.down_proj",),
    "mlp.gate_proj": ("mlp.gate_proj",),
    "mlp.up_proj": ("mlp.up_proj",),
    "mlp.down_proj": ("mlp.down_proj",),
}


def parse_layers(spec: str) -> list[int]:
    layers: set[int] = set()
    if not spec.strip():
        return []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                start, end = end, start
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    bad = sorted(layer for layer in layers if layer < 0)
    if bad:
        raise ValueError(f"layer indices must be non-negative: {bad}")
    return sorted(layers)


def parse_modules(spec: str) -> list[str]:
    modules: list[str] = []
    unknown: list[str] = []
    aliases = [part.strip().lower() for part in spec.split(",") if part.strip()]
    if not aliases:
        aliases = ["all"]
    for alias in aliases:
        resolved = MODULE_ALIASES.get(alias)
        if resolved is None:
            unknown.append(alias)
            continue
        for module in resolved:
            if module not in modules:
                modules.append(module)
    if unknown:
        raise ValueError(
            f"unknown module alias(es): {unknown}; known={sorted(MODULE_ALIASES)}"
        )
    return modules


def load_keys(artifact: Path) -> set[str]:
    shards = sorted(artifact.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files found under {artifact}")
    keys: set[str] = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def expected_skip_set(layers: list[int], modules: list[str]) -> set[str]:
    return {f"model.layers.{layer}.{module}" for layer in layers for module in modules}


def infer_num_layers(keys: set[str]) -> int:
    layers: set[int] = set()
    pattern = re.compile(r"^model\.layers\.(\d+)\.")
    for key in keys:
        match = pattern.match(key)
        if match:
            layers.add(int(match.group(1)))
    if not layers:
        raise RuntimeError("could not infer model.layers.N keys from artifact")
    return max(layers) + 1


def has_dynamic_rule(
    quant_cfg: dict,
    layers: list[int],
    modules: list[str],
) -> bool:
    if not layers:
        return True
    dynamic = quant_cfg.get("dynamic")
    if not isinstance(dynamic, dict):
        return False
    layer_alt = "|".join(str(layer) for layer in layers)
    wanted_modules = set(modules)
    for rule in dynamic:
        if not rule.startswith("-:"):
            continue
        regex = rule[2:]
        if f"layers\\.({layer_alt})" not in regex:
            continue
        matched = {
            module
            for module in TARGET_MODULES
            if re.search(regex, f"model.layers.{layers[0]}.{module}")
        }
        if wanted_modules.issubset(matched):
            return True
    return False


def verify(args: argparse.Namespace) -> int:
    artifact = Path(args.artifact)
    if not artifact.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {artifact}")

    layers = parse_layers(args.mixed_skip_layers)
    modules = parse_modules(args.mixed_skip_modules)
    skip = expected_skip_set(layers, modules)
    keys = load_keys(artifact)
    num_layers = infer_num_layers(keys)
    problems: list[str] = []

    for layer in range(num_layers):
        for module in TARGET_MODULES:
            base = f"model.layers.{layer}.{module}"
            weight = f"{base}.weight"
            gptq_keys = {f"{base}.{suffix}" for suffix in GPTQ_SUFFIXES}
            is_skip = base in skip
            present_gptq = sorted(gptq_keys & keys)
            if is_skip:
                if weight not in keys:
                    problems.append(f"{base}: expected BF16 .weight, missing")
                if present_gptq:
                    problems.append(
                        f"{base}: expected BF16 skip but found GPTQ keys {present_gptq}"
                    )
            else:
                missing = sorted(gptq_keys - keys)
                if missing:
                    problems.append(f"{base}: expected GPTQ keys, missing {missing}")
                if weight in keys:
                    problems.append(f"{base}: expected GPTQ but found BF16 .weight")

    quant_path = artifact / "quantize_config.json"
    if not quant_path.is_file():
        problems.append("quantize_config.json missing")
        quant_cfg = {}
    else:
        quant_cfg = json.loads(quant_path.read_text(encoding="utf-8"))
        if quant_cfg.get("bits") != 4:
            problems.append(f"quantize_config bits={quant_cfg.get('bits')!r}, expected 4")
        if quant_cfg.get("group_size") != args.group_size:
            problems.append(
                f"quantize_config group_size={quant_cfg.get('group_size')!r}, "
                f"expected {args.group_size}"
            )
        if quant_cfg.get("quant_method") != "gptq":
            problems.append(
                f"quantize_config quant_method={quant_cfg.get('quant_method')!r}, expected gptq"
            )
        if not quant_cfg.get("sym", False):
            problems.append("quantize_config sym is not true")
        if not has_dynamic_rule(quant_cfg, layers, modules):
            problems.append(
                "quantize_config dynamic does not contain the expected mixed skip rule "
                f"for layers={layers} modules={modules}"
            )

    if problems:
        print("[verify_full_w4a16_artifact] FAIL", file=sys.stderr)
        for problem in problems[:80]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 80:
            print(f"  ... {len(problems) - 80} more", file=sys.stderr)
        return 1

    print("[verify_full_w4a16_artifact] OK")
    print(f"  artifact: {artifact}")
    print(f"  layers:   {num_layers}")
    print(f"  skips:    {len(skip)} target module(s)")
    if skip:
        for item in sorted(skip):
            print(f"    - {item}")
    print(f"  keys:     {len(keys)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--mixed-skip-layers", default="")
    parser.add_argument("--mixed-skip-modules", default="all")
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    return verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
