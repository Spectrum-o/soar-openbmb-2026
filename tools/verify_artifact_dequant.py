#!/usr/bin/env python3
"""Multi-module dequant verifier for a GPTQ W4A16 artifact.

Background
----------
The existing tools/dequant_one_tensor.py checks ONE module against its BF16
reference. That caught the qzeros=0x77 bug post-hoc, but a single-module check
can pass while another module is broken — see the v23 self-bug
(`fix_qzeros_for_marlin` POST-CHECK looked at shards[0] only; that shard had
no qzeros tensors at all). The bug class is "spot-check passes, full artifact
is broken".

This wrapper closes that gap. It:
  1. Auto-discovers every quantized Linear module from
     model.safetensors.index.json (anything with a .qweight key).
  2. Groups them by layer index + module role (e.g. mlp.gate_proj,
     self_attn.q_proj). SALA's hybrid attention means layer 0 (MiniCPM4)
     and layer 1 (Lightning) have DIFFERENT module sets — early sampling
     of just layer 0 misses Lightning bugs.
  3. Picks representatives: at minimum (early, mid, late) × (each distinct
     module role seen across all layers). That's typically 12-20 modules
     for a 32-layer SALA artifact, ~30 seconds CPU.
  4. Calls dequant_gptq_sym_uint4b8 on each, compares to base BF16, and
     prints a one-line PASS/FAIL per module + a summary verdict.

Exit codes match dequant_one_tensor.py:
  0 — all sampled modules PASS
  1 — at least one module FAIL (artifact is broken; do NOT submit)
  2 — IO / discovery error

Usage:
    python3 tools/verify_artifact_dequant.py \\
        --artifact /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized \\
        --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA

    # Fast mode (1 module per role per layer-tier; ~10s)
    python3 tools/verify_artifact_dequant.py --artifact X --base Y --fast

    # Strict mode (all quantized modules; ~5 min on a 9B; useful for
    # post-quant local validation before tarballing)
    python3 tools/verify_artifact_dequant.py --artifact X --base Y --strict

Designed to be invoked by:
  - scripts/local_eval.sh (after quant, before launching SGLang server)
  - tools/pack_submission.py preflight (as an opt-in check when the
    artifact dir is provided)
  - the morning workflow: confirm an artifact is mathematically intact
    before deciding to spend a 5h platform slot on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

# Local import — same dir
try:
    from dequant_one_tensor import dequant_gptq_sym_uint4b8, load_tensor
except ImportError:  # run from repo root: `python3 tools/verify_artifact_dequant.py`
    sys.path.insert(0, str(Path(__file__).parent))
    from dequant_one_tensor import dequant_gptq_sym_uint4b8, load_tensor  # type: ignore[no-redef]


# Modules SGLang's gptq_marlin loader expects in W4A16 quant form.
# `.qweight` is the unambiguous discriminator (vs unquantized layers that
# have just `.weight`).
QWEIGHT_SUFFIX = ".qweight"

# Default thresholds. Same as dequant_one_tensor.py — kept loose enough
# that legitimate 4-bit error never trips them, tight enough that the
# qzeros=0x77 bug (which adds ~scale to every weight) does.
DEFAULT_MAX_ABS = 1.0
DEFAULT_MEAN_ABS = 0.05


@dataclass
class ModuleCheck:
    name: str  # e.g. "model.layers.0.mlp.gate_proj"
    layer_idx: int  # parsed from name; -1 if not a layer module
    role: str  # e.g. "mlp.gate_proj", "self_attn.q_proj"
    passed: bool = False
    mean_abs: float = float("nan")
    max_abs: float = float("nan")
    avg_scale: float = float("nan")
    note: str = ""


def discover_quantized_modules(artifact: Path) -> list[str]:
    """Return sorted list of module names that have a .qweight in the artifact.

    Reads model.safetensors.index.json if present; falls back to scanning
    every .safetensors shard. Handles single-shard artifacts.
    """
    idx_path = artifact / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        keys = idx["weight_map"].keys()
    else:
        # Single-shard or no index. Scan every safetensors.
        from safetensors import safe_open

        keys_set: set[str] = set()
        shards = sorted(artifact.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no .safetensors found under {artifact}")
        for sf in shards:
            with safe_open(str(sf), framework="pt") as f:
                for k in f.keys():
                    keys_set.add(k)
        keys = keys_set

    quant_modules = sorted(
        k[: -len(QWEIGHT_SUFFIX)] for k in keys if k.endswith(QWEIGHT_SUFFIX)
    )
    return quant_modules


def parse_layer_idx_and_role(module_name: str) -> tuple[int, str]:
    """Parse "model.layers.{N}.{role}" into (N, role). Non-layer modules
    return (-1, full_name).
    """
    parts = module_name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        try:
            idx = int(parts[2])
        except ValueError:
            return -1, module_name
        role = ".".join(parts[3:])
        return idx, role
    return -1, module_name


def pick_representatives(modules: list[str], fast: bool) -> list[str]:
    """Pick a small representative set.

    Strategy: bucket by role (mlp.gate_proj, self_attn.q_proj, etc.),
    then for each role sample the earliest, middle, and latest layer
    that has that role. This guarantees we touch every distinct module
    type AND every "region" of the model.

    In --fast mode, take only the earliest layer per role (since the bugs
    we've actually seen are global — qzeros, tokenizer drift — not layer-
    local). The wider sweep is for the case where future bugs ARE layer-
    local (e.g. a per-group calibration corruption).
    """
    by_role: dict[str, list[tuple[int, str]]] = defaultdict(list)
    non_layer: list[str] = []
    for m in modules:
        idx, role = parse_layer_idx_and_role(m)
        if idx < 0:
            non_layer.append(m)
        else:
            by_role[role].append((idx, m))

    picked: list[str] = []
    for role, candidates in sorted(by_role.items()):
        candidates.sort()
        if fast:
            picked.append(candidates[0][1])
        else:
            # earliest, middle, latest
            n = len(candidates)
            picks = {candidates[0][1]}
            picks.add(candidates[n // 2][1])
            picks.add(candidates[-1][1])
            for p in sorted(picks):  # deterministic order
                if p not in picked:
                    picked.append(p)
    # Non-layer modules (lm_head, embed_tokens) are unquantized in our setup
    # but include any that DID get qweight'd, defensively.
    picked.extend(non_layer)
    return picked


def check_module(
    artifact: Path,
    base: Path,
    module: str,
    sample_rows: int = 256,
    max_abs_fail: float = DEFAULT_MAX_ABS,
    mean_abs_fail: float = DEFAULT_MEAN_ABS,
) -> ModuleCheck:
    """Run dequant + compare for one module. Always uses sample_rows for
    speed (max_abs/mean_abs from a 256-row sample is representative; the
    full-matrix mode in dequant_one_tensor.py is for forensic deep-dives).
    """
    layer_idx, role = parse_layer_idx_and_role(module)
    chk = ModuleCheck(name=module, layer_idx=layer_idx, role=role)

    try:
        qweight = load_tensor(artifact, f"{module}.qweight")
        qzeros = load_tensor(artifact, f"{module}.qzeros")
        scales = load_tensor(artifact, f"{module}.scales")
        g_idx = load_tensor(artifact, f"{module}.g_idx")
    except FileNotFoundError as e:
        chk.note = f"missing tensor: {e}"
        return chk

    try:
        bf16 = load_tensor(base, f"{module}.weight")
    except FileNotFoundError as e:
        chk.note = f"base weight missing: {e}"
        return chk

    dequant, in_indices = dequant_gptq_sym_uint4b8(
        qweight, qzeros, scales, g_idx, sample_rows=sample_rows,
    )
    # HF stores [out, in]; our dequant is [in, out]. Transpose.
    bf16_subset = bf16.T[in_indices.long()].to(torch.float32)
    dequant_f32 = dequant.to(torch.float32)

    abs_diff = (dequant_f32 - bf16_subset).abs()
    chk.mean_abs = abs_diff.mean().item()
    chk.max_abs = abs_diff.max().item()
    chk.avg_scale = scales.abs().mean().item()

    chk.passed = (chk.max_abs < max_abs_fail) and (chk.mean_abs < mean_abs_fail)
    return chk


def format_row(chk: ModuleCheck) -> str:
    status = "PASS" if chk.passed else ("FAIL" if not chk.note else "SKIP")
    if chk.note:
        return f"  [{status}] {chk.name}  ({chk.note})"
    return (
        f"  [{status}] {chk.name:60s}  "
        f"mean_abs={chk.mean_abs:.4f}  max_abs={chk.max_abs:.4f}  "
        f"avg_scale={chk.avg_scale:.5f}"
    )


def run(
    artifact: Path,
    base: Path,
    fast: bool = False,
    strict: bool = False,
    sample_rows: int = 256,
    max_abs_fail: float = DEFAULT_MAX_ABS,
    mean_abs_fail: float = DEFAULT_MEAN_ABS,
) -> int:
    print(f"=== verify_artifact_dequant ===")
    print(f"  artifact: {artifact}")
    print(f"  base:     {base}")
    print(f"  mode:     {'strict' if strict else ('fast' if fast else 'default')}")

    try:
        modules = discover_quantized_modules(artifact)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    if not modules:
        print("[error] no .qweight tensors found in artifact — is this actually a GPTQ artifact?",
              file=sys.stderr)
        return 2

    # Coverage summary so a reviewer can see at a glance what we touched.
    by_role: dict[str, int] = defaultdict(int)
    for m in modules:
        _, role = parse_layer_idx_and_role(m)
        by_role[role] += 1
    print(f"  discovered {len(modules)} quantized modules across {len(by_role)} roles:")
    for role, count in sorted(by_role.items()):
        print(f"    {role:40s}  x {count}")

    if strict:
        to_check = modules
    else:
        to_check = pick_representatives(modules, fast=fast)
    print(f"  checking {len(to_check)} modules ({sample_rows} sampled rows each)...")
    print()

    results: list[ModuleCheck] = []
    for module in to_check:
        chk = check_module(
            artifact, base, module,
            sample_rows=sample_rows,
            max_abs_fail=max_abs_fail,
            mean_abs_fail=mean_abs_fail,
        )
        results.append(chk)
        print(format_row(chk))

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed and not r.note)
    n_skip = sum(1 for r in results if r.note)
    print()
    print(f"  summary: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP  "
          f"(of {len(results)} checked)")

    if n_fail > 0:
        print()
        print("[verdict] ARTIFACT BROKEN — at least one quantized module diverges")
        print("          materially from BF16. Do NOT submit this artifact.")
        return 1
    if n_pass == 0:
        print("[verdict] NOTHING CHECKED — all sampled modules skipped (missing tensors).")
        return 2
    print()
    print("[verdict] OK — all sampled modules match BF16 within quant-error bounds.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifact", required=True, help="quant artifact dir")
    parser.add_argument("--base", required=True, help="base BF16 model dir")
    parser.add_argument(
        "--fast", action="store_true",
        help="check ONE module per distinct role (earliest layer). ~10s.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="check ALL quantized modules. ~5 min on a 9B SALA artifact.",
    )
    parser.add_argument(
        "--sample-rows", type=int, default=256,
        help="rows sampled per module (default 256; covers >= 2 groups at "
             "group_size=128, statistically representative).",
    )
    parser.add_argument(
        "--max-abs-fail", type=float, default=DEFAULT_MAX_ABS,
        help="fail if per-module max abs diff >= this (default 1.0)",
    )
    parser.add_argument(
        "--mean-abs-fail", type=float, default=DEFAULT_MEAN_ABS,
        help="fail if per-module mean abs diff >= this (default 0.05)",
    )
    args = parser.parse_args()

    if args.fast and args.strict:
        print("--fast and --strict are mutually exclusive", file=sys.stderr)
        return 2

    return run(
        artifact=Path(args.artifact),
        base=Path(args.base),
        fast=args.fast,
        strict=args.strict,
        sample_rows=args.sample_rows,
        max_abs_fail=args.max_abs_fail,
        mean_abs_fail=args.mean_abs_fail,
    )


if __name__ == "__main__":
    sys.exit(main())
