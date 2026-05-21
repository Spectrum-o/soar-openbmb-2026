#!/usr/bin/env python3
"""Dequantize ONE Linear weight from a GPTQ artifact and compare to its BF16
reference. CPU-only sanity test that catches "the safetensors loaded but the
math is broken" failures — e.g. wrong qzeros encoding, wrong scales dtype,
wrong qweight bit packing — that fix_qzeros_for_marlin can't see.

If reconstructed weights are within ~few % of BF16 reference per-element,
the artifact is mathematically intact. If diffs are huge, something
deeper than qzeros is broken.

Usage:
    python3 tools/dequant_one_tensor.py \\
        --artifact /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized \\
        --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\
        --module model.layers.0.mlp.gate_proj

    # Optional: sample N rows instead of full matrix (faster, ~same signal)
    python3 tools/dequant_one_tensor.py ... --sample-rows 64

Exit codes:
    0   sanity passed (max abs diff < 1.0, mean abs diff < 0.05)
    1   sanity failed (diffs too large; quant is broken)
    2   CLI / IO error
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def find_shard(model_dir: Path, tensor_name: str) -> Path:
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        wm = idx["weight_map"]
        if tensor_name in wm:
            return model_dir / wm[tensor_name]
    # Single-file fallback
    single = model_dir / "model.safetensors"
    if single.exists():
        return single
    # Glob fallback
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            if tensor_name in f.keys():
                return sf
    raise FileNotFoundError(f"tensor {tensor_name} not found in any shard under {model_dir}")


def load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    shard = find_shard(model_dir, name)
    with safe_open(str(shard), framework="pt") as f:
        return f.get_tensor(name)


def dequant_gptq_sym_uint4b8(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    sample_rows: int | None = None,
) -> torch.Tensor:
    """Dequantize GPTQ sym=True uint4b8 layout back to a [in_features, out_features]
    float matrix. Vectorized in torch.

    Layout (gptqmodel 7.0.0):
        qweight: (in_features // 8, out_features) int32 — 8 nibbles per int32, axis 0
        qzeros:  (num_groups, out_features // 8) int32 — 8 nibbles per int32, axis 1
        scales:  (num_groups, out_features) float16/bfloat16
        g_idx:   (in_features,) int32 — group index per input row

    Dequant per element [in, out]:
        nibble = (qweight[in // 8, out] >> ((in % 8) * 4)) & 0xF       # in [0, 15]
        qzero  = (qzeros[g_idx[in], out // 8] >> ((out % 8) * 4)) & 0xF  # =8 for Marlin sym
        value  = (nibble - qzero) * scales[g_idx[in], out]
    """
    in_features = qweight.shape[0] * 8
    out_features = qweight.shape[1]

    if sample_rows is not None and sample_rows < in_features:
        # Pick contiguous block from somewhere in the middle so we hit
        # multiple groups (group_size is typically 128).
        start = (in_features - sample_rows) // 2
        in_indices = torch.arange(start, start + sample_rows, dtype=torch.int64)
    else:
        in_indices = torch.arange(in_features, dtype=torch.int64)

    n_in = in_indices.shape[0]

    # Unpack qweight: for each in_idx, nibble at (in_idx // 8, all_out_features)
    # >> (in_idx % 8) * 4 & 0xF
    rows = in_indices // 8  # (n_in,)
    shifts = ((in_indices % 8) * 4).to(torch.int32)  # (n_in,)
    qw_int32 = qweight[rows].to(torch.int64)  # (n_in, out_features), int64 to avoid overflow on shift
    nibbles = ((qw_int32 >> shifts.unsqueeze(1).to(torch.int64)) & 0xF).to(torch.int32)  # (n_in, out_features)

    # Unpack qzeros: for each (group_idx, out_idx), nibble at (group_idx, out_idx // 8) >> (out_idx % 8) * 4 & 0xF
    groups_for_inputs = g_idx[in_indices].to(torch.int64)  # (n_in,)
    out_indices = torch.arange(out_features, dtype=torch.int64)
    qz_cols = out_indices // 8  # (out_features,)
    qz_shifts = ((out_indices % 8) * 4).to(torch.int64)  # (out_features,)
    # qzeros[groups_for_inputs, qz_cols] has shape (n_in, out_features) via broadcasting
    qz_int32 = qzeros[groups_for_inputs.unsqueeze(1), qz_cols.unsqueeze(0)].to(torch.int64)  # (n_in, out_features)
    qzero_nibbles = ((qz_int32 >> qz_shifts.unsqueeze(0)) & 0xF).to(torch.int32)  # (n_in, out_features)

    # Scales: scales[group, out]
    scale_vals = scales[groups_for_inputs.unsqueeze(1), out_indices.unsqueeze(0)]  # (n_in, out_features) fp16

    # Dequant: (nibble - qzero) * scale
    dequant = (nibbles - qzero_nibbles).to(scale_vals.dtype) * scale_vals  # (n_in, out_features)

    return dequant, in_indices  # both for caller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="quant artifact dir")
    parser.add_argument("--base", required=True, help="base BF16 model dir")
    parser.add_argument(
        "--module",
        default="model.layers.0.mlp.gate_proj",
        help="module name (without .weight / .qweight / etc)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="sample only this many input rows (default: all)",
    )
    parser.add_argument(
        "--max-abs-fail",
        type=float,
        default=1.0,
        help="fail if max abs diff (single element) >= this",
    )
    parser.add_argument(
        "--mean-abs-fail",
        type=float,
        default=0.05,
        help="fail if mean abs diff >= this",
    )
    args = parser.parse_args()

    artifact = Path(args.artifact)
    base = Path(args.base)
    module = args.module

    print(f"=== dequant sanity: {module} ===")
    print(f"  artifact: {artifact}")
    print(f"  base:     {base}")

    # Load quant tensors
    try:
        qweight = load_tensor(artifact, f"{module}.qweight")
        qzeros = load_tensor(artifact, f"{module}.qzeros")
        scales = load_tensor(artifact, f"{module}.scales")
        g_idx = load_tensor(artifact, f"{module}.g_idx")
    except FileNotFoundError as e:
        print(f"[fail] {e}", file=sys.stderr)
        sys.exit(2)

    print(f"  qweight: {list(qweight.shape)} {qweight.dtype}")
    print(f"  qzeros:  {list(qzeros.shape)} {qzeros.dtype}")
    print(f"  scales:  {list(scales.shape)} {scales.dtype}")
    print(f"  g_idx:   {list(g_idx.shape)} {g_idx.dtype}  unique groups={len(g_idx.unique())}")

    # Quick qzeros sanity
    qz_view = qzeros if qzeros.dtype == torch.int32 else qzeros.view(torch.int32)
    qz_unique = qz_view.flatten().unique()[:10].tolist()
    print(f"  qzeros unique[:10] = {[f'0x{v & 0xFFFFFFFF:08x}' for v in qz_unique]}")
    EXPECTED = -2004318072  # 0x88888888 as int32
    pct_good = (qz_view == EXPECTED).float().mean().item() * 100
    print(f"  qzeros == 0x88888888: {pct_good:.2f}%")

    # Dequant
    dequant, in_indices = dequant_gptq_sym_uint4b8(
        qweight, qzeros, scales, g_idx,
        sample_rows=args.sample_rows,
    )
    print(f"  dequant shape: {list(dequant.shape)}  dtype={dequant.dtype}")

    # Load BF16 reference
    try:
        bf16 = load_tensor(base, f"{module}.weight")
    except FileNotFoundError as e:
        print(f"[fail] base weight not found: {e}", file=sys.stderr)
        sys.exit(2)
    # HF stores [out, in]; our dequant is [in, out]. Transpose to compare.
    bf16_t = bf16.T  # (in_features, out_features)
    bf16_subset = bf16_t[in_indices.long()].to(torch.float32)
    dequant_f32 = dequant.to(torch.float32)

    print(f"  bf16 ref shape: {list(bf16_subset.shape)}")

    # Compare
    diff = dequant_f32 - bf16_subset
    abs_diff = diff.abs()
    bf16_abs = bf16_subset.abs()
    mean_abs = abs_diff.mean().item()
    max_abs = abs_diff.max().item()
    mean_rel = (abs_diff / (bf16_abs + 1e-8)).mean().item()
    p99_abs = abs_diff.flatten().kthvalue(int(0.99 * abs_diff.numel())).values.item()

    bf16_max = bf16_abs.max().item()
    bf16_mean = bf16_abs.mean().item()

    print()
    print(f"  ref BF16 stats:    mean_abs={bf16_mean:.4f}  max_abs={bf16_max:.4f}")
    print(f"  dequant stats:     mean_abs={dequant_f32.abs().mean().item():.4f}  max_abs={dequant_f32.abs().max().item():.4f}")
    print(f"  diff stats:        mean_abs={mean_abs:.4f}  p99_abs={p99_abs:.4f}  max_abs={max_abs:.4f}")
    print(f"  relative diff:     mean(|diff|/|ref|) = {mean_rel:.4f}")

    # Per-group scale, average diff vs scale (a 4-bit quant on [-8,7] has typical error scale/2)
    avg_scale = scales.abs().mean().item()
    print(f"  avg scale:         {avg_scale:.5f}  (expected diff ~ scale/2 = {avg_scale/2:.5f})")

    failed = False
    if max_abs >= args.max_abs_fail:
        print(f"[FAIL] max_abs={max_abs:.4f} >= {args.max_abs_fail}")
        failed = True
    if mean_abs >= args.mean_abs_fail:
        print(f"[FAIL] mean_abs={mean_abs:.4f} >= {args.mean_abs_fail}")
        failed = True

    if failed:
        print("[verdict] DEQUANT BROKEN — math is wrong, artifact will produce garbage")
        sys.exit(1)
    else:
        print("[verdict] OK — dequant matches BF16 reference within expected quant-error bounds")
        sys.exit(0)


if __name__ == "__main__":
    main()
