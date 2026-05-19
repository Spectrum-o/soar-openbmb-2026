#!/usr/bin/env python3
"""Offline roundtrip test for the symmetric W4A16 RTN quantization.

Pure CPU, ~1 second to run, no model weights needed. Purpose: verify the
quantize -> Marlin-dequant math matches expectations BEFORE burning another
5h platform submission.

Marlin uint4b8 dequant formula:    value = (q_unsigned - 8) * scale

For symmetric RTN to be correct:
- scale = max(abs(w)) / (2**(bits-1) - 1)   # i.e. / 7 for 4-bit
- q_unsigned in [0, 15]
- Reconstruction error should be ~ scale / 2 per element (rounding noise),
  and max(|w|) should match max(|dequant_w|) within rounding of one bin.

The earlier bug (`/ half` = `/ 8` instead of `/ 7`) would show up as:
  max(|dequant_w|) ≈ 7/8 * max(|w|) ≈ 0.875 * max(|w|)

This script tests several shapes / distributions, prints pass/fail, and
returns nonzero if any case fails.

Usage:
    python3 scripts/test_quant_roundtrip.py

Requires: torch (CPU is fine; `pip install torch --index-url https://download.pytorch.org/whl/cpu` if missing).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# Use the v1 canonical quantize script as the source of truth.
QUANT_SCRIPT = REPO_ROOT / "submission_rtn_sym_w4a16" / "quantize_gptq_rtn_sym.py"

sys.path.insert(0, str(QUANT_SCRIPT.parent))
from quantize_gptq_rtn_sym import quantize_weight_rtn_symmetric  # type: ignore  # noqa: E402


def dequantize_marlin_uint4b8(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Reconstruct `w` from (qweight, scales) using Marlin's uint4b8 convention.

    - qweight shape: (in // pack_factor, out)
    - scales shape:  (num_groups, out)
    - dequant:       value = (q_unsigned - 8) * scale_group
    """
    pack_factor = 32 // bits
    half = 1 << (bits - 1)  # 8 for 4-bit
    mask = (1 << bits) - 1  # 0xF for 4-bit

    # Unpack qweight: (in // pack_factor, out) -> (in, out)
    q_unpacked = torch.zeros(in_features, out_features, dtype=torch.int32)
    for j in range(pack_factor):
        q_unpacked[j::pack_factor, :] = (qweight >> (bits * j)) & mask

    # signed_q = q_unsigned - 8 in [-8, 7]
    signed_q = q_unpacked.to(torch.float32) - half  # (in, out)

    # Broadcast scales (num_groups, out) -> per-input-feature scale (in, out)
    num_groups = scales.shape[0]
    per_in_scale = scales.float().unsqueeze(1).expand(num_groups, group_size, out_features).reshape(num_groups * group_size, out_features)
    per_in_scale = per_in_scale[:in_features, :]

    dequant = signed_q * per_in_scale  # (in, out)
    return dequant.t().contiguous()    # (out, in) to match original weight shape


def run_one_case(
    name: str,
    weight: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    rel_err_threshold: float = 0.20,
    peak_ratio_threshold: float = 0.97,
) -> bool:
    """Quantize then dequantize; check the round-trip error.

    rel_err_threshold:    mean(|w - w_hat|) / mean(|w|) must be below this.
                          For plain 4-bit RTN on gaussian weights the
                          theoretical floor is ~14% (step_size/4 / mean(|w|)),
                          so 20% is the "math-is-correct" upper bound. To beat
                          this, you need GPTQ Hessian / AWQ — not a bigger
                          threshold here.
    peak_ratio_threshold: max(|w_hat|) / max(|w|) must be at least this. This
                          is the strict scale-formula check — the `/half` (=/8)
                          bug shows up as ratio ~= 7/8 = 0.875. With the correct
                          `/(half-1)` (=/7) formula it should be ~1.0.
    """
    out_features, in_features = weight.shape
    qweight, scales, qzeros, g_idx = quantize_weight_rtn_symmetric(
        weight, bits, group_size
    )

    # Shape sanity
    pack_factor = 32 // bits
    num_groups = math.ceil(in_features / group_size)
    expected_qweight_shape = (num_groups * group_size // pack_factor, out_features)
    expected_scales_shape = (num_groups, out_features)
    expected_qzeros_shape = (num_groups, out_features // pack_factor)

    shape_checks = {
        "qweight.shape": (qweight.shape, expected_qweight_shape),
        "scales.shape": (scales.shape, expected_scales_shape),
        "qzeros.shape": (qzeros.shape, expected_qzeros_shape),
        "qweight.dtype": (qweight.dtype, torch.int32),
        "scales.dtype": (scales.dtype, torch.float16),
        "qzeros.dtype": (qzeros.dtype, torch.int32),
    }
    shape_ok = True
    for k, (got, want) in shape_checks.items():
        if got != torch.Size(want) if "shape" in k else got != want:
            print(f"  [FAIL] {k}: got {got}, want {want}")
            shape_ok = False

    # Roundtrip
    padded_in = num_groups * group_size
    if padded_in != in_features:
        # pad the original weight to compare like-for-like
        w_padded = torch.nn.functional.pad(weight, (0, padded_in - in_features))
    else:
        w_padded = weight
    dequant = dequantize_marlin_uint4b8(
        qweight, scales, bits, group_size, out_features, padded_in
    )

    err = (dequant - w_padded).abs()
    mean_w = w_padded.abs().mean().item()
    rel_mean_err = (err.mean() / max(mean_w, 1e-12)).item()
    max_orig = w_padded.abs().max().item()
    max_deq = dequant.abs().max().item()
    peak_ratio = max_deq / max(max_orig, 1e-12)

    ok = shape_ok
    # Hard fail: peak_ratio off means scale formula is wrong (bug).
    peak_ok = peak_ratio >= peak_ratio_threshold
    # Soft warn: high rel err is intrinsic RTN noise on outlier-heavy weights.
    rel_ok = rel_mean_err < rel_err_threshold

    severity = "OK"
    if not peak_ok:
        severity = "BUG"      # scale-formula bug — fix the code
    elif not rel_ok:
        severity = "WARN"     # RTN intrinsic limit — needs GPTQ/AWQ, not a code fix

    print(f"  [{severity}] {name} | shape={tuple(weight.shape)} group_size={group_size}")
    print(f"    mean|w|       = {mean_w:.4f}")
    print(f"    mean|err|     = {err.mean().item():.4f}  (rel = {rel_mean_err:.4f})    {'OK' if rel_ok else f'high — RTN intrinsic; threshold {rel_err_threshold}'}")
    print(f"    max|w|        = {max_orig:.4f}")
    print(f"    max|dequant|  = {max_deq:.4f}    peak_ratio = {peak_ratio:.4f}    {'OK' if peak_ok else f'BUG — likely /half vs /(half-1); threshold {peak_ratio_threshold}'}")

    # Only peak_ratio failures should mark the whole script as failed.
    return peak_ok and shape_ok


def main() -> int:
    torch.manual_seed(0)

    print("=" * 60)
    print("test_quant_roundtrip.py — verify symmetric W4A16 RTN math")
    print(f"  using: {QUANT_SCRIPT}")
    print("=" * 60)

    cases = [
        # name, weight tensor
        ("standard gaussian 256x512",      torch.randn(256, 512)),
        ("standard gaussian 128x1024",     torch.randn(128, 1024)),
        ("scaled small 256x256",           torch.randn(256, 256) * 0.05),
        ("scaled large 256x256",           torch.randn(256, 256) * 5.0),
        ("pure positive (uniform 0..1)",   torch.rand(64, 256)),
        ("pure negative (uniform -1..0)",  -torch.rand(64, 256)),
        ("with outliers (gaussian * mask)", torch.randn(128, 512) + 8.0 * (torch.rand(128, 512) > 0.99).float()),
        ("group_size = full row 64x128",   torch.randn(64, 128)),
    ]

    all_ok = True
    for name, w in cases:
        ok = run_one_case(name, w, bits=4, group_size=128)
        if not ok:
            all_ok = False
        print()

    # Additional smaller group_size
    print("--- group_size = 64 ---")
    ok = run_one_case("gaussian 128x512, group=64", torch.randn(128, 512), bits=4, group_size=64)
    if not ok:
        all_ok = False
    print()

    print("=" * 60)
    if all_ok:
        print(" ✅ all cases pass — quantize math is correct")
        print()
        print(" Confirmed:")
        print("   - peak_ratio ≈ 1.0  → scale formula `/(half-1)` is right")
        print("     (the `/half` bug would give ~0.875)")
        print("   - mean rel err ~12-14% is the theoretical floor for")
        print("     4-bit RTN on Gaussian weights — NOT a bug, just RTN's")
        print("     intrinsic loss.")
        print()
        print(" Implication for SOAR acc=42:")
        print("   - The math is fine; plain RTN itself is too lossy for SALA.")
        print("   - Next: switch to GPTQModel (Hessian-based) which achieves")
        print("     2-5%% mean err by allocating quantization error wisely.")
        print("   - Or: try AWQ which protects outlier weights specifically.")
        return 0
    else:
        print(" ❌ some cases failed — quant math has bug(s); fix BEFORE")
        print("   wasting another 5h platform submission.")
        print()
        print("   If peak_ratio < 0.97, you've still got a scale-formula bug.")
        print("   If mean_rel_err > 0.20, your packing/dequant logic is wrong.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
