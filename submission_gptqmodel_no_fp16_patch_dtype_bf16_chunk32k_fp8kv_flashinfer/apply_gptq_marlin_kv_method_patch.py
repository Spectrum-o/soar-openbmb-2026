#!/usr/bin/env python3
"""Patch gptq.py: make GPTQMarlinConfig attach BaseKVCacheMethod for RadixAttention.

Background:
  SOAR official toolkit guidance (soar.openbmb.cn/toolkit) explicitly
  recommends GPTQ W4A16 + Marlin + FP8 KV Cache as the canonical stack.
  But SGLang's current GPTQMarlinConfig.get_quant_method only handles
  LinearBase / FusedMoE, NOT RadixAttention. So when --quantization gptq_marlin
  is combined with --kv-cache-dtype fp8_*, RadixAttention.k_scale stays
  None and FlashAttention rejects the fp8 query without scales.

  This patch (Path D from notes/fp8-kv-cache-investigation) extends
  GPTQMarlinConfig.get_quant_method to also return BaseKVCacheMethod for
  RadixAttention. After patching, layer.k_scale gets initialized to -1.0
  and process_weights_after_loading defaults it to 1.0 (since no fp8 KV
  scales are in the checkpoint).

  This is the SAME pattern fp8 quant_config uses (fp8.py:185-186):
      elif isinstance(layer, RadixAttention):
          return Fp8KVCacheMethod(self)   # subclass of BaseKVCacheMethod

  3-line change, idempotent.

Usage:
  python3 apply_gptq_marlin_kv_method_patch.py /path/to/sglang/python/sglang/srt/layers/quantization/gptq.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PATCH_MARKER = "# FP8 KV cache support for gptq_marlin (Path D from notes/fp8-kv-cache-investigation)"


def apply_patch(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        print(f"[gptq_kv_patch] {path}: already patched (marker present)")
        return 0

    # Target: GPTQMarlinConfig.get_quant_method (around line 364)
    # The method body looks like:
    #     def get_quant_method(
    #         self, layer: torch.nn.Module, prefix: str
    #     ) -> Optional[QuantizeMethodBase]:
    #         # Delay the import to avoid circular dependency
    #         from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    #
    #         if isinstance(layer, FusedMoE):
    #             return GPTQMarlinMoEMethod(self)
    #         return get_linear_quant_method(self, layer, prefix, GPTQMarlinLinearMethod)
    #
    # We insert RadixAttention handling before the FusedMoE check and
    # before the get_linear_quant_method fallback.

    # Find the GPTQMarlinConfig class first, then its get_quant_method
    marker = "        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE\n\n        if isinstance(layer, FusedMoE):\n            return GPTQMarlinMoEMethod(self)\n        return get_linear_quant_method(self, layer, prefix, GPTQMarlinLinearMethod)"

    replacement = """        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

        if isinstance(layer, FusedMoE):
            return GPTQMarlinMoEMethod(self)
        if isinstance(layer, RadixAttention):
            # FP8 KV cache support for gptq_marlin (Path D from notes/fp8-kv-cache-investigation)
            # Mirrors fp8.py:185-186 — attach BaseKVCacheMethod so k_scale/v_scale
            # parameters get created on RadixAttention. With no fp8 scales in the
            # checkpoint, process_weights_after_loading defaults them to 1.0.
            return BaseKVCacheMethod(self)
        return get_linear_quant_method(self, layer, prefix, GPTQMarlinLinearMethod)"""

    if marker not in text:
        print(f"[gptq_kv_patch] ERROR: target marker not found in {path}", file=sys.stderr)
        print("  Expected GPTQMarlinConfig.get_quant_method with current code shape.", file=sys.stderr)
        return 1

    occurrences = text.count(marker)
    if occurrences != 1:
        print(f"[gptq_kv_patch] WARN: expected 1 occurrence, found {occurrences}",
              file=sys.stderr)

    patched = text.replace(marker, replacement, 1)
    path.write_text(patched, encoding="utf-8")
    print(f"[gptq_kv_patch] patched {path}: added RadixAttention → BaseKVCacheMethod")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"[gptq_kv_patch] target file not found: {target}", file=sys.stderr)
        return 2
    return apply_patch(target)


if __name__ == "__main__":
    sys.exit(main())
