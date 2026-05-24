#!/usr/bin/env python3
"""Apply the FP8 KV cache fallback patch to a bundled SGLang's minicpm_backend.py.

Background:
  notes/fp8-kv-cache-investigation (2026-05-14) identified that SALA's
  fp8 KV cache fails because layer.k_scale is None unless a quant_config
  with FP8 KV cache support (e.g. fp8 quant_config) is attached. With
  --quantization gptq_marlin (our W4A16 path), gptq_marlin's quant_config
  only handles LinearBase, not RadixAttention, so k_scale stays None.

  This patch implements "Path B" from the investigation: add a default
  scale=1.0 fallback when layer.k_scale is None, so fp8 KV cache works
  without requiring a calibrated checkpoint or fp8 weight quant.

  Two patch sites in minicpm_backend.py:
    - forward_extend (around line 920-935): prefill path
    - forward_decode (around line 1145-1155): decode path

  Both have the same pattern:
    if layer.k_scale is not None:
        ...k_descale = layer.k_scale.expand(...)
    q = q.to(self.kv_cache_dtype)

  Patched to:
    if layer.k_scale is not None:
        ...
    else:
        # FP8 KV cache fallback: use scale=1.0 when no quant_config
        # provided KV cache scales. Crude but works for un-calibrated FP8 KV.
        descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        k_descale = torch.ones(descale_shape, dtype=torch.float32, device=q.device)
        v_descale = torch.ones(descale_shape, dtype=torch.float32, device=q.device)
    q = q.to(self.kv_cache_dtype)

Idempotent: detects the patch marker and skips if already applied.

Usage:
  python3 apply_fp8_kv_fallback_patch.py path/to/sglang/python/sglang/srt/layers/attention/minicpm_backend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PATCH_MARKER = "# FP8 KV cache fallback (Path B from notes/fp8-kv-cache-investigation)"

# The two patch sites have the same pattern (both gated by k_scale check).
# We match the EXACT lines and inject an else-branch.
OLD_PATTERN = """            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)"""

NEW_PATTERN = """            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            else:
                # FP8 KV cache fallback (Path B from notes/fp8-kv-cache-investigation)
                # gptq_marlin quant_config doesn't attach KV cache method to
                # RadixAttention (only handles LinearBase), so layer.k_scale
                # stays None. Default scale=1.0 lets fa3 kernel accept fp8 KV.
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = torch.ones(descale_shape, dtype=torch.float32, device=q.device)
                v_descale = torch.ones(descale_shape, dtype=torch.float32, device=q.device)
            q = q.to(self.kv_cache_dtype)"""


def apply_patch(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        print(f"[fp8_kv_patch] {path}: already patched (marker present), skipping")
        return 0

    occurrences = text.count(OLD_PATTERN)
    if occurrences == 0:
        print(f"[fp8_kv_patch] {path}: ERROR — pattern not found (file out of sync?)",
              file=sys.stderr)
        return 1

    if occurrences != 2:
        print(f"[fp8_kv_patch] {path}: WARN — expected 2 occurrences, found {occurrences}",
              file=sys.stderr)

    patched = text.replace(OLD_PATTERN, NEW_PATTERN)
    path.write_text(patched, encoding="utf-8")
    print(f"[fp8_kv_patch] {path}: patched {occurrences} site(s)")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"[fp8_kv_patch] target file not found: {target}", file=sys.stderr)
        return 2
    return apply_patch(target)


if __name__ == "__main__":
    sys.exit(main())
