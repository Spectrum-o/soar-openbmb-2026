#!/usr/bin/env python3
"""Patch minicpm_backend.py: bypass Hopper-only FA fp8 kernel via dequant.

Background:
  When --kv-cache-dtype fp8_e5m2 is enabled, minicpm_backend.py:933 and
  1153 cast Q to fp8, and SGLang's attention kernel expects fp8 KV cache
  with descale. That path requires sgl_kernel's Hopper-only FA fp8
  binary (.../hopper/flash_fwd_launch_template.h:166), which is missing
  on Blackwell SM 12.0 (RTX 6000D / RTX PRO 6000).

  Path Y fix:
    1. Skip Q→fp8 cast (Q stays in original dtype, e.g. bf16).
    2. Immediately after get_kv_buffer, cast fp8 K/V cache to q.dtype.
    3. Standard bf16 FA path runs end-to-end — works on any GPU with
       fp16/bf16 FA support (which Blackwell does have).

  Trade-off vs Path X (FlashInfer SM12.0 fp8):
    - Path Y loses ~50% of fp8 attention bandwidth benefit (no fp8 GEMM
      in attention; KV gets up-cast before compute).
    - But KV pool STORAGE stays fp8 (memory_pool.py:666 — store as uint8).
      That preserves the 2x KV capacity benefit, which is the main
      reason long-context configs need fp8 KV at all.
    - Path Y is much safer: no flashinfer JIT, no SM 12.0 cubin gen,
      no backend swap. Just two minimal source patches.

  Idempotent — checks PATCH_MARKER before applying.

Usage:
  python3 apply_fp8kv_dequant_bypass_patch.py \\
      /path/to/sglang/python/sglang/srt/layers/attention/minicpm_backend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PATCH_MARKER = "# PATH Y BYPASS: skip Q to fp8 cast"

OLD_Q_CAST = """            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None"""

NEW_Q_CAST = """            # PATH Y BYPASS: skip Q to fp8 cast (FA fp8 kernel needs Hopper
            # binary, not on Blackwell SM 12.0). KV gets dequanted to q.dtype
            # immediately after get_kv_buffer below.
            pass"""

OLD_KV_FETCH = """        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )"""

NEW_KV_FETCH = """        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        # PATH Y BYPASS: dequant fp8 K/V to q.dtype so standard bf16 FA path
        # runs end-to-end on Blackwell (no Hopper-only FP8 cubin needed).
        # KV pool storage stays fp8 (2x capacity vs bf16); only the read-path
        # gets up-cast.
        if key_cache.dtype != q.dtype:
            key_cache = key_cache.to(q.dtype)
            value_cache = value_cache.to(q.dtype)"""


def apply_patch(path: Path) -> int:
    text = path.read_text(encoding="utf-8")

    if PATCH_MARKER in text:
        print(f"[fp8kv_dequant_patch] {path}: already patched")
        return 0

    n_q_cast = text.count(OLD_Q_CAST)
    n_kv_fetch = text.count(OLD_KV_FETCH)

    if n_q_cast == 0:
        print(
            f"[fp8kv_dequant_patch] ERROR: Q-cast marker not found in {path}",
            file=sys.stderr,
        )
        print(
            "  Expected the three-line block:\n" + OLD_Q_CAST,
            file=sys.stderr,
        )
        return 1

    if n_kv_fetch == 0:
        print(
            f"[fp8kv_dequant_patch] ERROR: get_kv_buffer marker not found in {path}",
            file=sys.stderr,
        )
        return 1

    if n_q_cast != 2:
        print(
            f"[fp8kv_dequant_patch] WARN: expected 2 Q-cast occurrences (sparse "
            f"+ dense decode path), found {n_q_cast}",
            file=sys.stderr,
        )

    if n_kv_fetch < 2:
        print(
            f"[fp8kv_dequant_patch] WARN: expected at least 2 get_kv_buffer "
            f"occurrences, found {n_kv_fetch}",
            file=sys.stderr,
        )

    patched = text.replace(OLD_Q_CAST, NEW_Q_CAST)
    patched = patched.replace(OLD_KV_FETCH, NEW_KV_FETCH)

    path.write_text(patched, encoding="utf-8")
    print(
        f"[fp8kv_dequant_patch] patched {path}: "
        f"{n_q_cast} Q-cast block(s) bypassed, "
        f"{n_kv_fetch} KV-fetch site(s) get dequant"
    )
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"[fp8kv_dequant_patch] target not found: {target}", file=sys.stderr)
        return 2
    return apply_patch(target)


if __name__ == "__main__":
    sys.exit(main())
