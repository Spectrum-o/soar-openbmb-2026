#!/usr/bin/env python3
"""Path X overlay: keep Q in bf16 (model dtype) when using minicpm_flashinfer.

Why this is needed
------------------
2026-05-25 Path X platform attempt crashed at CUDA-graph capture with:
  flashinfer/jit/attention/modules.py:978
  AssertionError: fp8 tensor core is not supported in fa2 backend

flashinfer's FA2 backend gates FP8 ONLY on `dtype_q`. The comment in
gen_batch_prefill_module explicitly says KV-only quantization is fine in FA2.
Two SALA-fork sites poison Q with the fp8 KV dtype, both fixed here.

Patches applied (idempotent; marker-based)
------------------------------------------
1. minicpm_attention_kernels.py: FlashInferKernel.__init__ uses
   `self.q_data_type = self.kv_cache_dtype` — switch to `model_runner.dtype`,
   matching upstream flashinfer_backend.py:911.

2. minicpm_backend.py: two `q = q.to(self.kv_cache_dtype)` blocks (in
   forward_extend ~L933 and forward_decode ~L1153) cast Q to fp8 for the
   sgl_kernel FA3 path. Skip the cast when attention_kernel_type ==
   "flashinfer" — flashinfer FA2 wants bf16 Q with fp8 KV.

Usage
-----
    python3 apply_pathx_q_bf16_overlay.py <SUBMISSION_DIR>/sglang/python/sglang/srt/layers/attention

Exit code 0 on success (including idempotent re-run). Non-zero on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

KERNEL_MARKER = "# PATH X overlay: Q dtype follows model dtype, not KV cache dtype"
BACKEND_MARKER = "# PATH X overlay: skip Q->fp8 cast when using flashinfer kernel"


def patch_kernels(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if KERNEL_MARKER in text:
        print(f"[pathx] {path.name}: kernel patch already applied")
        return 0

    old = (
        "        # Query data type (same as KV cache dtype, but flashinfer uses separate parameters)\n"
        "        self.q_data_type = self.kv_cache_dtype\n"
    )
    new = (
        "        # PATH X overlay: Q dtype follows model dtype, not KV cache dtype.\n"
        "        # flashinfer FA2's fp8_enabled flag is derived purely from dtype_q;\n"
        "        # using kv_cache_dtype (fp8_*) here would trip\n"
        "        # `AssertionError: fp8 tensor core is not supported in fa2 backend`.\n"
        "        # Upstream flashinfer_backend.py:911 uses model_runner.dtype (bf16).\n"
        "        self.q_data_type = model_runner.dtype\n"
    )

    if old not in text:
        print(
            f"[pathx] ERROR: expected kernels block not found in {path}",
            file=sys.stderr,
        )
        return 1

    if text.count(old) != 1:
        print(
            f"[pathx] WARN: expected 1 occurrence in {path.name}, "
            f"found {text.count(old)}",
            file=sys.stderr,
        )

    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[pathx] patched {path.name}: q_data_type -> model_runner.dtype")
    return 0


def patch_backend(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if BACKEND_MARKER in text:
        print(f"[pathx] {path.name}: backend patch already applied")
        return 0

    cast_block = (
        "            q = q.to(self.kv_cache_dtype)\n"
        "            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None\n"
        "            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None\n"
    )
    new_block = (
        "            # PATH X overlay: skip Q->fp8 cast when using flashinfer kernel.\n"
        "            # flashinfer FA2 wants bf16 Q with fp8 KV (kernel dequants KV\n"
        "            # internally); casting Q to fp8 trips the FA2 fp8_enabled assertion.\n"
        "            if getattr(self, 'attention_kernel_type', None) != 'flashinfer':\n"
        "                q = q.to(self.kv_cache_dtype)\n"
        "                q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None\n"
        "                k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None\n"
    )

    occurrences = text.count(cast_block)
    if occurrences == 0:
        print(
            f"[pathx] ERROR: expected backend cast block not found in {path}",
            file=sys.stderr,
        )
        return 1
    if occurrences != 2:
        print(
            f"[pathx] WARN: expected 2 occurrences in {path.name} "
            f"(forward_extend + forward_decode), found {occurrences}",
            file=sys.stderr,
        )

    patched = text.replace(cast_block, new_block)
    path.write_text(patched, encoding="utf-8")
    print(
        f"[pathx] patched {path.name}: skip q->fp8 cast for flashinfer "
        f"({occurrences} sites)"
    )
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    attn_dir = Path(sys.argv[1])
    if not attn_dir.is_dir():
        print(f"[pathx] attention dir not found: {attn_dir}", file=sys.stderr)
        return 2

    kernels = attn_dir / "minicpm_attention_kernels.py"
    backend = attn_dir / "minicpm_backend.py"
    if not kernels.is_file():
        print(f"[pathx] missing file: {kernels}", file=sys.stderr)
        return 2
    if not backend.is_file():
        print(f"[pathx] missing file: {backend}", file=sys.stderr)
        return 2

    rc = patch_kernels(kernels)
    if rc != 0:
        return rc
    return patch_backend(backend)


if __name__ == "__main__":
    raise SystemExit(main())
