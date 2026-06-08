#!/usr/bin/env python3
"""Strict MiniCPM Lightning Q/K RMSNorm fusion probe.

MiniCPM Lightning currently runs:

    q = q_norm(q.reshape(-1, head_dim))
    k = k_norm(k.reshape(-1, head_dim))

SGLang has a generic apply_qk_norm helper that may dispatch to the JIT fused
in-place QKNorm kernel. That kernel is a performance candidate, but it is not a
safe submit change unless it reproduces the MiniCPM path tightly enough for the
competition's no-accuracy-drop invariant.

Run this on the GPU server before considering a QKNorm-fusion package. By
default it requires bitwise equality against the current two-RMSNorm path.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if PYTHON_DIR.is_dir():
    sys.path.insert(0, str(PYTHON_DIR))

DEFAULT_NUM_HEADS = 16
DEFAULT_NUM_KV_HEADS = 16
DEFAULT_HEAD_DIM = 64
DEFAULT_EPS = 1e-6


@dataclass
class ProbeRow:
    dtype: str
    tokens: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    fused_available: bool
    q_max_abs: float
    k_max_abs: float
    q_mean_abs: float
    k_mean_abs: float
    q_bitwise: bool
    k_bitwise: bool

    @property
    def bitwise(self) -> bool:
        return self.q_bitwise and self.k_bitwise

    @property
    def max_abs(self) -> float:
        return max(self.q_max_abs, self.k_max_abs)


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def nested_get(cfg: dict, *keys: str) -> object | None:
    cur: object = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_int(cfg: dict, *paths: tuple[str, ...]) -> int | None:
    for path in paths:
        value = nested_get(cfg, *path)
        if isinstance(value, int):
            return value
    return None


def first_float(cfg: dict, *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value = nested_get(cfg, *path)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_minicpm_lightning_shape(config_path: Path) -> dict[str, int | float]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return {
        "num_heads": first_int(
            cfg,
            ("lightning_nh",),
            ("lightning", "nh"),
            ("lightning", "num_heads"),
        ),
        "num_kv_heads": first_int(
            cfg,
            ("lightning_nkv",),
            ("lightning", "nkv"),
            ("lightning", "num_kv_heads"),
        ),
        "head_dim": first_int(
            cfg,
            ("lightning_head_dim",),
            ("lightning", "head_dim"),
        ),
        "eps": first_float(cfg, ("rms_norm_eps",), ("lightning", "rms_norm_eps")),
    }


def make_norm(head_dim: int, eps: float, dtype: torch.dtype, device: str) -> RMSNorm:
    from sglang.srt.layers.layernorm import RMSNorm

    norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    # Use nontrivial but stable weights. This catches weight-load and dtype
    # behavior without making the random range pathological.
    norm.weight.data.uniform_(0.25, 1.75)
    return norm


@torch.inference_mode()
def run_case(
    *,
    tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    eps: float,
    seed: int,
    device: str,
) -> ProbeRow:
    from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm
    from sglang.srt.models.utils import apply_qk_norm

    torch.manual_seed(seed + tokens + head_dim)
    q_norm = make_norm(head_dim, eps, dtype, device)
    k_norm = make_norm(head_dim, eps, dtype, device)
    q = torch.randn(tokens, num_heads * head_dim, device=device, dtype=dtype)
    k = torch.randn(tokens, num_kv_heads * head_dim, device=device, dtype=dtype)

    q_ref = q_norm(q.reshape(-1, head_dim)).view_as(q)
    k_ref = k_norm(k.reshape(-1, head_dim)).view_as(k)

    q_fused = q.clone()
    k_fused = k.clone()
    fused_available = can_use_fused_inplace_qknorm(head_dim, dtype)
    q_fused, k_fused = apply_qk_norm(
        q=q_fused,
        k=k_fused,
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        allow_inplace=True,
    )

    q_diff = (q_ref.float() - q_fused.float()).abs()
    k_diff = (k_ref.float() - k_fused.float()).abs()
    return ProbeRow(
        dtype=str(dtype).replace("torch.", ""),
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        fused_available=fused_available,
        q_max_abs=float(q_diff.max().item()),
        k_max_abs=float(k_diff.max().item()),
        q_mean_abs=float(q_diff.mean().item()),
        k_mean_abs=float(k_diff.mean().item()),
        q_bitwise=bool(torch.equal(q_ref, q_fused)),
        k_bitwise=bool(torch.equal(k_ref, k_fused)),
    )


def iter_rows(args: argparse.Namespace) -> Iterable[ProbeRow]:
    dtype = dtype_from_name(args.dtype)
    for tokens in parse_ints(args.tokens):
        yield run_case(
            tokens=tokens,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            eps=args.eps,
            seed=args.seed,
            device=args.device,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16"])
    ap.add_argument("--tokens", default="1,2,7,16,64,513,4096")
    ap.add_argument(
        "--model-config",
        type=Path,
        help=(
            "MiniCPM-SALA config.json. When provided, lightning_nh, "
            "lightning_nkv, lightning_head_dim, and rms_norm_eps are used "
            "unless the corresponding CLI flag is set."
        ),
    )
    ap.add_argument("--num-heads", type=int)
    ap.add_argument("--num-kv-heads", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--eps", type=float)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--max-abs", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Exit 0 with a skip record if CUDA is unavailable.",
    )
    args = ap.parse_args()

    config_shape: dict[str, int | float | None] = {}
    if args.model_config is not None:
        config_shape = load_minicpm_lightning_shape(args.model_config)

    args.num_heads = args.num_heads or int(
        config_shape.get("num_heads") or DEFAULT_NUM_HEADS
    )
    args.num_kv_heads = args.num_kv_heads or int(
        config_shape.get("num_kv_heads") or DEFAULT_NUM_KV_HEADS
    )
    args.head_dim = args.head_dim or int(config_shape.get("head_dim") or DEFAULT_HEAD_DIM)
    args.eps = args.eps if args.eps is not None else float(
        config_shape.get("eps") or DEFAULT_EPS
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        message = {
            "status": "skipped",
            "reason": "CUDA is unavailable",
            "device": args.device,
        }
        print(json.dumps(message, indent=2) if args.json else message["reason"])
        return 0 if args.allow_no_cuda else 2

    rows = list(iter_rows(args))
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print(
            "dtype,tokens,heads,kv_heads,head_dim,fused_available,"
            "q_max_abs,k_max_abs,q_mean_abs,k_mean_abs,q_bitwise,k_bitwise"
        )
        for row in rows:
            print(
                f"{row.dtype},{row.tokens},{row.num_heads},{row.num_kv_heads},"
                f"{row.head_dim},{int(row.fused_available)},"
                f"{row.q_max_abs:.8g},{row.k_max_abs:.8g},"
                f"{row.q_mean_abs:.8g},{row.k_mean_abs:.8g},"
                f"{int(row.q_bitwise)},{int(row.k_bitwise)}"
            )

    failed = [row for row in rows if (not row.bitwise) or row.max_abs > args.max_abs]
    if failed:
        print(
            "[qknorm-probe] FAIL: fused path is not strictly identical to the "
            "current MiniCPM two-RMSNorm path.",
            file=sys.stderr,
        )
        return 1
    print("[qknorm-probe] PASS: strict equivalence threshold satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
