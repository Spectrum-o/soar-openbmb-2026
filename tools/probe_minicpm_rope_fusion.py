#!/usr/bin/env python3
"""Strict MiniCPM-SALA RoPE no-upcast probe.

MiniCPM currently applies RoPE as:

    q, k = q.float(), k.float()
    q, k = rotary_emb(positions, q, k)
    q, k = q.to(orig_dtype), k.to(orig_dtype)

The possible RoPE fusion removes the explicit Python-side casts and lets the
CUDA RoPE kernel load bf16/fp16 values, compute in float, and store back to the
original dtype. This probe blocks the change unless that candidate is bitwise
identical to the current MiniCPM path for the actual MiniCPM-SALA config.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "python", ROOT / "sglang" / "python"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

FAST_CUDA_ROPE_HEAD_DIMS = {64, 128, 256, 512}
LIGHTNING_MIXERS = {"lightning", "lightning_attn", "lightning-attn"}


@dataclass(frozen=True)
class RopeShape:
    label: str
    mixer_type: str
    layer_count: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rope_scaling: dict[str, Any] | None


@dataclass
class ProbeRow:
    label: str
    mixer_type: str
    dtype: str
    tokens: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
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


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return cfg


def require_int(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if not isinstance(value, int):
        raise ValueError(f"config is missing required integer field {key!r}")
    return value


def require_number(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"config is missing required numeric field {key!r}")
    return float(value)


def check_model_identity(cfg: dict[str, Any]) -> None:
    model_type = str(cfg.get("model_type", "")).lower()
    architectures = [str(x).lower() for x in cfg.get("architectures", [])]
    joined = " ".join([model_type, *architectures])
    if "minicpm" not in joined or "sala" not in joined:
        raise ValueError(
            "config does not look like MiniCPM-SALA; refusing to validate a "
            f"model-specific RoPE fusion on model_type={cfg.get('model_type')!r} "
            f"architectures={cfg.get('architectures')!r}"
        )


def layer_counts(mixer_types: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mixer in mixer_types:
        counts[mixer] = counts.get(mixer, 0) + 1
    return counts


def load_rope_shapes(config_path: Path) -> list[RopeShape]:
    cfg = load_config(config_path)
    check_model_identity(cfg)

    hidden_size = require_int(cfg, "hidden_size")
    num_hidden_layers = require_int(cfg, "num_hidden_layers")
    mixer_types_raw = cfg.get("mixer_types")
    if not isinstance(mixer_types_raw, list) or not mixer_types_raw:
        raise ValueError("MiniCPM-SALA RoPE fusion requires a non-empty mixer_types list")
    mixer_types = [str(x) for x in mixer_types_raw]
    if len(mixer_types) != num_hidden_layers:
        raise ValueError(
            f"mixer_types length {len(mixer_types)} != num_hidden_layers {num_hidden_layers}"
        )

    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rope_scaling = cfg.get("rope_scaling")
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        raise ValueError("rope_scaling must be null or an object")
    max_position_embeddings = int(cfg.get("max_position_embeddings", 8192))
    counts = layer_counts(mixer_types)

    shapes: list[RopeShape] = []
    if counts.get("minicpm4", 0) and bool(cfg.get("attn_use_rope", True)):
        num_heads = require_int(cfg, "num_attention_heads")
        num_kv_heads = require_int(cfg, "num_key_value_heads")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by num_attention_heads {num_heads}"
            )
        head_dim = hidden_size // num_heads
        shapes.append(
            RopeShape(
                label="dense-minicpm4",
                mixer_type="minicpm4",
                layer_count=counts["minicpm4"],
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
            )
        )

    lightning_count = sum(counts.get(name, 0) for name in LIGHTNING_MIXERS)
    if lightning_count and bool(cfg.get("lightning_use_rope", True)):
        shapes.append(
            RopeShape(
                label="lightning",
                mixer_type="lightning",
                layer_count=lightning_count,
                num_heads=require_int(cfg, "lightning_nh"),
                num_kv_heads=require_int(cfg, "lightning_nkv"),
                head_dim=require_int(cfg, "lightning_head_dim"),
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
            )
        )

    if not shapes:
        raise ValueError("config has no RoPE-enabled MiniCPM attention shapes")

    for shape in shapes:
        if shape.head_dim not in FAST_CUDA_ROPE_HEAD_DIMS:
            raise ValueError(
                f"{shape.label} head_dim={shape.head_dim} does not use the fast CUDA "
                "RoPE path; no-upcast equivalence is only valid for "
                f"{sorted(FAST_CUDA_ROPE_HEAD_DIMS)}"
            )
        if shape.num_heads <= 0 or shape.num_kv_heads <= 0:
            raise ValueError(f"{shape.label} has invalid head counts: {shape}")
    return shapes


def make_positions(tokens: int, max_position_embeddings: int, device: str) -> torch.Tensor:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    max_pos = max(1, max_position_embeddings)
    if tokens == 1:
        values = torch.tensor([max_pos - 1], dtype=torch.long)
    else:
        values = torch.linspace(0, max_pos - 1, steps=tokens, dtype=torch.float64).long()
        values[0] = 0
        values[-1] = max_pos - 1
    return values.to(device=device)


@torch.inference_mode()
def run_case(
    *,
    shape: RopeShape,
    tokens: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> ProbeRow:
    from sglang.srt.layers.rotary_embedding import get_rope

    torch.manual_seed(seed + tokens + shape.head_dim + shape.num_heads)
    rope = get_rope(
        shape.head_dim,
        rotary_dim=shape.head_dim,
        max_position=shape.max_position_embeddings,
        base=shape.rope_theta,
        rope_scaling=shape.rope_scaling,
    ).to(device)

    positions = make_positions(tokens, shape.max_position_embeddings, device)
    q = torch.randn(
        tokens,
        shape.num_heads * shape.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        tokens,
        shape.num_kv_heads * shape.head_dim,
        device=device,
        dtype=dtype,
    )
    # Include a stable magnitude spread; this catches cast/rounding differences
    # without creating infinities for fp16.
    q = (q * 3.0).to(dtype)
    k = (k * 3.0).to(dtype)

    q_ref_f = q.float()
    k_ref_f = k.float()
    q_ref_f, k_ref_f = rope(positions, q_ref_f, k_ref_f)
    q_ref = q_ref_f.to(dtype)
    k_ref = k_ref_f.to(dtype)

    q_fused = q.clone()
    k_fused = k.clone()
    q_fused, k_fused = rope(positions, q_fused, k_fused)

    q_diff = (q_ref.float() - q_fused.float()).abs()
    k_diff = (k_ref.float() - k_fused.float()).abs()
    return ProbeRow(
        label=shape.label,
        mixer_type=shape.mixer_type,
        dtype=str(dtype).replace("torch.", ""),
        tokens=tokens,
        num_heads=shape.num_heads,
        num_kv_heads=shape.num_kv_heads,
        head_dim=shape.head_dim,
        q_max_abs=float(q_diff.max().item()),
        k_max_abs=float(k_diff.max().item()),
        q_mean_abs=float(q_diff.mean().item()),
        k_mean_abs=float(k_diff.mean().item()),
        q_bitwise=bool(torch.equal(q_ref, q_fused)),
        k_bitwise=bool(torch.equal(k_ref, k_fused)),
    )


def iter_rows(args: argparse.Namespace, shapes: list[RopeShape]) -> Iterable[ProbeRow]:
    dtype = dtype_from_name(args.dtype)
    for shape in shapes:
        for tokens in parse_ints(args.tokens):
            yield run_case(
                shape=shape,
                tokens=tokens,
                dtype=dtype,
                seed=args.seed,
                device=args.device,
            )


def print_text(rows: list[ProbeRow], shapes: list[RopeShape]) -> None:
    print("MiniCPM-SALA RoPE shapes:")
    for shape in shapes:
        print(
            f"  {shape.label}: layers={shape.layer_count} heads={shape.num_heads} "
            f"kv_heads={shape.num_kv_heads} head_dim={shape.head_dim} "
            f"max_pos={shape.max_position_embeddings}"
        )
    print(
        "label,dtype,tokens,heads,kv_heads,head_dim,"
        "q_max_abs,k_max_abs,q_mean_abs,k_mean_abs,q_bitwise,k_bitwise"
    )
    for row in rows:
        print(
            f"{row.label},{row.dtype},{row.tokens},{row.num_heads},"
            f"{row.num_kv_heads},{row.head_dim},{row.q_max_abs:.8g},"
            f"{row.k_max_abs:.8g},{row.q_mean_abs:.8g},{row.k_mean_abs:.8g},"
            f"{int(row.q_bitwise)},{int(row.k_bitwise)}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16"],
    )
    ap.add_argument("--tokens", default="1,2,7,16,64,257,1024")
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--max-abs", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Exit 0 with a skip record if CUDA is unavailable.",
    )
    args = ap.parse_args()

    try:
        shapes = load_rope_shapes(args.model_config)
    except Exception as exc:
        print(f"[rope-probe] FAIL: model config is not supported: {exc}", file=sys.stderr)
        return 2

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        message = {
            "status": "skipped",
            "reason": "CUDA is unavailable",
            "device": args.device,
            "model_config": str(args.model_config),
            "shapes": [asdict(shape) for shape in shapes],
        }
        print(json.dumps(message, indent=2) if args.json else message["reason"])
        return 0 if args.allow_no_cuda else 2

    rows = list(iter_rows(args, shapes))
    if args.json:
        print(
            json.dumps(
                {
                    "status": "ran",
                    "model_config": str(args.model_config),
                    "threshold": {"bitwise": True, "max_abs": args.max_abs},
                    "shapes": [asdict(shape) for shape in shapes],
                    "rows": [asdict(row) for row in rows],
                },
                indent=2,
            )
        )
    else:
        print_text(rows, shapes)

    failed = [row for row in rows if (not row.bitwise) or row.max_abs > args.max_abs]
    if failed:
        print(
            "[rope-probe] FAIL: no-upcast RoPE path is not bitwise identical "
            "to MiniCPM's current fp32-upcast path for this model config.",
            file=sys.stderr,
        )
        return 1
    print("[rope-probe] PASS: strict MiniCPM-SALA RoPE equivalence satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
