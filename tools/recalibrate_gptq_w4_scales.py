#!/usr/bin/env python3
"""Recalibrate GPTQ W4A16 scale tensors while keeping qweight/qzeros fixed.

This is a conservative post-process for GPTQModel artifacts:

    weight ~= (q_unsigned - qzero) * scale

For each quantized Linear module, each input group, and each output column, we
solve the one-dimensional least-squares scale that best reconstructs the
original BF16 weight from the already-packed int4 values. Only `.scales`
tensors are rewritten; `.qweight`, `.qzeros`, `.g_idx`, configs, and module
selection stay unchanged.

It is intentionally weight-only. It does not need calibration prompts or a GPU.
An activation-aware variant would need hidden-state calibration and should be
treated as a separate, riskier experiment.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


QWEIGHT_SUFFIX = ".qweight"
QZEROS_SUFFIX = ".qzeros"
SCALES_SUFFIX = ".scales"
GIDX_SUFFIX = ".g_idx"


@dataclass
class ModuleReport:
    module: str
    changed: bool
    old_mse: float
    new_mse: float
    old_mean_scale: float
    new_mean_scale: float
    max_rel_scale_change: float
    note: str = ""


def find_shard(model_dir: Path, tensor_name: str) -> Path:
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        shard = idx.get("weight_map", {}).get(tensor_name)
        if shard:
            return model_dir / shard

    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            if tensor_name in f.keys():
                return single

    for shard_path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt") as f:
            if tensor_name in f.keys():
                return shard_path
    raise FileNotFoundError(f"{tensor_name} not found under {model_dir}")


def load_tensor(model_dir: Path, tensor_name: str) -> torch.Tensor:
    shard = find_shard(model_dir, tensor_name)
    with safe_open(str(shard), framework="pt") as f:
        return f.get_tensor(tensor_name)


def discover_quantized_modules(artifact: Path) -> list[str]:
    idx_path = artifact / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        keys = idx.get("weight_map", {}).keys()
    else:
        keys_set: set[str] = set()
        for shard in sorted(artifact.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as f:
                keys_set.update(f.keys())
        keys = keys_set

    modules = []
    for key in keys:
        if key.endswith(QWEIGHT_SUFFIX):
            module = key[: -len(QWEIGHT_SUFFIX)]
            required = (f"{module}{QZEROS_SUFFIX}", f"{module}{SCALES_SUFFIX}", f"{module}{GIDX_SUFFIX}")
            if all(any(k == req for k in keys) for req in required):
                modules.append(module)
    return sorted(set(modules))


def unpack_qweight_rows(qweight: torch.Tensor, in_indices: torch.Tensor) -> torch.Tensor:
    rows = in_indices // 8
    shifts = ((in_indices % 8) * 4).to(torch.int64)
    packed = qweight[rows].to(torch.int64)
    return ((packed >> shifts.unsqueeze(1)) & 0xF).to(torch.float32)


def unpack_qzeros_row(qzeros: torch.Tensor, group_idx: int, out_features: int) -> torch.Tensor:
    out_idx = torch.arange(out_features, dtype=torch.int64)
    cols = out_idx // 8
    shifts = (out_idx % 8) * 4
    packed = qzeros[group_idx, cols].to(torch.int64)
    return ((packed >> shifts) & 0xF).to(torch.float32)


def recalibrate_module_scales(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bf16_weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, float, float]:
    """Return (new_scales, old_mse, new_mse).

    qweight/scales use GPTQ layout [in, out] after unpacking. Hugging Face
    Linear weights are [out, in], so bf16_weight is transposed for comparison.
    """
    in_features = qweight.shape[0] * 8
    out_features = qweight.shape[1]
    if bf16_weight.shape != (out_features, in_features):
        raise ValueError(
            f"BF16 weight shape {tuple(bf16_weight.shape)} does not match "
            f"GPTQ qweight-derived shape {(out_features, in_features)}"
        )
    if scales.shape[1] != out_features:
        raise ValueError(
            f"scales shape {tuple(scales.shape)} incompatible with out_features={out_features}"
        )
    if g_idx.numel() != in_features:
        raise ValueError(
            f"g_idx length {g_idx.numel()} incompatible with in_features={in_features}"
        )

    g_idx_i64 = g_idx.to(torch.int64).cpu()
    qzeros_i32 = qzeros.to(torch.int32).cpu()
    qweight_i32 = qweight.to(torch.int32).cpu()
    ref = bf16_weight.T.contiguous().to(torch.float32).cpu()
    old_scales_f32 = scales.to(torch.float32).cpu()
    new_scales = old_scales_f32.clone()

    old_sse = 0.0
    new_sse = 0.0
    total = 0

    for group_idx in range(scales.shape[0]):
        in_indices = torch.nonzero(g_idx_i64 == group_idx, as_tuple=False).flatten()
        if in_indices.numel() == 0:
            continue

        q_unsigned = unpack_qweight_rows(qweight_i32, in_indices)
        qzero = unpack_qzeros_row(qzeros_i32, group_idx, out_features)
        q_centered = q_unsigned - qzero.unsqueeze(0)
        ref_group = ref[in_indices]

        denom = (q_centered * q_centered).sum(dim=0)
        numer = (q_centered * ref_group).sum(dim=0)
        solved = torch.where(
            denom > eps,
            numer / denom.clamp_min(eps),
            old_scales_f32[group_idx],
        )
        # Symmetric int4 scales are expected to be non-negative. Preserve the
        # old scale if LS degenerates or flips sign due to all-zero q values.
        solved = torch.where(
            torch.isfinite(solved) & (solved > eps),
            solved,
            old_scales_f32[group_idx],
        )
        new_scales[group_idx] = solved

        old_recon = q_centered * old_scales_f32[group_idx].unsqueeze(0)
        new_recon = q_centered * solved.unsqueeze(0)
        old_sse += float(((old_recon - ref_group) ** 2).sum().item())
        new_sse += float(((new_recon - ref_group) ** 2).sum().item())
        total += ref_group.numel()

    if total == 0:
        raise ValueError("no GPTQ groups were visited")

    return new_scales.to(dtype=scales.dtype), old_sse / total, new_sse / total


def rewrite_scale_shards(artifact: Path, updated: dict[str, torch.Tensor]) -> int:
    rewritten = 0
    for shard_path in sorted(artifact.glob("*.safetensors")):
        state = load_file(str(shard_path))
        touched = False
        for key, value in updated.items():
            if key in state:
                state[key] = value.contiguous()
                touched = True
        if touched:
            save_file(state, str(shard_path), metadata={"format": "pt"})
            rewritten += 1
    return rewritten


def run(args: argparse.Namespace) -> int:
    artifact = Path(args.artifact)
    base = Path(args.base)
    module_re = re.compile(args.module_regex) if args.module_regex else None

    modules = discover_quantized_modules(artifact)
    if module_re is not None:
        modules = [m for m in modules if module_re.search(m)]
    if args.limit is not None:
        modules = modules[: args.limit]
    if not modules:
        print("[w4-scale] no matching quantized modules found")
        return 2

    reports: list[ModuleReport] = []
    updated_scales: dict[str, torch.Tensor] = {}

    print(f"[w4-scale] artifact={artifact}")
    print(f"[w4-scale] base={base}")
    print(f"[w4-scale] modules={len(modules)} dry_run={args.dry_run}")

    for module in modules:
        scale_name = f"{module}{SCALES_SUFFIX}"
        try:
            qweight = load_tensor(artifact, f"{module}{QWEIGHT_SUFFIX}")
            qzeros = load_tensor(artifact, f"{module}{QZEROS_SUFFIX}")
            scales = load_tensor(artifact, scale_name)
            g_idx = load_tensor(artifact, f"{module}{GIDX_SUFFIX}")
            bf16_weight = load_tensor(base, f"{module}.weight")
            new_scales, old_mse, new_mse = recalibrate_module_scales(
                qweight, qzeros, scales, g_idx, bf16_weight, eps=args.eps
            )
            rel = ((new_scales.to(torch.float32) - scales.to(torch.float32)).abs() / scales.to(torch.float32).abs().clamp_min(args.eps)).max().item()
            changed = bool((new_scales != scales).any().item())
            if changed:
                updated_scales[scale_name] = new_scales
            reports.append(
                ModuleReport(
                    module=module,
                    changed=changed,
                    old_mse=old_mse,
                    new_mse=new_mse,
                    old_mean_scale=float(scales.to(torch.float32).abs().mean().item()),
                    new_mean_scale=float(new_scales.to(torch.float32).abs().mean().item()),
                    max_rel_scale_change=float(rel),
                )
            )
            gain = 0.0 if old_mse == 0 else (old_mse - new_mse) / old_mse * 100.0
            print(
                f"[w4-scale] {module}: mse {old_mse:.6g}->{new_mse:.6g} "
                f"({gain:+.2f}%) mean_scale {reports[-1].old_mean_scale:.6g}->{reports[-1].new_mean_scale:.6g} "
                f"max_rel_change={rel:.4f}"
            )
        except Exception as exc:
            reports.append(
                ModuleReport(
                    module=module,
                    changed=False,
                    old_mse=float("nan"),
                    new_mse=float("nan"),
                    old_mean_scale=float("nan"),
                    new_mean_scale=float("nan"),
                    max_rel_scale_change=float("nan"),
                    note=str(exc),
                )
            )
            print(f"[w4-scale] SKIP {module}: {exc}")

    ok_reports = [r for r in reports if not r.note]
    if not ok_reports:
        print("[w4-scale] no module could be recalibrated")
        return 2

    old_total = sum(r.old_mse for r in ok_reports)
    new_total = sum(r.new_mse for r in ok_reports)
    avg_gain = 0.0 if old_total == 0 else (old_total - new_total) / old_total * 100.0
    print(
        f"[w4-scale] SUMMARY modules_ok={len(ok_reports)} skipped={len(reports)-len(ok_reports)} "
        f"changed={sum(1 for r in ok_reports if r.changed)} avg_mse_gain={avg_gain:+.2f}%"
    )

    if args.report:
        report_path = Path(args.report)
        payload = {
            "artifact": str(artifact),
            "base": str(base),
            "dry_run": args.dry_run,
            "summary": {
                "modules_ok": len(ok_reports),
                "modules_skipped": len(reports) - len(ok_reports),
                "modules_changed": sum(1 for r in ok_reports if r.changed),
                "avg_mse_gain_pct": avg_gain,
            },
            "modules": [r.__dict__ for r in reports],
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[w4-scale] report={report_path}")

    if args.dry_run:
        print("[w4-scale] dry-run: not rewriting safetensors")
        return 0

    rewritten = rewrite_scale_shards(artifact, updated_scales)
    print(f"[w4-scale] rewrote {len(updated_scales)} .scales tensors across {rewritten} shard(s)")
    if updated_scales and rewritten == 0:
        print("[w4-scale] FATAL: computed scale updates but no shard was rewritten")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="Quantized GPTQ artifact directory")
    parser.add_argument("--base", required=True, help="Original BF16 model directory")
    parser.add_argument("--module-regex", default="", help="Optional regex over module names")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N matching modules")
    parser.add_argument("--report", default="", help="Optional JSON report path")
    parser.add_argument("--dry-run", action="store_true", help="Compute and report, but do not rewrite .scales")
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
