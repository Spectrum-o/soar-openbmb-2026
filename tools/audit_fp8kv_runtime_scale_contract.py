#!/usr/bin/env python3
"""Audit the runtime contract for FP8 KV scale granularity.

This is a static guardrail for the MiniCPM-SALA FP8KV + W4A16 work.  The
prepared SOAR packages can inject scalar `k_scale` / `v_scale` values. A real
per-head or per-block accuracy fix needs the runtime stack to carry non-scalar
scales through loading, KV-cache write, and FlashInfer attention.

The audit answers one concrete question:

    Which parts of the MiniCPM paged-FP8KV path are still scalar-only, and
    which parts have an experimental per-head bridge?

It intentionally treats explicit restrictions as PASS when they document the
current boundary. Today the local SGLang side can preserve per-head checkpoint
scales, broadcast them on cache write, and apply the Q/output bridge around the
MiniCPM grouped attention call. The paged FlashInfer API itself remains scalar.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Check:
    label: str
    passed: bool
    detail: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_contains(label: str, text: str, needles: Iterable[str], detail: str) -> Check:
    missing = [needle for needle in needles if needle not in text]
    return Check(
        label=label,
        passed=not missing,
        detail=detail if not missing else f"missing: {missing!r}",
    )


def audit_sglang_repo(repo_root: Path) -> list[Check]:
    files = {
        "kv_doc": repo_root / "docs" / "advanced_features" / "quantized_kv_cache.md",
        "backend_doc": repo_root / "docs" / "advanced_features" / "attention_backend.md",
        "kv_cache": repo_root / "python" / "sglang" / "srt" / "layers" / "quantization" / "kv_cache.py",
        "weight_utils": repo_root / "python" / "sglang" / "srt" / "model_loader" / "weight_utils.py",
        "radix_attention": repo_root / "python" / "sglang" / "srt" / "layers" / "radix_attention.py",
        "memory_pool": repo_root / "python" / "sglang" / "srt" / "mem_cache" / "memory_pool.py",
        "minicpm_backend": repo_root / "python" / "sglang" / "srt" / "layers" / "attention" / "minicpm_backend.py",
        "minicpm_kernels": repo_root / "python" / "sglang" / "srt" / "layers" / "attention" / "minicpm_attention_kernels.py",
    }

    missing_files = [str(path) for path in files.values() if not path.is_file()]
    if missing_files:
        return [Check("required files exist", False, f"missing files: {missing_files}")]

    texts = {name: read_text(path) for name, path in files.items()}
    checks = [
        check_contains(
            "docs say FP8 KV scale is scalar per-tensor",
            texts["kv_doc"],
            ["Currently, only per-tensor (scalar) scaling factors are supported."],
            "docs/advanced_features/quantized_kv_cache.md",
        ),
        check_contains(
            "docs say MHA FlashInfer supports FP8 KV but not FP4 KV",
            texts["backend_doc"],
            ["**FlashInfer**", "FP8 KV Cache", "FP4 KV Cache"],
            "docs/advanced_features/attention_backend.md support matrix",
        ),
        check_contains(
            "BaseKVCacheMethod keeps scalar fallback parameters",
            texts["kv_cache"],
            [
                "layer.k_scale = torch.nn.Parameter",
                "torch.tensor(-1.0, dtype=torch.float32)",
                "layer.v_scale = torch.nn.Parameter",
            ],
            "scalar fallback remains compatible with existing checkpoints",
        ),
        check_contains(
            "BaseKVCacheMethod preserves optional per-head checkpoint scales",
            texts["kv_cache"],
            [
                "layer.k_scale_per_head = None",
                "layer.v_scale_per_head = None",
                "make_kv_cache_scale_loader",
                "finalize_kv_cache_scales",
            ],
            "non-scalar checkpoint scales now have a tensor side channel",
        ),
        check_contains(
            "JSON quantization schema stores one float per layer",
            texts["weight_utils"],
            ["scaling_factor: Dict[int, Dict[int, float]]"],
            "--quantization-param-path cannot express per-head arrays",
        ),
        check_contains(
            "RadixAttention stores scalar scale attributes",
            texts["radix_attention"],
            ["self.k_scale = None", "self.v_scale = None", "self.k_scale_float = None", "self.v_scale_float = None"],
            "paged FlashInfer still consumes scalar Python-float fallback fields",
        ),
        check_contains(
            "MHA KV pool can broadcast per-head scale before FP8 cast",
            texts["memory_pool"],
            [
                "divide_kv_cache_by_scale_ as _divide_kv_cache_by_scale_",
                "k_scale: Optional[Union[float, torch.Tensor]] = None",
                "num_heads=self.head_num",
                "head_dim=self.head_dim",
            ],
            "cache-write scaling is no longer limited to Python floats",
        ),
        check_contains(
            "FP4 KV path already has block scale buffers",
            texts["memory_pool"],
            ["class MHATokenToKVPoolFP4", "self.k_scale_buffer", "self.v_scale_buffer"],
            "block-scale metadata exists for FP4, not for MiniCPM FP8 MHA",
        ),
        check_contains(
            "MiniCPM backend writes KV with layer scales",
            texts["minicpm_backend"],
            ["_get_kv_cache_write_scales", "token_to_kv_pool.set_kv_buffer"],
            "KV write path can choose per-head scale tensors when present",
        ),
        check_contains(
            "MiniCPM backend wires Q/output per-head bridge",
            texts["minicpm_backend"],
            [
                "_scale_query_for_per_head_fp8_kv",
                "scale_flat_query_for_per_head_k",
                "_finalize_per_head_fp8_kv_output",
                "scale_grouped_output_for_per_head_v",
                "q=q_reshaped",
                "q=q_reshaped_by_head_group",
            ],
            "attention path applies the algebraic bridge around grouped FlashInfer calls",
        ),
        check_contains(
            "MiniCPM fallback dequantizes per-head FP8 KV scales",
            texts["minicpm_backend"],
            [
                "multiply_kv_cache_by_scale_",
                "k_scale_per_head",
                "v_scale_per_head",
                "num_heads=layer.tp_k_head_num",
                "num_heads=layer.tp_v_head_num",
            ],
            "non-FlashInfer fallback no longer treats per-head tensors as scalar max scales",
        ),
        check_contains(
            "MiniCPM FlashInfer wrapper receives float scale side channel",
            texts["minicpm_kernels"],
            ["k_scale_per_head", "scale_kwargs[\"k_scale\"] = 1.0", "layer.k_scale_float"],
            "paged FlashInfer call remains scalar, using 1.0 when the per-head bridge is active",
        ),
    ]
    return checks


def read_flashinfer_file(source: Path, member: str) -> str:
    if source.is_dir():
        return read_text(source / member)
    with zipfile.ZipFile(source) as zf:
        return zf.read(member).decode("utf-8", errors="replace")


def audit_flashinfer(source: Path) -> list[Check]:
    try:
        decode = read_flashinfer_file(source, "flashinfer/decode.py")
        prefill = read_flashinfer_file(source, "flashinfer/prefill.py")
    except Exception as exc:
        return [Check("FlashInfer source readable", False, str(exc))]

    return [
        check_contains(
            "FlashInfer single prefill exposes per-head tensor scales",
            prefill,
            [
                "scale_k : Optional[torch.Tensor]",
                "per-head quantization",
                "scale_v : Optional[torch.Tensor]",
            ],
            "this is not the paged wrapper used by MiniCPM serving",
        ),
        check_contains(
            "FlashInfer paged prefill wrapper exposes scalar k/v scales",
            prefill,
            ["k_scale: Optional[float]", "v_scale: Optional[float]", "None,  # scale_k", "None,  # scale_v"],
            "BatchPrefillWithPagedKVCacheWrapper does not forward tensor scale_k/scale_v",
        ),
        check_contains(
            "FlashInfer paged decode wrapper exposes scalar k/v scales",
            decode,
            ["class BatchDecodeWithPagedKVCacheWrapper", "k_scale: Optional[float]", "v_scale: Optional[float]", "None,  # scale_k", "None,  # scale_v"],
            "BatchDecodeWithPagedKVCacheWrapper does not forward tensor scale_k/scale_v",
        ),
    ]


def print_checks(title: str, checks: list[Check]) -> bool:
    print(f"== {title} ==")
    ok = True
    for check in checks:
        ok = ok and check.passed
        status = "PASS" if check.passed else "WARN"
        print(f"{status} {check.label}: {check.detail}")
    print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="SGLang repo root to audit.",
    )
    ap.add_argument(
        "--flashinfer-source",
        type=Path,
        help="Optional FlashInfer source directory or wheel path to audit.",
    )
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = ap.parse_args()

    repo_checks = audit_sglang_repo(args.repo_root)
    flashinfer_checks = audit_flashinfer(args.flashinfer_source) if args.flashinfer_source else []

    if args.json:
        print(
            json.dumps(
                {
                    "repo_root": str(args.repo_root),
                    "flashinfer_source": str(args.flashinfer_source) if args.flashinfer_source else None,
                    "sglang": [asdict(check) for check in repo_checks],
                    "flashinfer": [asdict(check) for check in flashinfer_checks],
                },
                indent=2,
            )
        )
    else:
        repo_ok = print_checks("SGLang FP8KV scale contract", repo_checks)
        flashinfer_ok = True
        if flashinfer_checks:
            flashinfer_ok = print_checks("FlashInfer paged-wrapper scale contract", flashinfer_checks)
        else:
            print("== FlashInfer paged-wrapper scale contract ==")
            print("SKIP provide --flashinfer-source <source-dir-or-wheel> for upstream wrapper audit")
            print()

        if repo_ok and flashinfer_ok:
            print("conclusion: local MiniCPM has an experimental per-head FP8KV bridge, but paged FlashInfer remains scalar-scale.")
            print("next runtime work: GPU smoke-test the bridge and build a package that emits per-head scale tensors.")
        else:
            print("conclusion: contract drift detected; inspect WARN lines before changing FP8KV scale granularity.")

    return 0 if all(c.passed for c in repo_checks + flashinfer_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
