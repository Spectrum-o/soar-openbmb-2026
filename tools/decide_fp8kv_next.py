#!/usr/bin/env python3
"""Decide the next FP8KV package after a SOAR platform result.

This helper is intentionally small and evidence-oriented. It does not call the
platform API. Paste a Score JSON or pass a log file; the script extracts
`acc_ori`, `final_score`, and benchmark durations when present, then prints the
next package in the real-calib FP8KV queue.  The submitted 225432 E5M2 package
is treated as a legacy in-flight signal package, not the final E5M2_OFFLOAD
backup in the current queue.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ATTNSCALE_ACC_ORI = 77.24
PASS_ACC_ORI = 80.0
CURRENT_PACKAGE_KEY = ""
CURRENT_ACTION = (
    "No prepared FP8KV tarball is the clean current next submit. "
    "The 101730 DENSEQKV_MULTI8K result completed below the non-FP8 W4A16 record; "
    "if continuing the FP8KV precision queue, build a new POSTQ_MULTI8K HP224 package. "
    "Do not submit the old 4K POSTQ/PERHEAD tarballs as the next comparison."
)


@dataclass(frozen=True)
class Package:
    key: str
    tarball: str
    md5: str
    role: str


@dataclass(frozen=True)
class GpuDiag:
    tag: str
    total_mib: int | None = None
    used_mib: int | None = None
    free_mib: int | None = None
    torch_free_gib: float | None = None
    torch_total_gib: float | None = None
    has_processes: bool = False


PACKAGES = {
    "hp224": Package(
        key="hp224",
        tarball="soar_fp8kv_REALCALIB_TAIL_HP224_OFFLOAD_FIXED_20260531_121439.tar.gz",
        md5="a1ab2a273390f87a1a5c8c041a252c62",
        role="heavy fixed HP224 package, now superseded for platform submission after 121439 OOM: e4m3, safe-max 224, real question-field + tail-window KV calibration, GPTQ disk offload enabled through QuantizeConfig only",
    ),
    "denseqkv": Package(
        key="denseqkv",
        tarball="soar_fp8kv_DENSEQKV_HP224_OFFLOAD_FIXED_20260531_123154.tar.gz",
        md5="f5da669be848b913bc0181ed6180ccf3",
        role="heavy diagnostic precision package, now superseded for platform submission after 121439 OOM: e4m3 HP224 with dense minicpm4 q/k/v restored to BF16 so BF16 KV-scale calibration matches the runtime KV-write source; fixed to avoid offload_to_disk load kwarg",
    ),
    "hp224_lowmem": Package(
        key="hp224_lowmem",
        tarball="soar_fp8kv_REALCALIB_TAIL4K_HP224_OFFLOAD_FIXED_20260531_141006.tar.gz",
        md5="6626675a061960393b1cf70943b93372",
        role="prepare-OOM recovery package after 121439: e4m3 HP224 with real question-field calibration kept, but GPTQ replay reduced to 150 public rows, 4K tail windows, and no multi-adaptive expansion",
    ),
    "hp224_midcalib": Package(
        key="hp224_midcalib",
        tarball="soar_fp8kv_REALCALIB_TAIL4K300_HP224_DIAG_OFFLOAD_FIXED_20260531_174343.tar.gz",
        md5="b1bcfa37d9764490743065cc3af60306",
        role="mid-calibration HP224 package after 141006 proved 150 x 4K is too light: e4m3 HP224 with real question-field calibration, GPTQ replay restored to 300 public rows, 4K tail windows, no multi-adaptive expansion, and GPU memory diagnostics before/after quantize",
    ),
    "hp224_multi8k_diag": Package(
        key="hp224_multi8k_diag",
        tarball="soar_fp8kv_REALCALIB_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260531_225344.tar.gz",
        md5="40b692635b4743bc64d5d57db8ce2fde",
        role="diagnostic HP224 package after 174343 prepared too quickly and stayed at ATTNSCALE quality: e4m3 HP224 with real question-field calibration, GPTQ replay restored to 300 public prompts x 8K multi-adaptive windows, offload fix, and GPU diagnostics before/after quantize",
    ),
    "denseqkv_midcalib": Package(
        key="denseqkv_midcalib",
        tarball="soar_fp8kv_DENSEQKV_TAIL4K300_HP224_DIAG_OFFLOAD_FIXED_20260531_214103.tar.gz",
        md5="afe85f92dd3f104954c6368cbac82593",
        role="comparable DENSEQKV follow-up after hp224_midcalib: e4m3 HP224 with dense minicpm4 q/k/v restored to BF16, real question-field calibration, GPTQ replay matched to 300 public rows x 4K tail windows, and GPU memory diagnostics",
    ),
    "denseqkv_multi8k_diag": Package(
        key="denseqkv_multi8k_diag",
        tarball="soar_fp8kv_DENSEQKV_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260601_101730.tar.gz",
        md5="3ad3f92828d5595768d6977a4e839dab",
        role="clean DENSEQKV follow-up after 225344: same e4m3 HP224, real question-field calibration, GPTQ replay preserved at 300 public prompts x 8K multi-adaptive windows, and dense minicpm4 q/k/v restored to BF16 so BF16 KV-scale calibration matches the runtime KV-write source",
    ),
    "postq": Package(
        key="postq",
        tarball="soar_fp8kv_POSTQ_HP224_OFFLOAD_FIXED_20260531_124940.tar.gz",
        md5="2c6d679098c89689b1c0940058cc09ef",
        role="heavy post-GPTQ precision package, now superseded for platform submission after 121439 OOM: e4m3 HP224 with KV scales measured from the final qzeros-fixed W4A16 artifact after selective BF16 overlay, then injected as final mutation; fixed to avoid offload_to_disk load kwarg",
    ),
    "postq_midcalib": Package(
        key="postq_midcalib",
        tarball="soar_fp8kv_POSTQ_TAIL4K300_HP224_DIAG_OFFLOAD_FIXED_20260531_215412.tar.gz",
        md5="d048431fcb4ff31f8eb756938ba6da0e",
        role="comparable POSTQ follow-up after denseqkv_midcalib: e4m3 HP224 with KV scales measured from the final qzeros-fixed W4A16 artifact after selective BF16 overlay, GPTQ replay matched to 300 public rows x 4K tail windows, and GPU memory diagnostics",
    ),
    "perhead_midcalib": Package(
        key="perhead_midcalib",
        tarball="soar_fp8kv_POSTQ_PERHEAD_TAIL4K300_HP224_BRIDGE_DIAG_OFFLOAD_FIXED_20260531_221735.tar.gz",
        md5="5f18e1aae6d7dda4a365ff8dc49a7080",
        role="comparable PERHEAD follow-up after postq_midcalib: e4m3 HP224 with post-GPTQ rank-1 per-head KV scale tensors, MiniCPM Q/output bridge, GPTQ replay matched to 300 public rows x 4K tail windows, and GPU memory diagnostics",
    ),
    "perhead": Package(
        key="perhead",
        tarball="soar_fp8kv_POSTQ_PERHEAD_HP224_BRIDGE_FIXED_20260531_124940.tar.gz",
        md5="e746cf4d450f6f418bf8699bcc8f3659",
        role="heavy runtime precision package, now superseded for platform submission after 121439 OOM: e4m3 HP224 with post-GPTQ rank-1 per-head KV scale tensors measured from the final W4A16 artifact, KV-cache write broadcasting, MiniCPM FlashInfer Q/output bridge, per-head fallback dequant, and offload kwarg fix",
    ),
    "denseqkv_lowmem": Package(
        key="denseqkv_lowmem",
        tarball="soar_fp8kv_DENSEQKV_TAIL4K_HP224_OFFLOAD_FIXED_20260531_142521.tar.gz",
        md5="85068b1fa69c22d9996c34b30f34a328",
        role="low-memory diagnostic precision package after 121439 OOM: e4m3 HP224, DENSEQKV BF16-restored q/k/v, real question-field calibration kept, GPTQ replay reduced to 150 public rows and 4K tail windows",
    ),
    "postq_lowmem": Package(
        key="postq_lowmem",
        tarball="soar_fp8kv_POSTQ_TAIL4K_HP224_OFFLOAD_FIXED_20260531_142521.tar.gz",
        md5="04e6713bdcdb8c61f8708df07b41dc4a",
        role="low-memory post-GPTQ precision package after 121439 OOM: e4m3 HP224 with KV scales measured from the final qzeros-fixed W4A16 artifact, GPTQ replay reduced to 150 public rows and 4K tail windows",
    ),
    "perhead_lowmem": Package(
        key="perhead_lowmem",
        tarball="soar_fp8kv_POSTQ_PERHEAD_TAIL4K_HP224_BRIDGE_FIXED_20260531_142521.tar.gz",
        md5="3de0fadb7254586b93c4832b64309e82",
        role="low-memory runtime precision package after 121439 OOM: e4m3 HP224 with post-GPTQ rank-1 per-head KV scale tensors and MiniCPM Q/output bridge, GPTQ replay reduced to 150 public rows and 4K tail windows",
    ),
    "max448": Package(
        key="max448",
        tarball="soar_fp8kv_REALCALIB_TAIL_MAX448_20260530_234219.tar.gz",
        md5="72386164882f14dd7bd4549e6636ecab",
        role="legacy precision backup: e4m3, safe-max 448, tail-window calibration, no hidden-outlier headroom; lacks GPTQ disk offload, so prefer denseqkv first after HP224_OFFLOAD",
    ),
    "e5m2": Package(
        key="e5m2",
        tarball="soar_fp8kv_REALCALIB_TAIL_E5M2_OFFLOAD_FIXED_20260531_130420.tar.gz",
        md5="ad603f6183956f73bd0a05f029c9ca3f",
        role="heavy range backup, now superseded for platform submission after 121439 OOM: e5m2, safe-max 224, tail-window calibration, GPTQ disk offload enabled through QuantizeConfig only",
    ),
    "e5m2_lowmem": Package(
        key="e5m2_lowmem",
        tarball="soar_fp8kv_REALCALIB_TAIL4K_E5M2_OFFLOAD_FIXED_20260531_143106.tar.gz",
        md5="3d653a47dbe68e2066e4dc2d47523a92",
        role="low-memory range backup after 121439 OOM: e5m2, safe-max 224, real question-field calibration kept, GPTQ replay reduced to 150 public rows and 4K tail windows",
    ),
}


PACKAGE_ORDER = [
    "hp224",
    "hp224_lowmem",
    "hp224_midcalib",
    "hp224_multi8k_diag",
    "denseqkv_multi8k_diag",
    "denseqkv_midcalib",
    "denseqkv",
    "denseqkv_lowmem",
    "postq_midcalib",
    "perhead_midcalib",
    "postq",
    "postq_lowmem",
    "perhead",
    "perhead_lowmem",
    "e5m2",
    "e5m2_lowmem",
    "max448",
]


def read_text(args: argparse.Namespace) -> str:
    if args.score_json:
        return args.score_json
    if args.score_file:
        return Path(args.score_file).read_text(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def infer_last(args: argparse.Namespace, text: str) -> str:
    if args.last:
        return args.last
    lower = text.lower()
    if "095129" in lower or "postq_perhead_hp224_bridge_20260531_095129" in lower:
        return "legacy_perhead_095129"
    if "225432" in lower or "realcalib_e5m2_20260530" in lower:
        return "legacy_e5m2"
    if "denseqkv" in lower:
        if "multi8k300" in lower or "101730" in lower:
            return "denseqkv_multi8k_diag"
        if "tail4k300" in lower or "214103" in lower or "midcalib" in lower:
            return "denseqkv_midcalib"
        return "denseqkv_lowmem" if "tail4k" in lower or "142521" in lower else "denseqkv"
    if "postq_perhead" in lower or "perhead" in lower:
        if "tail4k300" in lower or "221735" in lower or "midcalib" in lower:
            return "perhead_midcalib"
        return "perhead_lowmem" if "tail4k" in lower or "142521" in lower else "perhead"
    if "postq" in lower:
        if "tail4k300" in lower or "215412" in lower or "midcalib" in lower:
            return "postq_midcalib"
        return "postq_lowmem" if "tail4k" in lower or "142521" in lower else "postq"
    if "max448" in lower:
        return "max448"
    if "multi8k300" in lower or "225344" in lower:
        return "hp224_multi8k_diag"
    if "tail4k300" in lower or "midcalib" in lower or "172606" in lower or "174343" in lower:
        return "hp224_midcalib"
    if "tail4k" in lower or "141006" in lower:
        if "e5m2" in lower or "143106" in lower:
            return "e5m2_lowmem"
        return "hp224_lowmem"
    if "e5m2" in lower:
        return "e5m2"
    if "hp224" in lower or "realcalib" in lower:
        return "hp224"
    return "hp224"


def get_float(obj: dict[str, Any] | None, key: str) -> float | None:
    if not obj:
        return None
    val = obj.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def has_startup_failure(text: str) -> bool:
    lower = text.lower()
    markers = (
        "sglang 服务启动失败",
        "server start failed",
        "keyerror",
        "scheduler hit an exception",
        "received sigquit",
        "failed before inference",
    )
    return any(marker in lower for marker in markers)


def has_prepare_oom(text: str) -> bool:
    lower = text.lower()
    markers = (
        "prepare_model.sh 执行失败",
        "torch.outofmemoryerror",
        "cuda out of memory",
        "tried to allocate",
        "quantize exited 1",
    )
    return any(marker in lower for marker in markers)


def has_offload_kwarg_failure(text: str) -> bool:
    lower = text.lower()
    return "offload_to_disk" in lower and (
        "unexpected keyword argument" in lower
        or "got an unexpected keyword" in lower
        or "minicpmsalaforcausallm.__init__" in lower
    )


def has_range_symptom(text: str) -> bool:
    lower = text.lower()
    substring_markers = (
        "saturat",
        "overflow",
        "outlier",
        "e4m3 range",
        "range-limited",
    )
    if any(marker in lower for marker in substring_markers):
        return True
    return bool(re.search(r"(?<![a-z0-9_])(inf|nan)(?![a-z0-9_])", lower))


def format_pkg(pkg: Package) -> str:
    return f"{pkg.tarball}\nmd5 {pkg.md5}\nrole: {pkg.role}"


def parse_gpu_diags(text: str) -> dict[str, GpuDiag]:
    """Parse prepare_model.sh GPU DIAG blocks from the diagnostic packages."""

    lines = text.splitlines()
    by_tag: dict[str, dict[str, Any]] = {}
    active_process_tag: str | None = None
    for idx, line in enumerate(lines):
        marker = re.search(r"GPU DIAG \(([^)]+)\): ([^\n]+)", line)
        if marker:
            tag, section = marker.group(1), marker.group(2).strip().lower()
            by_tag.setdefault(tag, {"tag": tag})
            active_process_tag = tag if section == "processes" else None
            continue

        if active_process_tag:
            stripped = line.strip()
            next_is_diag = "GPU DIAG (" in line
            if next_is_diag:
                active_process_tag = None
            elif stripped and "no running processes found" not in stripped.lower():
                # Ignore table borders/header rows but keep real process rows.
                if re.search(r"\b(C|G|C\+G)\b", stripped) and re.search(
                    r"\d+MiB", stripped
                ):
                    by_tag.setdefault(active_process_tag, {"tag": active_process_tag})[
                        "has_processes"
                    ] = True

        if idx > 0:
            prev = lines[idx - 1]
            summary = re.search(r"GPU DIAG \(([^)]+)\): nvidia-smi summary", prev)
            if summary:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    try:
                        tag = summary.group(1)
                        by_tag.setdefault(tag, {"tag": tag}).update(
                            {
                                "total_mib": int(parts[2]),
                                "used_mib": int(parts[3]),
                                "free_mib": int(parts[4]),
                            }
                        )
                    except ValueError:
                        pass

        torch_mem = re.search(
            r"torch\.cuda\.mem_get_info free_gib=([0-9.]+) total_gib=([0-9.]+)",
            line,
        )
        if torch_mem:
            tag = None
            for prev in reversed(lines[max(0, idx - 8) : idx]):
                section = re.search(
                    r"GPU DIAG \(([^)]+)\): torch mem_get_info", prev
                )
                if section:
                    tag = section.group(1)
                    break
            if tag:
                by_tag.setdefault(tag, {"tag": tag}).update(
                    {
                        "torch_free_gib": float(torch_mem.group(1)),
                        "torch_total_gib": float(torch_mem.group(2)),
                    }
                )

    return {tag: GpuDiag(**values) for tag, values in by_tag.items()}


def classify_prepare_oom(text: str) -> str | None:
    """Return the dominant OOM cause when the log has enough evidence."""

    if not has_prepare_oom(text):
        return None

    diags = parse_gpu_diags(text)
    before = diags.get("before-gptq-quantize")
    failure = diags.get("after-gptq-quantize-failure")
    if before is None:
        return "mixed_no_gpu_diag"

    before_free_gib = (
        before.free_mib / 1024 if before.free_mib is not None else before.torch_free_gib
    )
    before_total_gib = (
        before.total_mib / 1024
        if before.total_mib is not None
        else before.torch_total_gib
    )
    failure_free_gib = None
    if failure is not None:
        failure_free_gib = (
            failure.free_mib / 1024
            if failure.free_mib is not None
            else failure.torch_free_gib
        )

    if before_free_gib is not None and before_total_gib is not None:
        free_ratio = before_free_gib / before_total_gib
        if before.has_processes or before_free_gib < 60 or free_ratio < 0.72:
            return "platform_dirty_gpu"
        if before_free_gib >= 70 and (
            failure_free_gib is None or failure_free_gib < 2
        ):
            return "package_peak_memory"

    return "mixed_gpu_diag"


def format_oom_classification(kind: str | None) -> str | None:
    if kind is None:
        return None
    if kind == "platform_dirty_gpu":
        return (
            "oom_classification: platform_dirty_gpu\n"
            "evidence: before-gptq-quantize GPU diagnostics already show low free memory or external GPU processes, so the platform state dominates this failure."
        )
    if kind == "package_peak_memory":
        return (
            "oom_classification: package_peak_memory\n"
            "evidence: GPU was clean before quantize, then the run reached card-limit memory during GPTQ replay."
        )
    if kind == "mixed_no_gpu_diag":
        return (
            "oom_classification: mixed_no_gpu_diag\n"
            "evidence: prepare OOM log has no GPU DIAG block, so platform dirty-GPU vs package peak cannot be separated from this run alone."
        )
    return (
        "oom_classification: mixed_gpu_diag\n"
        "evidence: GPU DIAG exists, but it does not cleanly prove either dirty platform state or package-only peak memory."
    )


def decide(last: str, text: str, score: dict[str, Any] | None) -> str:
    acc_ori = get_float(score, "acc_ori")
    final_score = get_float(score, "final_score")
    duration = score.get("benchmark_duration") if isinstance(score, dict) else None

    lines: list[str] = []
    lines.append(f"last={last}")
    if acc_ori is not None:
        lines.append(f"acc_ori={acc_ori:.2f}")
    if final_score is not None:
        lines.append(f"final_score={final_score:.2f}")
    if isinstance(duration, dict):
        bench = " ".join(f"{k}={v}" for k, v in duration.items())
        lines.append(f"benchmark_duration: {bench}")

    if has_startup_failure(text):
        lines.append("decision: STOP FP8KV queue and fix startup/load path first.")
        lines.append("reason: result looks like service startup or scheduler failure, so dtype/safe-max A/B would be confounded.")
        return "\n".join(lines)

    if has_offload_kwarg_failure(text):
        pkg = PACKAGES["hp224_midcalib"]
        lines.append("decision: ignore this old offload package result and submit MIDCALIB fixed HP224 next.")
        lines.append(
            "reason: the run failed before inference because GPTQModel forwarded "
            "offload_to_disk into MiniCPMSALAForCausalLM.__init__; this is a "
            "packaging/load-kwarg bug, not a FP8KV precision signal. "
            "After 121439 proved the 300 x 8K multi-adaptive package can OOM and "
            "141006 proved 150 x 4K prepares too quickly to be a strong precision signal, "
            "the next fixed package should be the 300 x 4K tail-window HP224 build."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if has_prepare_oom(text):
        oom_kind = classify_prepare_oom(text)
        oom_note = format_oom_classification(oom_kind)
        if oom_note:
            lines.append(oom_note)
        if oom_kind == "platform_dirty_gpu":
            pkg = PACKAGES[last] if last in PACKAGES else PACKAGES[CURRENT_PACKAGE_KEY]
            lines.append("decision: resubmit the same diagnostic package once before shrinking calibration.")
            lines.append(
                "reason: the package did not get a clean GPU at GPTQ start, so a smaller package would mix two variables: calibration strength and platform state."
            )
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        if last == "hp224_midcalib":
            lines.append("decision: STOP and build an even smaller calibrated package; do not advance to precision variants yet.")
            lines.append(
                "reason: the 300 x 4K tail-window package still failed during GPTQ prepare replay, "
                "so DENSEQKV/POSTQ/PERHEAD variants with comparable calibration would share the same memory risk. "
                "Next memory step would be 300 x 2K or 150 x 4K, but 141006 already showed the latter is a weak precision signal."
            )
            return "\n".join(lines)
        if last == "hp224_multi8k_diag":
            lines.append("decision: classify this as the 8K multi-adaptive GPTQ peak-memory result before changing FP8KV precision variants.")
            lines.append(
                "reason: this package restores the record-style 300 x 8K multi-adaptive GPTQ replay. "
                "If OOM occurs before the fp8 KV scale calibration step, the failure is GPTQ/platform memory, not KV-scale calibration."
            )
            return "\n".join(lines)
        if last == "hp224_lowmem":
            lines.append("decision: STOP precision queue and build a smaller LOWMEM HP224 package before submitting variants.")
            lines.append(
                "reason: even the 150 x 4K tail-window package failed during GPTQ prepare replay, "
                "so DENSEQKV/POSTQ/PERHEAD variants share the same memory risk. "
                "Reduce GPTQ replay first, e.g. NUM_CALIB=75 or MAX_CALIB_LEN=2048."
            )
            return "\n".join(lines)
        pkg = PACKAGES["hp224_midcalib"]
        lines.append("decision: submit MIDCALIB TAIL4K300 HP224 next.")
        lines.append("reason: result failed before inference during GPTQ prepare_model replay with CUDA OOM; the 300 x 8K multi-adaptive real-question calibration path is too large for the current 83GB platform run, but the 150 x 4K low-memory run prepared in ~12m and is too weak as a precision signal. Try 300 x 4K tail before testing FP8KV precision variants.")
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if acc_ori is None:
        lines.append("decision: WAIT / provide Score JSON.")
        lines.append("reason: no acc_ori found, so accuracy branch cannot be chosen.")
        return "\n".join(lines)

    if acc_ori >= PASS_ACC_ORI and (final_score is None or final_score > 0):
        lines.append("decision: KEEP this FP8KV result; do not submit another FP8KV backup yet.")
        lines.append("reason: correctness gate cleared, so next action should be based on final_score/speed comparison, not another precision guess.")
        return "\n".join(lines)

    if last == "legacy_perhead_095129":
        pkg = PACKAGES["hp224_midcalib"]
        lines.append("decision: submit MIDCALIB fixed HP224 next.")
        lines.append(
            "reason: the 095129 tarball is a superseded pre-fix PERHEAD package; "
            "it still contains the known offload_to_disk load-kwarg bug and an "
            "older fallback path, so its result is not a PERHEAD accuracy signal. "
            "The heavy fixed HP224 package has since OOMed during prepare, and "
            "the 150 x 4K low-memory run was too light, so use the 300 x 4K "
            "tail-window HP224 package."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last == "legacy_e5m2":
        pkg = PACKAGES["hp224_midcalib"]
        lines.append("decision: submit OOM-safe TAIL HP224 next.")
        lines.append(
            "reason: the 225432 E5M2 package is a useful in-flight signal, "
            "but it predates explicit left/tail-window KV calibration and GPTQ disk offload; "
            "a low or zero-score result does not invalidate the current HP224_OFFLOAD queue."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last == "hp224_lowmem":
        pkg = PACKAGES["hp224_midcalib"]
        lines.append("decision: submit MIDCALIB TAIL4K300 HP224 next.")
        lines.append(
            "reason: the 150 x 4K low-memory HP224 package completed but prepared in about 12 minutes, "
            "which confirms it is mainly an OOM-avoidance probe rather than a strong GPTQ precision test. "
            "Use 300 x 4K tail before moving to DENSEQKV/POSTQ/PERHEAD hypotheses."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last in ("hp224", "hp224_lowmem"):
        if has_range_symptom(text):
            pkg = PACKAGES["e5m2_lowmem"]
            lines.append("decision: submit LOWMEM E5M2 range backup next.")
            lines.append("reason: HP224 completed below gate and logs/text suggest e4m3 range/outlier symptoms.")
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        pkg = PACKAGES["denseqkv_lowmem"]
        delta = acc_ori - ATTNSCALE_ACC_ORI
        lines.append("decision: submit LOWMEM DENSEQKV HP224 diagnostic next.")
        lines.append(
            "reason: HP224 completed below gate without explicit range symptoms; "
            "the strongest remaining hypothesis is BF16-calibrated KV scales not "
            "matching W4A16 dense qkv runtime outputs "
            f"(delta vs ATTNSCALE {delta:+.2f}pp)."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last == "hp224_midcalib":
        if has_range_symptom(text):
            pkg = PACKAGES["e5m2_lowmem"]
            lines.append("decision: submit LOWMEM E5M2 range backup next.")
            lines.append("reason: mid-calibration HP224 completed below gate and logs/text suggest e4m3 range/outlier symptoms.")
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        pkg = PACKAGES["hp224_multi8k_diag"]
        lines.append("decision: submit MULTI8K300 HP224 diagnostic next before DENSEQKV.")
        lines.append(
            "reason: 174343 completed at ATTNSCALE quality, but its prepare stage was much faster than the earlier successful 8K multi-adaptive W4A16 run because GPTQ replay was compressed to 300 x 4K tail windows. "
            "Restore 300 x 8K multi-adaptive GPTQ coverage first; KV-scale calibration runs later and does not explain an OOM during GPTQ replay."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last == "hp224_multi8k_diag":
        if has_range_symptom(text):
            pkg = PACKAGES["e5m2_lowmem"]
            lines.append("decision: submit LOWMEM E5M2 range backup next.")
            lines.append("reason: 8K multi-adaptive HP224 completed below gate and logs/text suggest e4m3 range/outlier symptoms.")
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        pkg = PACKAGES["denseqkv_multi8k_diag"]
        lines.append("decision: submit MULTI8K300 DENSEQKV HP224 next, not the 4K DENSEQKV packages.")
        lines.append(
            "reason: if strong 300 x 8K multi-adaptive HP224 is still low without range symptoms, the next hypothesis is BF16-calibrated KV scales not matching W4A16 dense qkv runtime outputs. "
            "Keep the same 300 x 8K multi-adaptive GPTQ coverage so DENSEQKV is the only new variable."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last == "denseqkv_multi8k_diag":
        if has_range_symptom(text):
            pkg = PACKAGES["e5m2_lowmem"]
            lines.append("decision: submit LOWMEM E5M2 range backup next.")
            lines.append("reason: 8K DENSEQKV did not clear the gate and the result suggests e4m3 range/outlier symptoms.")
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        lines.append("decision: STOP and build a POSTQ_MULTI8K HP224 package before using old 4K POSTQ/PERHEAD packages.")
        lines.append(
            "reason: DENSEQKV preserved 300 x 8K multi-adaptive GPTQ coverage. "
            "If it is still low without range symptoms, the next clean hypothesis is measuring KV scales after the final W4A16 artifact, also with 300 x 8K multi-adaptive GPTQ coverage. "
            "Submitting the existing 4K POSTQ/PERHEAD tarballs would reintroduce the calibration-strength confound."
        )
        return "\n".join(lines)

    if last == "denseqkv_midcalib":
        if has_range_symptom(text):
            pkg = PACKAGES["e5m2_lowmem"]
            lines.append("decision: submit LOWMEM E5M2 range backup next.")
            lines.append("reason: comparable DENSEQKV did not clear the gate and the result suggests e4m3 range/outlier symptoms.")
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        pkg = PACKAGES["postq_midcalib"]
        lines.append("decision: submit MIDCALIB POSTQ HP224 package before submitting POSTQ/PERHEAD low-memory variants.")
        lines.append(
            "reason: DENSEQKV matched 300 x 4K calibration strength. If it is still low without range symptoms, "
            "the next clean hypothesis is measuring KV scales after the final W4A16 artifact, also at comparable calibration strength. "
            "Submitting the existing 150 x 4K POSTQ/PERHEAD packages would weaken the comparison."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last in ("denseqkv", "denseqkv_lowmem"):
        if has_range_symptom(text):
            pkg = PACKAGES["e5m2_lowmem"]
            lines.append("decision: submit LOWMEM E5M2 range backup next.")
            lines.append("reason: dense-qkv BF16 alignment did not clear the gate and the result suggests e4m3 range/outlier symptoms.")
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        pkg = PACKAGES["postq_lowmem"]
        lines.append("decision: submit LOWMEM POSTQ HP224 diagnostic next.")
        lines.append("reason: dense-qkv BF16 alignment directly tested the BF16-vs-W4A16 KV-scale mismatch hypothesis; if it is still low without range symptoms, another safe-max-only package is weak evidence.")
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last == "postq_midcalib":
        if has_range_symptom(text):
            pkg = PACKAGES["e5m2_lowmem"]
            lines.append("decision: submit LOWMEM E5M2 range backup next.")
            lines.append("reason: comparable POSTQ aligned scale source with final W4A16 outputs, so remaining explicit range symptoms should be tested with e5m2.")
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        pkg = PACKAGES["perhead_midcalib"]
        lines.append("decision: submit MIDCALIB PERHEAD HP224 BRIDGE next, not the old low-memory per-head variants.")
        lines.append(
            "reason: POSTQ matched 300 x 4K calibration strength and still did not clear the gate without range symptoms. "
            "The next clean FP8KV lever is per-head runtime scaling at comparable calibration strength."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last in ("postq", "postq_lowmem"):
        if has_range_symptom(text):
            pkg = PACKAGES["e5m2_lowmem"]
            lines.append("decision: submit LOWMEM E5M2 range backup next.")
            lines.append("reason: post-GPTQ calibration aligned the scale source with runtime W4A16 outputs, so remaining explicit range symptoms should be tested with e5m2.")
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        pkg = PACKAGES["perhead_lowmem"]
        lines.append("decision: submit LOWMEM PERHEAD HP224 BRIDGE next.")
        lines.append(
            "reason: post-GPTQ calibration is the strongest script-only "
            "scale-source alignment test; if it is still below gate without "
            "range symptoms, the next FP8KV accuracy lever is no longer another "
            "safe-max constant. Use the packaged per-head runtime bridge so scalar "
            "layer scale no longer wastes FP8 resolution across heads."
        )
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last in ("perhead_midcalib", "perhead", "perhead_lowmem"):
        if has_range_symptom(text):
            pkg = PACKAGES["e5m2_lowmem"]
            lines.append("decision: submit LOWMEM E5M2 range backup next.")
            lines.append("reason: per-head scaling reduced scalar-resolution waste; remaining explicit range symptoms should be tested with e5m2.")
            lines.append(format_pkg(pkg))
            return "\n".join(lines)
        lines.append("decision: STOP FP8KV queue for now and fall back to non-FP8 best package.")
        lines.append("reason: scalar calibration, post-GPTQ calibration, and per-head runtime bridge have all been tested below gate without range symptoms.")
        return "\n".join(lines)

    if last == "max448":
        pkg = PACKAGES["e5m2_lowmem"]
        lines.append("decision: submit LOWMEM E5M2 range backup next.")
        lines.append("reason: both e4m3 real-calib precision points are below gate; remaining prepared FP8KV hypothesis is range.")
        lines.append(format_pkg(pkg))
        return "\n".join(lines)

    if last in ("e5m2", "e5m2_lowmem"):
        lines.append("decision: STOP FP8KV queue for now and fall back to non-FP8 best package.")
        lines.append("reason: HP224/DENSEQKV/POSTQ/PERHEAD/E5M2 low-memory variants are the prepared FP8KV precision/range sweep.")
        return "\n".join(lines)

    lines.append("decision: unknown last package; default to MIDCALIB HP224 first.")
    lines.append(format_pkg(PACKAGES["hp224_midcalib"]))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--current",
        action="store_true",
        help="Print the current recommended package and exit.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List known FP8KV packages in queue order and exit.",
    )
    ap.add_argument(
        "--last",
        choices=sorted([*PACKAGES, "legacy_e5m2", "legacy_perhead_095129"]),
        help="Package whose result is being analyzed. Use legacy_e5m2 for the submitted 225432 tarball and legacy_perhead_095129 for the old 095129 tarball.",
    )
    ap.add_argument("--score-json", help="Raw Score JSON string.")
    ap.add_argument("--score-file", help="File containing Score JSON or platform log text.")
    args = ap.parse_args()

    if args.current:
        if CURRENT_PACKAGE_KEY:
            print(format_pkg(PACKAGES[CURRENT_PACKAGE_KEY]))
        else:
            print(CURRENT_ACTION)
        return 0
    if args.list:
        for key in PACKAGE_ORDER:
            pkg = PACKAGES[key]
            print(f"[{pkg.key}] {pkg.tarball}")
            print(f"md5 {pkg.md5}")
            print(f"role: {pkg.role}")
            print()
        return 0

    text = read_text(args)
    score = extract_json_object(text)
    last = infer_last(args, text)
    print(decide(last, text, score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
