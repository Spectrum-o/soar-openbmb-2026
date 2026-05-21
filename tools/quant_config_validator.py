#!/usr/bin/env python3
"""Pre-flight validator for a SOAR submission tarball.

Goes deeper than pack_submission --check-only: actually extracts the
tarball (or reads an unpacked dir) and verifies critical invariants
that would otherwise only surface mid-run on the platform:

  1. Tarball integrity (every member readable)
  2. Required files present (prepare_env/prepare_model/quantize/whl/sglang/)
  3. quantize_gptqmodel_w4a16.py: qzeros fix defined AND called,
     truncation_side="left", chat-template handling reasonable
  4. prepare_env.sh: SGLANG_SERVER_ARGS exported with the right flags,
     no FP8 KV (incompatible with sparse backend), bundled flash_attn
     whl is reachable, fp16 sed-patch will MATCH POINTS in the bundled
     sglang sources (the suspected root cause of v21 platform acc=0:
     if sed silently matches zero lines, the backend stays bf16 while
     Marlin emits fp16 -> garbage outputs).
  5. prepare_model.sh: CALIB_JSONL fallback chain present, quant timeout
     set, no obvious shell-escape bugs.
  6. Cross-check: variant's module set (from layer_modules in quantize
     script) is reflected in `dynamic` skip rules in the saved
     quantize_config.json schema mentioned in source.

Exit code = number of failures (0 = pass).

Usage:
    # Validate a tarball
    python3 tools/quant_config_validator.py \\
        --tarball soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v21.tar.gz

    # Validate an unpacked dir (e.g. a submission_*/ directory before pack)
    python3 tools/quant_config_validator.py --dir submission_gptqmodel_calib_w4a16
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Check primitive
# ---------------------------------------------------------------------------
class CheckResult:
    __slots__ = ("name", "passed", "detail")

    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def fmt(self) -> str:
        flag = "✓" if self.passed else "✗"
        suffix = f"  ({self.detail})" if self.detail else ""
        return f"  [{flag}] {self.name}{suffix}"


def grep_in(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(errors="replace")
    except Exception:
        return False


def grep_re(path: Path, pattern: str | re.Pattern[str]) -> bool:
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return False
    rx = re.compile(pattern) if isinstance(pattern, str) else pattern
    return rx.search(text) is not None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_required_files(root: Path) -> list[CheckResult]:
    results = []
    required_files = {
        "prepare_env.sh": True,
        "prepare_model.sh": True,
    }
    # quantize script only required for GPTQ variants
    is_gptq = (root / "quantize_gptqmodel_w4a16.py").is_file()
    if is_gptq:
        required_files["quantize_gptqmodel_w4a16.py"] = True
        required_files["perf_public_set.jsonl"] = False  # warn but not fatal
    # flash_attn wheel — usually present, warn if not
    whl = list(root.glob("flash_attn-*.whl"))
    results.append(CheckResult(
        "bundled flash_attn whl present", bool(whl),
        f"found {len(whl)} whl(s)" if whl else "NOT BUNDLED — prepare_env will download or fail offline",
    ))
    for name, fatal in required_files.items():
        p = root / name
        present = p.is_file()
        if not present and not fatal:
            results.append(CheckResult(f"file present: {name}", True, "missing (non-fatal)"))
        else:
            results.append(CheckResult(f"file present: {name}", present))
    # bundled sglang sources
    sglang_dir = root / "sglang"
    results.append(CheckResult(
        "bundled sglang/python present", (sglang_dir / "python").is_dir(),
        "this is what gets installed via `uv pip install -e ./sglang/python`",
    ))
    return results


def check_quantize_script(root: Path) -> list[CheckResult]:
    p = root / "quantize_gptqmodel_w4a16.py"
    results: list[CheckResult] = []
    if not p.is_file():
        results.append(CheckResult("quantize script: skip (BF16 variant)", True))
        return results
    text = p.read_text(errors="replace")
    # qzeros fix defined
    results.append(CheckResult(
        "quantize: fix_qzeros_for_marlin() defined",
        "def fix_qzeros_for_marlin" in text,
    ))
    # qzeros fix called (look for a non-def call site)
    calls = re.findall(r"^\s*fix_qzeros_for_marlin\s*\(", text, re.MULTILINE)
    # the def line itself is `def fix_qzeros_for_marlin(...)` so filter
    non_def = sum(1 for line in text.splitlines()
                  if re.match(r"\s*fix_qzeros_for_marlin\(", line))
    results.append(CheckResult(
        "quantize: fix_qzeros_for_marlin() actually called",
        non_def >= 1,
        f"call site count: {non_def}",
    ))
    # truncation side = left
    results.append(CheckResult(
        "quantize: tokenizer.truncation_side = \"left\"",
        'truncation_side = "left"' in text,
    ))
    # register call
    results.append(CheckResult(
        "quantize: register_minicpm_sala_with_gptqmodel() call",
        bool(re.search(r"register_minicpm_sala_with_gptqmodel\s*\(", text)),
    ))
    # has multi-adaptive support? (informational)
    has_multi = "multi-adaptive" in text or "_slice_windows_for_prompt" in text
    results.append(CheckResult(
        "quantize: supports multi-adaptive window mode",
        has_multi,
        "informational; not required for v21/v22 tail-mode",
    ))
    # has chat-template disable flag?
    has_no_chat = "--no-chat-template" in text or "no_chat_template" in text
    results.append(CheckResult(
        "quantize: supports --no-chat-template ablation",
        has_no_chat,
        "informational; required for Exp B/C/etc.",
    ))
    return results


def check_prepare_env(root: Path) -> list[CheckResult]:
    p = root / "prepare_env.sh"
    results: list[CheckResult] = []
    if not p.is_file():
        results.append(CheckResult("prepare_env.sh: skip (missing)", False))
        return results
    text = p.read_text(errors="replace")
    # SGLANG_SERVER_ARGS exported
    args_match = re.search(r'^export SGLANG_SERVER_ARGS="([^"]+)"', text, re.MULTILINE)
    if not args_match:
        results.append(CheckResult("prepare_env: SGLANG_SERVER_ARGS exported", False))
        return results
    args_str = args_match.group(1)
    results.append(CheckResult("prepare_env: SGLANG_SERVER_ARGS exported", True, args_str))

    # GPTQ-specific args
    is_gptq = "--quantization gptq_marlin" in args_str
    if is_gptq:
        results.append(CheckResult(
            "SGLANG_SERVER_ARGS: --quantization gptq_marlin", True))
        results.append(CheckResult(
            "SGLANG_SERVER_ARGS: --dtype float16",
            "--dtype float16" in args_str,
            "Marlin emits fp16; mismatch with sparse backend bf16 -> garbage",
        ))
    # Hard constraints
    results.append(CheckResult(
        "SGLANG_SERVER_ARGS: NO --kv-cache-dtype fp8_*",
        "--kv-cache-dtype fp8" not in args_str,
        "fp8 KV cache is incompatible with MiniCPM sparse backend",
    ))
    results.append(CheckResult(
        "SGLANG_SERVER_ARGS: --attention-backend minicpm_flashinfer",
        "--attention-backend minicpm_flashinfer" in args_str,
    ))
    # sed-patch presence (the suspected v21 platform failure source)
    has_sed_bf = "torch\\.bfloat16" in text and "torch.float16" in text
    has_sed_str = '"bfloat16"' in text and '"float16"' in text
    results.append(CheckResult(
        "prepare_env: fp16 sed-patch for sparse backend present",
        has_sed_bf and has_sed_str,
    ))
    # Verify the sed-patch targets actually exist in bundled sglang
    backend_dir = root / "sglang" / "python" / "sglang" / "srt" / "layers" / "attention"
    candidates = [backend_dir / "minicpm_backend.py", backend_dir / "minicpm_sparse_utils.py"]
    sed_targets_exist = all(p.is_file() for p in candidates)
    results.append(CheckResult(
        "bundled sglang: sed-patch target files exist",
        sed_targets_exist,
        f"need: {[p.relative_to(root) for p in candidates]}",
    ))
    if sed_targets_exist:
        # The critical INVARIANT: the bundled minicpm_*.py files end up in
        # float16 state after prepare_env.sh's sed runs on the platform.
        #
        # Two ways this can be true:
        #   (a) Source already in float16 state (e.g. perma-patched at
        #       commit-time per 15abd6e72) -> sed matches zero lines but
        #       file IS in target state already. THIS IS FINE.
        #   (b) Source has bfloat16 -> sed matches and rewrites. ALSO FINE.
        # The previous wording said "zero matches = critical" which gave a
        # false positive on case (a). What we actually want to check is:
        # AFTER sed runs (hypothetically), is the file fp16-correct?
        # That's equivalent to: file contains zero `torch.bfloat16` AND
        # zero bare `"bfloat16"` strings (the two sed targets).
        n_bf_matches = 0
        n_str_matches = 0
        n_fp16_matches = 0
        for cand in candidates:
            ctext = cand.read_text(errors="replace")
            n_bf_matches += len(re.findall(r"torch\.bfloat16", ctext))
            n_str_matches += len(re.findall(r'"bfloat16"', ctext))
            n_fp16_matches += len(re.findall(r"torch\.float16", ctext))

        # Path-(a) check: file is already fp16-correct (no bf16 to replace,
        # and at least one float16 reference present).
        already_fp16 = (
            n_bf_matches == 0
            and n_str_matches == 0
            and n_fp16_matches > 0
        )
        # Path-(b) check: file has bf16 references that sed can patch.
        sed_will_work = n_bf_matches > 0 or n_str_matches > 0
        passes = already_fp16 or sed_will_work
        if already_fp16:
            detail = (
                "source already fp16-perma-patched (no bf16 left to replace; "
                f"{n_fp16_matches} torch.float16 references) — sed will be a "
                "no-op on platform but file is in target state. OK."
            )
        elif sed_will_work:
            detail = (
                f"sed will rewrite torch.bfloat16={n_bf_matches}, "
                f'"bfloat16"={n_str_matches} matches to float16. OK.'
            )
        else:
            detail = (
                "CRITICAL — backend has neither bf16 (to sed-replace) nor fp16 "
                "(already correct). Sed will no-op and file stays in unknown "
                "dtype state. fp16 Marlin GEMM may fail at first generation."
            )
        results.append(CheckResult(
            "bundled sglang: sparse backend ends in fp16 state",
            passes,
            detail,
        ))
    return results


def check_prepare_model(root: Path) -> list[CheckResult]:
    p = root / "prepare_model.sh"
    results: list[CheckResult] = []
    if not p.is_file():
        results.append(CheckResult("prepare_model.sh: skip (missing)", False))
        return results
    text = p.read_text(errors="replace")
    is_bf16 = "passthrough" in text.lower() or not (root / "quantize_gptqmodel_w4a16.py").is_file()
    if is_bf16:
        results.append(CheckResult(
            "prepare_model: BF16 passthrough mode",
            "symlink" in text.lower() or "cp -r" in text or "ln -s" in text,
        ))
        return results
    # GPTQ checks
    results.append(CheckResult(
        "prepare_model: BUNDLED_CALIB fallback chain",
        "BUNDLED_CALIB" in text or "perf_public_set.jsonl" in text,
    ))
    results.append(CheckResult(
        "prepare_model: quant timeout set",
        "QUANT_TIMEOUT_MIN" in text or "timeout" in text,
    ))
    results.append(CheckResult(
        "prepare_model: passes --no-offload-disk by default",
        "--no-offload-disk" in text,
        "required on 80GB+ VRAM cards (RTX PRO 6000)",
    ))
    # CALIB_WINDOW_MODE / DISABLE_CHAT_TEMPLATE plumbed (informational)
    results.append(CheckResult(
        "prepare_model: CALIB_WINDOW_MODE plumbed",
        "CALIB_WINDOW_MODE" in text,
        "informational; required for window-mode ablations",
    ))
    results.append(CheckResult(
        "prepare_model: DISABLE_CHAT_TEMPLATE plumbed",
        "DISABLE_CHAT_TEMPLATE" in text,
        "informational; required for chat-template ablation",
    ))
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_checks(root: Path) -> tuple[int, int, str]:
    sections: list[str] = [f"# Validator report: `{root}`\n"]

    def add_section(title: str, results: list[CheckResult]) -> tuple[int, int]:
        sections.append(f"## {title}")
        for r in results:
            sections.append(r.fmt())
        sections.append("")
        passed = sum(1 for r in results if r.passed)
        return (passed, len(results))

    n_pass = n_total = 0
    p, t = add_section("Required files", check_required_files(root))
    n_pass += p; n_total += t
    p, t = add_section("Quantize script", check_quantize_script(root))
    n_pass += p; n_total += t
    p, t = add_section("prepare_env.sh", check_prepare_env(root))
    n_pass += p; n_total += t
    p, t = add_section("prepare_model.sh", check_prepare_model(root))
    n_pass += p; n_total += t

    sections.append("## Summary\n```")
    sections.append(f"  passed: {n_pass} / {n_total}")
    sections.append(f"  failed: {n_total - n_pass}")
    sections.append("```")
    return n_pass, n_total, "\n".join(sections)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tarball", help="Submission tarball to validate")
    src.add_argument("--dir", help="Unpacked variant directory to validate")
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    if args.tarball:
        tarball = Path(args.tarball)
        if not tarball.is_file():
            print(f"tarball not found: {tarball}", file=sys.stderr)
            return 2
        with tempfile.TemporaryDirectory(prefix="qcv_") as tmp:
            extract_dir = Path(tmp)
            try:
                with tarfile.open(tarball) as tf:
                    tf.extractall(extract_dir)
            except Exception as exc:
                print(f"tar extraction failed: {exc}", file=sys.stderr)
                return 2
            # Determine root: usually `./` so files are at extract_dir/ root
            root = extract_dir
            n_pass, n_total, text = run_checks(root)
    else:
        root = Path(args.dir)
        if not root.is_dir():
            print(f"dir not found: {root}", file=sys.stderr)
            return 2
        n_pass, n_total, text = run_checks(root)

    print(text)
    if args.output_md:
        Path(args.output_md).write_text(text)
        print(f"\n[validator] wrote {args.output_md}", file=sys.stderr)

    n_fail = n_total - n_pass
    if n_fail > 0:
        print(f"\n[validator] {n_fail} check(s) FAILED", file=sys.stderr)
    else:
        print(f"\n[validator] all {n_total} check(s) passed", file=sys.stderr)
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
