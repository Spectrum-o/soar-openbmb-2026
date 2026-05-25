#!/usr/bin/env python3
"""tools/hard_constraints_lint.py — pre-submission Hard Constraints linter.

The 2026-05-22 / 2026-05-23 string of failed submissions (v3, v5, v5c) all
hit the SAME root cause: `auto_map.AutoConfig` not stripped from config.json.
This was documented in SUBMISSIONS.md "Hard constraints" row 44 since
2026-05-20 but Claude forgot to check the table before designing the fix.

This linter mechanizes the Hard Constraints table — every constraint here
maps directly to one row in SUBMISSIONS.md, cited inline. Run before packing
any variant to catch the next "I forgot rule N" failure.

Usage:
    # Lint a single variant
    python3 tools/hard_constraints_lint.py --variant submission_bf16_config_fix

    # JSON output for scripts / CI
    python3 tools/hard_constraints_lint.py --variant <dir> --format json

    # Lint ALL submission_* variants
    python3 tools/hard_constraints_lint.py --all

Exit code:
    0 — all applicable constraints PASS (or N/A)
    1 — at least one FAIL
    2 — usage / IO error

The linter is INTENTIONALLY conservative: when in doubt, output WARN rather
than FAIL. Don't block packing on uncertain checks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    NA = "N/A"  # constraint doesn't apply to this variant mode


class Mode(str, Enum):
    BF16_SYMLINK = "bf16_symlink"       # prepare_model = ln/cp INPUT to OUTPUT (no config change)
    BF16_CONFIG_FIX = "bf16_config_fix" # prepare_model = symlink + write modified config.json
    GPTQ = "gptq"                       # prepare_model = quantize_gptqmodel_w4a16.py
    LLM_COMPRESSOR = "llm_compressor"   # prepare_model = quantize_llmcompressor_*.py
    RTN = "rtn"                         # prepare_model = quantize_gptq_rtn_sym.py
    UNKNOWN = "unknown"


@dataclass
class Result:
    constraint_id: str
    constraint_title: str
    status: Status
    message: str
    citation: str  # "SUBMISSIONS.md row N" or commit hash
    applies_to: tuple[Mode, ...] = ()


def _extract_sglang_server_args(env_text: str) -> str:
    """Extract just the SGLANG_SERVER_ARGS export line (not comments / heredocs)."""
    lines: list[str] = []
    for line in env_text.splitlines():
        stripped = line.lstrip()
        # Skip shell comments
        if stripped.startswith("#"):
            continue
        if "SGLANG_SERVER_ARGS" in line and ("export" in line or "=" in line):
            lines.append(line)
    return "\n".join(lines)


def _strip_shell_comments(text: str) -> str:
    """Return text with shell-comment lines removed (rough; doesn't handle in-line #)."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def detect_mode(variant_dir: Path) -> Mode:
    """Determine which submission mode the variant uses from prepare_model.sh.

    Mode is determined by what command prepare_model.sh ACTUALLY INVOKES,
    not what it mentions in comments. v5d/v5e/v6's prepare_model.sh has
    "quantize_gptqmodel_w4a16.py" in a comment explaining the auto_map fix,
    but only actually does symlink + json transform.
    """
    pm = variant_dir / "prepare_model.sh"
    if not pm.exists():
        return Mode.UNKNOWN
    raw = pm.read_text(encoding="utf-8", errors="replace")
    code = _strip_shell_comments(raw)  # comments stripped — look only at actual commands

    # Look for actual invocation patterns OR variable assignments referencing
    # the script (`QUANT_SCRIPT="${SCRIPT_DIR}/quantize_X.py"` then later
    # `python3 "$QUANT_SCRIPT"`).
    if re.search(r"(python3?\s+\S*quantize_gptqmodel_w4a16(\.py)?\b)|(\bquantize_gptqmodel_w4a16(\.py)?\b)", code):
        return Mode.GPTQ
    if re.search(r"(python3?\s+\S*quantize_gptq_rtn)|(\bquantize_gptq_rtn\w*\b)", code):
        return Mode.RTN
    if re.search(r"(python3?\s+\S*quantize_llmcompressor)|(\bquantize_llmcompressor\w*\b)|AWQModifier", code):
        return Mode.LLM_COMPRESSOR

    has_config_write = (
        "config.json" in code and
        ("auto_map" in code or "json.load" in code or re.search(r"python3?\s*-", code))
    )
    has_pure_symlink = "ln -sf" in code or "ln -snf" in code or "cp -r" in code
    if has_config_write:
        return Mode.BF16_CONFIG_FIX
    if has_pure_symlink:
        return Mode.BF16_SYMLINK
    return Mode.UNKNOWN


def _read_or_empty(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


# ============================================================================
# CONSTRAINT CHECKERS
# Each function returns a Result. Pure-function over variant_dir contents.
# ============================================================================

def _check_no_fp8_kv(variant_dir: Path, mode: Mode) -> Result:
    """SUBMISSIONS.md row 38: FP8 KV cache incompatible with MiniCPM sparse backend.

    UPDATED 2026-05-25: Earlier "incompatible" verdict was a misread of a single
    crash log path (`.../hopper/...`). 周冠军笔记 05 (智算一队 — Blackwell
    semifinal champion) confirms they shipped FP8 KV Cache as basic optimization.
    The actual blocker is hardware-binary routing, and there are two unlock paths:

      - Path X: --attention-backend minicpm_flashinfer + ENABLE_SM120=1 so
                flashinfer JITs the SM 12.0 FP8 attention cubin.
      - Path Y: bypass FA fp8 kernel — dequant fp8 K/V to bf16 after fetch from
                the paged pool, attention runs bf16 (variant marker file
                `.fp8kv_dequant_bypass` declares opt-in).

    Variants that opt into one of these paths are exempt (N/A). All other
    variants still get FAIL on fp8_* in args.
    """
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    sglang_args = _extract_sglang_server_args(env)
    if not re.search(r"--kv-cache-dtype\s+fp8", sglang_args):
        return Result(
            "C_no_fp8_kv",
            "no --kv-cache-dtype fp8_*",
            Status.PASS,
            "no FP8 KV flag in SGLANG_SERVER_ARGS",
            "SUBMISSIONS.md row 38",
        )
    # FP8 KV present — check for explicit unlock declarations.
    path_x_ok = (
        "ENABLE_SM120" in env
        and "minicpm_flashinfer" in sglang_args
    )
    path_y_ok = (variant_dir / ".fp8kv_dequant_bypass").exists()
    if path_x_ok:
        return Result(
            "C_no_fp8_kv",
            "no --kv-cache-dtype fp8_*",
            Status.NA,
            "FP8 KV via Path X (FlashInfer SM12.0): ENABLE_SM120 + minicpm_flashinfer detected — exempt",
            "SUBMISSIONS.md row 38 + KV_QUANT_PATH_GUIDE.md",
        )
    if path_y_ok:
        return Result(
            "C_no_fp8_kv",
            "no --kv-cache-dtype fp8_*",
            Status.NA,
            "FP8 KV via Path Y (dequant bypass): .fp8kv_dequant_bypass marker file present — exempt",
            "SUBMISSIONS.md row 38 + KV_QUANT_PATH_GUIDE.md",
        )
    return Result(
        "C_no_fp8_kv",
        "no --kv-cache-dtype fp8_*",
        Status.FAIL,
        "SGLANG_SERVER_ARGS contains --kv-cache-dtype fp8_*; neither Path X (ENABLE_SM120 + minicpm_flashinfer) nor Path Y (.fp8kv_dequant_bypass marker) declared. Add the unlock or remove the flag.",
        "SUBMISSIONS.md row 38 + KV_QUANT_PATH_GUIDE.md",
    )


def _check_no_disable_cuda_graph(variant_dir: Path, mode: Mode) -> Result:
    """SUBMISSIONS.md row 39: --disable-cuda-graph is debug only."""
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    sglang_args = _extract_sglang_server_args(env)
    if "--disable-cuda-graph" in sglang_args:
        return Result(
            "C_no_disable_cuda_graph",
            "no --disable-cuda-graph",
            Status.FAIL,
            "SGLANG_SERVER_ARGS contains --disable-cuda-graph; baseline runs WITH cuda graph. Remove.",
            "SUBMISSIONS.md row 39",
        )
    return Result(
        "C_no_disable_cuda_graph",
        "no --disable-cuda-graph",
        Status.PASS,
        "cuda graph enabled (default)",
        "SUBMISSIONS.md row 39",
    )


def _check_flash_attn_installed(variant_dir: Path, mode: Mode) -> Result:
    """SUBMISSIONS.md rows 40-42: SALA requires flash_attention_2, transformers >=5 imports it eagerly, bundle wheels."""
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    has_install_call = (
        "flash_attn" in env or "flash-attn" in env
    )
    has_bundled_wheel = any(
        p.name.startswith("flash_attn-") and p.suffix == ".whl"
        for p in variant_dir.glob("*.whl")
    )
    # Also check symlinks
    has_bundled_wheel = has_bundled_wheel or any(
        p.is_symlink() and "flash_attn" in p.name
        for p in variant_dir.iterdir() if p.is_symlink()
    )
    if mode == Mode.BF16_SYMLINK and not has_install_call and not has_bundled_wheel:
        # BF16 symlink variants may not need flash_attn install if base env has it
        # — but SALA hard-asserts flash_attention_2. Warn, don't fail.
        return Result(
            "C_flash_attn",
            "flash_attn installed or bundled",
            Status.WARN,
            "no flash_attn install step; relies on platform base env having it. SALA's modeling code hard-asserts flash_attention_2 (modeling_minicpm_sala.py:1328).",
            "SUBMISSIONS.md rows 40-42",
        )
    if has_bundled_wheel:
        return Result(
            "C_flash_attn",
            "flash_attn installed or bundled",
            Status.PASS,
            "flash_attn wheel bundled in variant dir",
            "SUBMISSIONS.md rows 40-42",
        )
    if has_install_call:
        return Result(
            "C_flash_attn",
            "flash_attn installed or bundled",
            Status.PASS,
            "prepare_env.sh installs flash_attn",
            "SUBMISSIONS.md rows 40-42",
        )
    return Result(
        "C_flash_attn",
        "flash_attn installed or bundled",
        Status.WARN,
        "no flash_attn install or bundle detected",
        "SUBMISSIONS.md rows 40-42",
    )


def _check_no_github_release_install(variant_dir: Path, mode: Mode) -> Result:
    """SUBMISSIONS.md row 42: bundle wheels, don't download from github at runtime."""
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    has_github_install = bool(re.search(r"uv pip install.*github\.com", env))
    has_fallback_to_bundled = "LOCAL_FLASH_ATTN_WHEEL" in env or "find.*-name.*\\.whl" in env
    if has_github_install and not has_fallback_to_bundled:
        return Result(
            "C_no_github_install",
            "no github.com download in prepare_env",
            Status.FAIL,
            "prepare_env.sh has `uv pip install ... github.com/...` without local-wheel fallback. Platform network may not reach github; bundle the wheel instead.",
            "SUBMISSIONS.md row 42",
        )
    if has_github_install and has_fallback_to_bundled:
        return Result(
            "C_no_github_install",
            "no github.com download (or fallback to bundled)",
            Status.PASS,
            "prepare_env has github URL but falls back to bundled local wheel",
            "SUBMISSIONS.md row 42",
        )
    return Result(
        "C_no_github_install",
        "no github.com download in prepare_env",
        Status.PASS,
        "no github install detected",
        "SUBMISSIONS.md row 42",
    )


def _check_auto_map_autoconfig_stripped(variant_dir: Path, mode: Mode) -> Result:
    """SUBMISSIONS.md row 44 (THE one we missed 3 times):
    Removing auto_map.AutoConfig is required for SGLang's MiniCPMHybridConfig
    to win the isinstance check."""
    if mode == Mode.UNKNOWN:
        return Result(
            "C_auto_map_autoconfig",
            "auto_map.AutoConfig stripped",
            Status.NA,
            "mode unknown",
            "SUBMISSIONS.md row 44",
        )
    pm = _read_or_empty(variant_dir / "prepare_model.sh")
    pe = _read_or_empty(variant_dir / "prepare_env.sh")
    # GPTQ / LLM-compressor / RTN variants strip AutoConfig in their quantize script
    if mode in (Mode.GPTQ, Mode.LLM_COMPRESSOR, Mode.RTN):
        # check that the bundled quantize script does this
        scripts = list(variant_dir.glob("quantize_*.py"))
        if not scripts:
            scripts = [variant_dir / "quantize_gptqmodel_w4a16.py"]
        for s in scripts:
            text = _read_or_empty(s)
            if "auto_map" in text and "AutoConfig" in text and ("pop" in text or "del " in text):
                return Result(
                    "C_auto_map_autoconfig",
                    "auto_map.AutoConfig stripped",
                    Status.PASS,
                    f"{s.name} strips auto_map.AutoConfig in quant pipeline",
                    "SUBMISSIONS.md row 44",
                )
        return Result(
            "C_auto_map_autoconfig",
            "auto_map.AutoConfig stripped",
            Status.FAIL,
            "no quantize script found that strips auto_map.AutoConfig. SGLang's MiniCPMHybridConfig isinstance() will fail. This is the v3/v5/v5c root cause.",
            "SUBMISSIONS.md row 44",
        )
    if mode == Mode.BF16_CONFIG_FIX:
        if "auto_map" in pm and ("AutoConfig" in pm or "auto_map.AutoConfig" in pm):
            return Result(
                "C_auto_map_autoconfig",
                "auto_map.AutoConfig stripped",
                Status.PASS,
                "prepare_model.sh handles auto_map.AutoConfig",
                "SUBMISSIONS.md row 44",
            )
        return Result(
            "C_auto_map_autoconfig",
            "auto_map.AutoConfig stripped",
            Status.FAIL,
            "BF16_CONFIG_FIX mode but prepare_model.sh doesn't seem to handle auto_map.AutoConfig",
            "SUBMISSIONS.md row 44",
        )
    if mode == Mode.BF16_SYMLINK:
        # Pure symlink WITHOUT config.json modification will inherit the BF16
        # source's auto_map.AutoConfig, which fails the isinstance check.
        # THIS IS WHY v3/v5/v5c CRASHED.
        return Result(
            "C_auto_map_autoconfig",
            "auto_map.AutoConfig stripped",
            Status.FAIL,
            "BF16_SYMLINK mode does NOT strip auto_map.AutoConfig from config.json. "
            "v3/v5/v5c failed this way (2026-05-23). Use BF16_CONFIG_FIX mode or add config.json transformation.",
            "SUBMISSIONS.md row 44",
        )
    return Result(
        "C_auto_map_autoconfig",
        "auto_map.AutoConfig stripped",
        Status.WARN,
        f"unhandled mode {mode}",
        "SUBMISSIONS.md row 44",
    )


def _check_no_derived_props_serialized(variant_dir: Path, mode: Mode) -> Result:
    """SUBMISSIONS.md row 45: don't serialize has_sparse_attention etc."""
    if mode == Mode.UNKNOWN:
        return Result(
            "C_no_derived_props",
            "no derived read-only properties in config.json",
            Status.NA,
            "mode unknown",
            "SUBMISSIONS.md row 45",
        )
    pm = _read_or_empty(variant_dir / "prepare_model.sh")
    if mode in (Mode.GPTQ, Mode.LLM_COMPRESSOR, Mode.RTN):
        scripts = list(variant_dir.glob("quantize_*.py"))
        for s in scripts:
            text = _read_or_empty(s)
            if "has_sparse_attention" in text and ("pop" in text or "del " in text):
                return Result(
                    "C_no_derived_props",
                    "no derived read-only properties in config.json",
                    Status.PASS,
                    f"{s.name} strips derived properties",
                    "SUBMISSIONS.md row 45",
                )
        return Result(
            "C_no_derived_props",
            "no derived read-only properties in config.json",
            Status.WARN,
            "no explicit strip of has_sparse_attention etc. (maybe input config doesn't have them either)",
            "SUBMISSIONS.md row 45",
        )
    if mode == Mode.BF16_CONFIG_FIX:
        if "has_sparse_attention" in pm:
            return Result(
                "C_no_derived_props",
                "no derived read-only properties in config.json",
                Status.PASS,
                "prepare_model.sh strips has_sparse_attention",
                "SUBMISSIONS.md row 45",
            )
        return Result(
            "C_no_derived_props",
            "no derived read-only properties in config.json",
            Status.WARN,
            "BF16_CONFIG_FIX doesn't explicitly strip derived properties (may be OK if input config doesn't have them)",
            "SUBMISSIONS.md row 45",
        )
    if mode == Mode.BF16_SYMLINK:
        return Result(
            "C_no_derived_props",
            "no derived read-only properties in config.json",
            Status.PASS,
            "BF16 source config.json comes pre-clean (assuming OpenBMB doesn't put derived props)",
            "SUBMISSIONS.md row 45",
        )
    return Result(
        "C_no_derived_props",
        "no derived read-only properties in config.json",
        Status.WARN,
        f"unhandled mode {mode}",
        "SUBMISSIONS.md row 45",
    )


def _check_marlin_sym_true(variant_dir: Path, mode: Mode) -> Result:
    """SUBMISSIONS.md row 30: Marlin requires (4, True). Only applies to GPTQ mode."""
    if mode != Mode.GPTQ:
        return Result(
            "C_marlin_sym_true",
            "Marlin (4, True) required",
            Status.NA,
            "not GPTQ mode",
            "SUBMISSIONS.md row 30",
        )
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    if "gptq_marlin" not in env:
        return Result(
            "C_marlin_sym_true",
            "Marlin (4, True) required",
            Status.NA,
            "GPTQ mode but not using gptq_marlin loader",
            "SUBMISSIONS.md row 30",
        )
    # Check default sym setting in quant script + check for GPTQ_SYM env var override
    scripts = list(variant_dir.glob("quantize_*.py"))
    sym_default_true = False
    has_env_override = False
    for s in scripts:
        text = _read_or_empty(s)
        if 'GPTQ_SYM' in text:
            has_env_override = True
            # env-var-driven default: look for `os.environ.get("GPTQ_SYM", "True")` etc
            if re.search(r'GPTQ_SYM["\']?\s*,\s*["\']True', text):
                sym_default_true = True
        if re.search(r'"sym"\s*:\s*True', text) or re.search(r"'sym'\s*:\s*True", text):
            sym_default_true = True
    if sym_default_true:
        msg = "default sym=True in quant config"
        if has_env_override:
            msg += " (env-var GPTQ_SYM defaults to True; caller must not flip to False when using gptq_marlin)"
        return Result("C_marlin_sym_true", "Marlin (4, True) required", Status.PASS, msg, "SUBMISSIONS.md row 30")
    return Result(
        "C_marlin_sym_true",
        "Marlin (4, True) required",
        Status.WARN,
        "couldn't statically verify sym=True in quant config",
        "SUBMISSIONS.md row 30",
    )


def _check_gptqmodel_supported_models(variant_dir: Path, mode: Mode) -> Result:
    """SUBMISSIONS.md rows 51-52: must mutate SUPPORTED_MODELS, not just MODEL_MAP."""
    if mode != Mode.GPTQ:
        return Result(
            "C_gptqmodel_supported_models",
            "SUPPORTED_MODELS mutated",
            Status.NA,
            "not GPTQ mode",
            "SUBMISSIONS.md rows 51-52",
        )
    scripts = list(variant_dir.glob("quantize_gptqmodel*.py"))
    for s in scripts:
        text = _read_or_empty(s)
        if "SUPPORTED_MODELS" in text and ("append" in text or "extend" in text or "= list" in text or ".add(" in text):
            return Result(
                "C_gptqmodel_supported_models",
                "SUPPORTED_MODELS mutated",
                Status.PASS,
                "quant script mutates gptqmodel.models.auto.SUPPORTED_MODELS",
                "SUBMISSIONS.md rows 51-52",
            )
    return Result(
        "C_gptqmodel_supported_models",
        "SUPPORTED_MODELS mutated",
        Status.FAIL,
        "GPTQ variant doesn't mutate SUPPORTED_MODELS. v10/v11/v12 failed exactly this way.",
        "SUBMISSIONS.md rows 51-52",
    )


def _check_qzeros_fix_hardened(variant_dir: Path, mode: Mode) -> Result:
    """Hardened fix_qzeros_for_marlin (commit d8aaaf1e4). Critical for GPTQ.

    Original fix only globbed `model-*.safetensors` and required exact int32 dtype.
    Hardened version handles uint32, single-file shard, per-element mask, HARD_EXIT.
    Without this, qzeros stay at 7 on platform → +1*scale bias → acc=0."""
    if mode != Mode.GPTQ:
        return Result(
            "C_qzeros_hardened",
            "qzeros fix hardened",
            Status.NA,
            "not GPTQ mode",
            "feedback_gptqmodel_qzeros_bug.md, commit d8aaaf1e4",
        )
    scripts = list(variant_dir.glob("quantize_gptqmodel*.py"))
    for s in scripts:
        text = _read_or_empty(s)
        has_glob_star = '"*.safetensors"' in text or "'*.safetensors'" in text or "glob('*.safetensors')" in text
        has_uint32_view = "torch.uint32" in text or "view(torch.int32)" in text
        has_post_check = "POST-CHECK" in text or "post_check" in text or "verified=True" in text or "verified = True" in text
        has_hard_exit = "HARD EXIT" in text or "HARD_EXIT" in text or ("raise" in text and "qzero" in text.lower())
        if has_glob_star and has_uint32_view and (has_post_check or has_hard_exit):
            return Result(
                "C_qzeros_hardened",
                "qzeros fix hardened",
                Status.PASS,
                "fix_qzeros_for_marlin has glob('*.safetensors') + uint32 + post-check",
                "feedback_gptqmodel_qzeros_bug.md, commit d8aaaf1e4",
            )
    return Result(
        "C_qzeros_hardened",
        "qzeros fix hardened",
        Status.FAIL,
        "fix_qzeros_for_marlin missing hardening markers. v21/v22 silently no-op'd on platform with old version.",
        "feedback_gptqmodel_qzeros_bug.md, commit d8aaaf1e4",
    )


def _check_h4_tokenizer_overwrite(variant_dir: Path, mode: Mode) -> Result:
    """H4 fix: copy_runtime_assets must overwrite tokenizer (no `if not target.exists()` guard).
    Without this, GPTQModel.save() re-serialized tokenizer drifts from base."""
    if mode != Mode.GPTQ:
        return Result(
            "C_h4_tokenizer_overwrite",
            "H4 tokenizer overwrite",
            Status.NA,
            "not GPTQ mode",
            "feedback_gptqmodel_tokenizer_reserialize.md, commit 57f9cef06",
        )
    scripts = list(variant_dir.glob("quantize_gptqmodel*.py"))
    for s in scripts:
        text = _read_or_empty(s)
        if "copy_runtime_assets" in text:
            # Check for the bug pattern: `if not target.exists()` BEFORE shutil.copy
            if re.search(r"if\s+not\s+target\.exists\(\)\s*:\s*\n\s*shutil\.copy", text):
                return Result(
                    "C_h4_tokenizer_overwrite",
                    "H4 tokenizer overwrite",
                    Status.FAIL,
                    "copy_runtime_assets has `if not target.exists()` guard that silently keeps GPTQModel's re-serialized tokenizer",
                    "feedback_gptqmodel_tokenizer_reserialize.md, commit 57f9cef06",
                )
            return Result(
                "C_h4_tokenizer_overwrite",
                "H4 tokenizer overwrite",
                Status.PASS,
                "copy_runtime_assets unconditionally overwrites (no `if not target.exists()` guard found)",
                "feedback_gptqmodel_tokenizer_reserialize.md, commit 57f9cef06",
            )
    return Result(
        "C_h4_tokenizer_overwrite",
        "H4 tokenizer overwrite",
        Status.WARN,
        "no copy_runtime_assets function detected in quant scripts",
        "feedback_gptqmodel_tokenizer_reserialize.md, commit 57f9cef06",
    )


def _check_h5_transformers_pin(variant_dir: Path, mode: Mode) -> Result:
    """H5: transformers must be pinned to 4.57.1 (or compatible). Platform default is 5.9.0.
    Verified by 1849 success (acc 58.61) vs v17/v21/v22 acc=0 without pin."""
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    has_force_reinstall = "--force-reinstall" in env and "transformers" in env
    has_pin = "transformers==4.57.1" in env or 'TRANSFORMERS_PIN' in env
    has_hub_enforce = "huggingface-hub" in env and "<1.0" in env
    if has_force_reinstall and has_pin and has_hub_enforce:
        return Result(
            "C_h5_transformers_pin",
            "transformers 4.57.1 force-reinstall + hub cascade",
            Status.PASS,
            "prepare_env pins transformers 4.57.1 + enforces hub<1.0",
            "feedback_h5_transformers_runtime_drift.md (VERIFIED 2026-05-22 platform=58.61)",
        )
    if mode == Mode.BF16_SYMLINK and not has_pin:
        # BF16 without transformers control is still vulnerable to model_type list dump
        return Result(
            "C_h5_transformers_pin",
            "transformers 4.57.1 force-reinstall + hub cascade",
            Status.FAIL,
            "BF16 path needs the H5 fix too. Platform's transformers 5.9.0 broke SALA loading. v3/v5 verified this.",
            "feedback_h5_transformers_runtime_drift.md",
        )
    if not has_pin:
        return Result(
            "C_h5_transformers_pin",
            "transformers 4.57.1 force-reinstall + hub cascade",
            Status.WARN,
            "no explicit transformers pin; may rely on platform default (risky)",
            "feedback_h5_transformers_runtime_drift.md",
        )
    return Result(
        "C_h5_transformers_pin",
        "transformers 4.57.1 force-reinstall + hub cascade",
        Status.WARN,
        "partial H5 implementation; review prepare_env.sh manually",
        "feedback_h5_transformers_runtime_drift.md",
    )


def _check_import_smoke_test(variant_dir: Path, mode: Mode) -> Result:
    """Install steps should be followed by an `import transformers` smoke test.
    Verified necessary by 2026-05-22 18:33 v24 failure (hub=1.16.0 metadata OK
    but import broken)."""
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    has_import = "python3 -c" in env and "import transformers" in env
    has_smoke_block = "smoke test" in env.lower() or "[smoke]" in env
    if has_import and has_smoke_block:
        return Result(
            "C_import_smoke",
            "import smoke test after install",
            Status.PASS,
            "prepare_env runs import-level smoke test after install",
            "feedback_h5_transformers_runtime_drift.md",
        )
    if mode == Mode.BF16_SYMLINK:
        return Result(
            "C_import_smoke",
            "import smoke test after install",
            Status.NA,
            "BF16 symlink doesn't install packages",
            "feedback_h5_transformers_runtime_drift.md",
        )
    return Result(
        "C_import_smoke",
        "import smoke test after install",
        Status.WARN,
        "no `python3 -c 'import transformers'` smoke test in prepare_env",
        "feedback_h5_transformers_runtime_drift.md",
    )


def _check_hard_guard_on_install(variant_dir: Path, mode: Mode) -> Result:
    """HARD GUARD on package versions after install — catches silent uv failures.
    Verified necessary by 2026-05-22 18:18 v24 failure (gptqmodel NOT_INSTALLED but uv exit 0)."""
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    has_hard_guard = "FATAL" in env and "exit 1" in env and "python_pkg_exact_version" in env
    if has_hard_guard:
        return Result(
            "C_hard_guard",
            "HARD GUARD after install",
            Status.PASS,
            "prepare_env has FATAL/exit 1 hard guards on package versions",
            "feedback_consult_past_failures_before_fix.md",
        )
    if mode == Mode.BF16_SYMLINK:
        return Result(
            "C_hard_guard",
            "HARD GUARD after install",
            Status.NA,
            "BF16 symlink doesn't install packages",
            "feedback_consult_past_failures_before_fix.md",
        )
    return Result(
        "C_hard_guard",
        "HARD GUARD after install",
        Status.WARN,
        "no HARD GUARD detected after install. uv can silently report success while packages are missing.",
        "feedback_consult_past_failures_before_fix.md",
    )


def _check_sglang_server_args_present(variant_dir: Path, mode: Mode) -> Result:
    """The platform contract: prepare_env.sh must export SGLANG_SERVER_ARGS."""
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    if "export SGLANG_SERVER_ARGS=" in env:
        return Result(
            "C_sglang_server_args",
            "SGLANG_SERVER_ARGS exported",
            Status.PASS,
            "prepare_env.sh exports SGLANG_SERVER_ARGS",
            "(platform contract)",
        )
    return Result(
        "C_sglang_server_args",
        "SGLANG_SERVER_ARGS exported",
        Status.FAIL,
        "prepare_env.sh does NOT export SGLANG_SERVER_ARGS. Platform will fall back to baseline default, defeating the variant.",
        "(platform contract)",
    )


def _check_attention_backend_minicpm(variant_dir: Path, mode: Mode) -> Result:
    """SALA needs a minicpm_* backend; both flashinfer and flashattn dispatch
    through MiniCPMSparseBackend (attention_registry.py:180-197). The kernel
    underneath differs (flash_attn vs flashinfer), but either gets SALA's
    sparse path. Path Y dequant-bypass intentionally uses minicpm_flashattn
    so this check accepts both."""
    env = _read_or_empty(variant_dir / "prepare_env.sh")
    if "--attention-backend minicpm_flashinfer" in env:
        return Result(
            "C_attention_backend",
            "--attention-backend minicpm_flash{attn,infer}",
            Status.PASS,
            "uses minicpm_flashinfer backend",
            "(SALA architecture requirement)",
        )
    if "--attention-backend minicpm_flashattn" in env:
        return Result(
            "C_attention_backend",
            "--attention-backend minicpm_flash{attn,infer}",
            Status.PASS,
            "uses minicpm_flashattn backend (sparse path same; FA kernel under the hood)",
            "(SALA architecture requirement)",
        )
    return Result(
        "C_attention_backend",
        "--attention-backend minicpm_flash{attn,infer}",
        Status.FAIL,
        "SGLANG_SERVER_ARGS doesn't have a minicpm_* attention backend. SALA hybrid attention won't dispatch correctly.",
        "(SALA architecture requirement)",
    )


# ============================================================================
# Registry
# ============================================================================

CHECKERS: tuple[Callable[[Path, Mode], Result], ...] = (
    _check_sglang_server_args_present,
    _check_attention_backend_minicpm,
    _check_no_fp8_kv,
    _check_no_disable_cuda_graph,
    _check_flash_attn_installed,
    _check_no_github_release_install,
    _check_auto_map_autoconfig_stripped,
    _check_no_derived_props_serialized,
    _check_marlin_sym_true,
    _check_gptqmodel_supported_models,
    _check_qzeros_fix_hardened,
    _check_h4_tokenizer_overwrite,
    _check_h5_transformers_pin,
    _check_import_smoke_test,
    _check_hard_guard_on_install,
)


def lint_variant(variant_dir: Path) -> tuple[Mode, list[Result]]:
    mode = detect_mode(variant_dir)
    results = [check(variant_dir, mode) for check in CHECKERS]
    return mode, results


def format_human(variant_dir: Path, mode: Mode, results: list[Result]) -> str:
    lines: list[str] = []
    lines.append(f"=== {variant_dir.name} ===")
    lines.append(f"mode: {mode.value}")
    counts = {s: 0 for s in Status}
    for r in results:
        counts[r.status] += 1
    lines.append(
        f"summary: {counts[Status.PASS]} PASS / {counts[Status.WARN]} WARN / "
        f"{counts[Status.FAIL]} FAIL / {counts[Status.NA]} N/A"
    )
    lines.append("")
    for r in results:
        if r.status == Status.NA:
            continue
        emoji = {"PASS": "✓", "FAIL": "✗", "WARN": "!", "N/A": "·"}[r.status.value]
        lines.append(f"  [{emoji} {r.status.value:4s}] {r.constraint_title}")
        if r.status in (Status.FAIL, Status.WARN):
            lines.append(f"           {r.message}")
            lines.append(f"           cite: {r.citation}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=Path, help="single variant dir to lint")
    ap.add_argument("--all", action="store_true", help="lint all submission_* variants in repo root")
    ap.add_argument("--format", choices=("human", "json"), default="human")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any FAIL (default: 0)")
    args = ap.parse_args()

    if not args.variant and not args.all:
        ap.error("specify --variant or --all")

    variants: list[Path] = []
    if args.variant:
        variants.append(args.variant)
    if args.all:
        variants.extend(sorted(REPO_ROOT.glob("submission_*")))

    overall = {"variants": [], "any_fail": False}
    for v in variants:
        v = v.resolve()
        if not v.is_dir():
            print(f"skip {v} (not a directory)", file=sys.stderr)
            continue
        mode, results = lint_variant(v)
        any_fail = any(r.status == Status.FAIL for r in results)
        if any_fail:
            overall["any_fail"] = True
        if args.format == "human":
            print(format_human(v, mode, results))
            print()
        else:
            overall["variants"].append({
                "variant": str(v.relative_to(REPO_ROOT)) if v.is_relative_to(REPO_ROOT) else str(v),
                "mode": mode.value,
                "results": [asdict(r) for r in results],
                "any_fail": any_fail,
            })

    if args.format == "json":
        print(json.dumps(overall, indent=2, default=str))

    if args.strict and overall["any_fail"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
