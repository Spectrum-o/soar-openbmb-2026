#!/usr/bin/env python3
"""Audit FP8KV real-calibration submission packages.

This catches the 2026-05-30 failure class where `kv_calibrate.py` loaded
SOAR `perf_public_set.jsonl` rows but did not read their `question` field,
so every calibration sample could be skipped and all k/v scales became 1.0.

Usage:
    python3 tools/audit_fp8kv_realcalib_package.py \
        soar_fp8kv_REALCALIB_TAIL_HP224_20260530_234154.tar.gz

The expected FP8 dtype and safe-max are inferred from the path name:
    * HP224 -> fp8_e4m3, safe-max 224
    * MAX448 -> fp8_e4m3, safe-max 448
    * E5M2 -> fp8_e5m2, safe-max 224

It also requires tail-window KV calibration (`truncation_side=left`), because
SOAR long prompts put the task near the end.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PackageText:
    names: list[str]
    symlinks: list[str]
    source_is_tar: bool
    kv_calibrate: str
    quantize: str
    overlay_tool: str
    prepare_env: str
    prepare_model: str
    calib_jsonl: str
    kv_cache: str = ""
    kv_scale_utils: str = ""
    memory_pool: str = ""
    minicpm_backend: str = ""
    minicpm_kernels: str = ""


def infer_expected(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if "e5m2" in name:
        return "fp8_e5m2", "224"
    if "max448" in name:
        return "fp8_e4m3", "448"
    if "hp224" in name or "realcalib" in name:
        return "fp8_e4m3", "224"
    raise ValueError(
        f"cannot infer expected dtype/safe-max from {path.name}; "
        "include HP224, MAX448, or E5M2 in the filename"
    )


def _norm_tar_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def _read_tar_member(tf: tarfile.TarFile, member_name: str) -> str:
    by_name = {_norm_tar_name(m.name): m for m in tf.getmembers()}
    member = by_name.get(member_name)
    if member is None:
        raise FileNotFoundError(f"missing {member_name} in tarball")
    if member.issym() or member.islnk():
        raise FileNotFoundError(f"{member_name} is a symlink in tarball")
    f = tf.extractfile(member)
    if f is None:
        raise FileNotFoundError(f"{member_name} is not a regular file")
    return f.read().decode("utf-8", "replace")


def _read_tar_member_optional(tf: tarfile.TarFile, member_name: str) -> str:
    try:
        return _read_tar_member(tf, member_name)
    except (FileNotFoundError, KeyError):
        return ""


def read_package(path: Path) -> PackageText:
    if path.is_dir():
        quantize_path = path / "quantize_gptqmodel_w4a16.py"
        return PackageText(
            names=[str(p.relative_to(path)) for p in path.rglob("*")],
            symlinks=[str(p.relative_to(path)) for p in path.rglob("*") if p.is_symlink()],
            source_is_tar=False,
            kv_calibrate=(path / "kv_calibrate.py").read_text(errors="replace"),
            quantize=quantize_path.read_text(errors="replace") if quantize_path.exists() else "",
            overlay_tool=(
                (path / "apply_lightning_skip_overlay.py").read_text(errors="replace")
                if (path / "apply_lightning_skip_overlay.py").exists()
                else ""
            ),
            prepare_env=(path / "prepare_env.sh").read_text(errors="replace"),
            prepare_model=(path / "prepare_model.sh").read_text(errors="replace"),
            calib_jsonl=(path / "perf_public_set.jsonl").read_text(errors="replace"),
            kv_cache=(path / "sglang/python/sglang/srt/layers/quantization/kv_cache.py").read_text(errors="replace") if (path / "sglang/python/sglang/srt/layers/quantization/kv_cache.py").exists() else "",
            kv_scale_utils=(path / "sglang/python/sglang/srt/layers/quantization/kv_scale_utils.py").read_text(errors="replace") if (path / "sglang/python/sglang/srt/layers/quantization/kv_scale_utils.py").exists() else "",
            memory_pool=(path / "sglang/python/sglang/srt/mem_cache/memory_pool.py").read_text(errors="replace") if (path / "sglang/python/sglang/srt/mem_cache/memory_pool.py").exists() else "",
            minicpm_backend=(path / "sglang/python/sglang/srt/layers/attention/minicpm_backend.py").read_text(errors="replace") if (path / "sglang/python/sglang/srt/layers/attention/minicpm_backend.py").exists() else "",
            minicpm_kernels=(path / "sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py").read_text(errors="replace") if (path / "sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py").exists() else "",
        )

    with tarfile.open(path, "r:gz") as tf:
        names = [_norm_tar_name(m.name) for m in tf.getmembers()]
        symlinks = [_norm_tar_name(m.name) for m in tf.getmembers() if m.issym() or m.islnk()]
        return PackageText(
            names=names,
            symlinks=symlinks,
            source_is_tar=True,
            kv_calibrate=_read_tar_member_optional(tf, "kv_calibrate.py"),
            quantize=_read_tar_member_optional(tf, "quantize_gptqmodel_w4a16.py"),
            overlay_tool=_read_tar_member_optional(tf, "apply_lightning_skip_overlay.py"),
            prepare_env=_read_tar_member_optional(tf, "prepare_env.sh"),
            prepare_model=_read_tar_member_optional(tf, "prepare_model.sh"),
            calib_jsonl=_read_tar_member_optional(tf, "perf_public_set.jsonl"),
            kv_cache=_read_tar_member_optional(tf, "sglang/python/sglang/srt/layers/quantization/kv_cache.py"),
            kv_scale_utils=_read_tar_member_optional(tf, "sglang/python/sglang/srt/layers/quantization/kv_scale_utils.py"),
            memory_pool=_read_tar_member_optional(tf, "sglang/python/sglang/srt/mem_cache/memory_pool.py"),
            minicpm_backend=_read_tar_member_optional(tf, "sglang/python/sglang/srt/layers/attention/minicpm_backend.py"),
            minicpm_kernels=_read_tar_member_optional(tf, "sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py"),
        )


def _extract_default_int(shell_text: str, var_name: str, default: int) -> int:
    marker = f'{var_name}="${{{var_name}:-'
    start = shell_text.find(marker)
    if start < 0:
        return default
    start += len(marker)
    end = shell_text.find("}", start)
    if end < 0:
        return default
    try:
        return int(shell_text[start:end])
    except ValueError:
        return default


def _contains_in_order(text: str, *needles: str) -> bool:
    pos = -1
    for needle in needles:
        pos = text.find(needle, pos + 1)
        if pos < 0:
            return False
    return True


def audit(path: Path) -> bool:
    expected_dtype, expected_safe_max = infer_expected(path)
    pkg = read_package(path)
    num_samples = _extract_default_int(pkg.prepare_model, "KV_CALIB_NUM_SAMPLES", 64)
    postq_mode = "--post-quant-kv-only" in pkg.prepare_model
    denseqkv_mode = (
        "denseqkv" in path.name.lower()
        or "ENABLE_DENSE_QKV_BF16" in pkg.prepare_model
    )
    perhead_mode = (
        "perhead" in path.name.lower()
        or "KV_CALIB_EMIT_PER_HEAD" in pkg.prepare_model
        or "--emit-per-head-scales" in pkg.prepare_model
        or "--post-quant-kv-emit-per-head-scales" in pkg.prepare_model
    )

    checks: list[tuple[str, bool, str]] = []
    checks.append(("no __pycache__", not any("__pycache__" in n for n in pkg.names), ""))
    if pkg.source_is_tar:
        critical_symlinks = [
            n for n in pkg.symlinks
            if n == "sglang"
            or n.startswith("sglang/")
            or n == "perf_public_set.jsonl"
            or n.startswith("flash_attn-")
            or n == "apply_lightning_skip_overlay.py"
        ]
        checks.append((
            "tarball has no critical symlink members",
            not critical_symlinks,
            ", ".join(critical_symlinks),
        ))
    checks.append(("reads SOAR question field", 'sample.get("question")' in pkg.kv_calibrate or 'row.get("question")' in pkg.quantize, ""))
    checks.append(("sets tokenizer truncation_side", "tokenizer.truncation_side = args.truncation_side" in pkg.kv_calibrate or "tokenizer.truncation_side = args.post_quant_kv_truncation_side" in pkg.quantize, ""))
    checks.append(("default truncation_side left", 'default="left"' in pkg.kv_calibrate or 'default="left"' in pkg.quantize, ""))
    if postq_mode:
        checks.append(("prepare_model uses post-GPTQ KV calibration", "--post-quant-kv-only" in pkg.prepare_model and "post_gptq_qzeros_fixed" in pkg.quantize, ""))
        checks.append(("prepare_model passes post-quant truncation-side left", 'KV_CALIB_TRUNCATION_SIDE="${KV_CALIB_TRUNCATION_SIDE:-left}"' in pkg.prepare_model and "--post-quant-kv-truncation-side" in pkg.prepare_model, ""))
        checks.append((
            "postq reload uses remote AutoConfig then restores SGLang config",
            "_temporarily_restore_remote_autoconfig_for_postq_reload" in pkg.quantize
            and "_infer_remote_autoconfig_value" in pkg.quantize
            and "_restore_postq_reload_config" in pkg.quantize
            and 'auto_map["AutoConfig"]' in pkg.quantize
            and "finally:" in pkg.quantize
            and "_restore_postq_reload_config(output_dir, original_config_text)" in pkg.quantize,
            "GPTQModel.load uses HF AutoModel trust_remote_code; config must be remote MiniCPMSALAConfig during POSTQ reload, then final serving config must drop AutoConfig again for SGLang MiniCPMHybridConfig",
        ))
        checks.append((
            "postq reload config window wraps GPTQModel.load and tokenizer fallback",
            _contains_in_order(
                pkg.quantize,
                "original_config_text = _temporarily_restore_remote_autoconfig_for_postq_reload",
                "qmodel = GPTQModel.load(",
                "tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)",
                "finally:",
                "_restore_postq_reload_config(output_dir, original_config_text)",
            ),
            "AutoConfig must be remote while HF AutoModel and tokenizer fallback inspect the artifact, and must be restored before final serving",
        ))
        checks.append((
            "postq reload adds HF-name skip aliases before config snapshot",
            "_augment_dynamic_skip_aliases_for_reload" in pkg.quantize
            and '"qkv_proj": ("q_proj", "k_proj", "v_proj")' in pkg.quantize
            and '"gate_up_proj": ("gate_proj", "up_proj")' in pkg.quantize
            and "quantize_config.json" in pkg.quantize
            and "quantization_config" in pkg.quantize
            and _contains_in_order(
                pkg.quantize,
                "_augment_dynamic_skip_aliases_for_reload(output_dir)",
                "original_config_text = _temporarily_restore_remote_autoconfig_for_postq_reload",
                "qmodel = GPTQModel.load(",
            ),
            "selective overlay writes SGLang fused qkv_proj/gate_up_proj skip rules, but GPTQModel reload sees HF q_proj/k_proj/v_proj and gate_proj/up_proj names",
        ))
        checks.append((
            "postq does not mix MiniCPMHybridConfig with remote HF model class",
            "register_minicpm_sala_hf_config()" not in pkg.quantize
            and 'AutoConfig.register("minicpm_sala", MiniCPMHybridConfig)' not in pkg.quantize,
            "this legacy POSTQ reload pattern caused HF auto_factory config_class mismatch on 2026-06-02",
        ))
        checks.append((
            "postq has GPTQModel 7 transformers compat shim",
            "_stub_transformers_for_gptqmodel_7" in pkg.quantize
            and "PreTrainedConfig" in pkg.quantize
            and "_class_to_module" in pkg.quantize
            and "_objects" in pkg.quantize,
            "POSTQ reload imports gptqmodel after GPTQ; platform transformers 4.57.1 needs the PreTrainedConfig alias for gptqmodel 7.0",
        ))
        bool_dynamic_fragments = [
            '"-:.*o_gate$": True',
            '"-:.*z_proj$": True',
            '"-:.*o_norm$": True',
            '"-:.*q_norm$": True',
            '"-:.*k_norm$": True',
            '] = True',
            "dict[str, bool]",
        ]
        bad_dynamic = [
            frag for frag in bool_dynamic_fragments
            if frag in pkg.quantize or frag in pkg.overlay_tool
        ]
        checks.append((
            "postq dynamic skip values are GPTQModel-dict compatible",
            not bad_dynamic,
            ", ".join(bad_dynamic)
            or "GPTQModel 7 reload normalizes dynamic values with .items(); use {} for -: skip values, not true",
        ))
    else:
        checks.append(("prepare_model passes truncation-side left", 'KV_CALIB_TRUNCATION_SIDE="${KV_CALIB_TRUNCATION_SIDE:-left}"' in pkg.prepare_model and "--truncation-side" in pkg.prepare_model, ""))
    checks.append(("zero-forward hard guard", "if ran_forwards == 0" in pkg.kv_calibrate or "if ran_forwards == 0" in pkg.quantize, ""))
    checks.append(("zero-hook hard guard", 's["n_samples"] == 0' in pkg.kv_calibrate or 's["n_samples"] == 0' in pkg.quantize, ""))
    checks.append(("direct RadixAttention scale write", "self_attn.attn.k_scale" in pkg.kv_calibrate or "self_attn.attn.k_scale" in pkg.quantize, ""))
    checks.append((
        "disk offload not passed as GPTQModel.load kwarg",
        'kwargs["offload_to_disk"]' not in pkg.quantize,
        "GPTQModel 7.0.0 forwards unknown load kwargs into MiniCPMSALAForCausalLM.__init__",
    ))
    checks.append((
        "no legacy self_attn.k_scale tensor write",
        "model.layers.{lid}.self_attn.k_scale" not in pkg.kv_calibrate
        and "model.layers.{lid}.self_attn.k_scale" not in pkg.quantize,
        "",
    ))
    checks.append((
        f"server dtype {expected_dtype}",
        f"--kv-cache-dtype {expected_dtype}" in pkg.prepare_env,
        "",
    ))
    checks.append((
        f"safe-max {expected_safe_max}",
        f'KV_CALIB_SAFE_MAX="${{KV_CALIB_SAFE_MAX:-{expected_safe_max}}}"' in pkg.prepare_model,
        "",
    ))
    kv_stage_marker = (
        "step 3: post-GPTQ fp8 KV scale calibration"
        if postq_mode
        else "step 3: fp8 KV scale calibration"
    )
    checks.append((
        "KV scale calibration runs after GPTQ quantize and overlay",
        _contains_in_order(
            pkg.prepare_model,
            'timeout "${QUANT_TIMEOUT_MIN}m" python3 "${SCRIPT_DIR}/quantize_gptqmodel_w4a16.py"',
            "step 2: applying selective BF16 overlay",
            kv_stage_marker,
        ),
        "needed to prove a GPTQ replay OOM is not caused by the later KV-scale calibration stage",
    ))
    if denseqkv_mode:
        checks.append((
            "DENSEQKV restores dense qkv BF16 before KV scale calibration",
            _contains_in_order(
                pkg.prepare_model,
                "step 2: applying selective BF16 overlay",
                "step 2b: restoring dense qkv to BF16",
                kv_stage_marker,
            ),
            "DENSEQKV must be a precision-source alignment test, not just a filename",
        ))
        checks.append((
            "DENSEQKV dense minicpm4 layer set and qkv-only modules",
            'DENSE_QKV_LAYERS="${DENSE_QKV_LAYERS:-0,9,16,17,22,29,30,31}"'
            in pkg.prepare_model
            and "--modules qkv" in pkg.prepare_model
            and "model.layers.{layer}.self_attn.qkv_proj$" in pkg.prepare_model,
            "",
        ))
    lower_name = path.name.lower()
    if "multi8k300" in lower_name:
        checks.append((
            "GPTQ calibration is 300 x 8K multi-adaptive",
            'NUM_CALIB="${NUM_CALIB:-300}"' in pkg.prepare_model
            and 'MAX_CALIB_LEN="${MAX_CALIB_LEN:-8192}"' in pkg.prepare_model
            and 'CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"' in pkg.prepare_model,
            "required because this package is the strong-calibration diagnostic after 174343 was too fast",
        ))
    if "tail4k300" in lower_name:
        checks.append((
            "GPTQ calibration is 300 x 4K tail",
            'NUM_CALIB="${NUM_CALIB:-300}"' in pkg.prepare_model
            and 'MAX_CALIB_LEN="${MAX_CALIB_LEN:-4096}"' in pkg.prepare_model
            and 'CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-tail}"' in pkg.prepare_model,
            "",
        ))
    if perhead_mode:
        checks.append((
            "per-head scale emission enabled",
            'KV_CALIB_EMIT_PER_HEAD="${KV_CALIB_EMIT_PER_HEAD:-1}"' in pkg.prepare_model
            and (
                "--emit-per-head-scales" in pkg.prepare_model
                or "--post-quant-kv-emit-per-head-scales" in pkg.prepare_model
            ),
            "",
        ))
        checks.append((
            "calibration can write per-head scale tensors",
            (
                "build_scale_tensor_values" in pkg.kv_calibrate
                and "output_tensor_granularity" in pkg.kv_calibrate
                and "k_tensor_granularity" in pkg.kv_calibrate
            )
            or (
                "_build_scale_tensor_values" in pkg.quantize
                and "post_quant_kv_emit_per_head_scales" in pkg.quantize
                and "output_tensor_granularity" in pkg.quantize
                and "k_tensor_granularity" in pkg.quantize
            ),
            "",
        ))
        checks.append((
            "runtime preserves per-head scale tensors",
            "k_scale_per_head" in pkg.kv_cache
            and "make_kv_cache_scale_loader" in pkg.kv_cache
            and "finalize_kv_cache_scales" in pkg.kv_cache,
            "",
        ))
        checks.append((
            "runtime broadcasts per-head KV cache scale",
            "divide_kv_cache_by_scale_" in pkg.memory_pool
            and "Optional[Union[float, torch.Tensor]]" in pkg.memory_pool
            and "num_heads=self.head_num" in pkg.memory_pool,
            "",
        ))
        checks.append((
            "MiniCPM per-head Q/output bridge packaged",
            "_scale_query_for_per_head_fp8_kv" in pkg.minicpm_backend
            and "scale_flat_query_for_per_head_k" in pkg.minicpm_backend
            and "q=q_reshaped" in pkg.minicpm_backend
            and "q=q_reshaped_by_head_group" in pkg.minicpm_backend
            and "scale_grouped_output_for_per_head_v" in pkg.minicpm_backend
            and 'scale_kwargs["k_scale"] = 1.0' in pkg.minicpm_kernels,
            "",
        ))
        checks.append((
            "MiniCPM fallback dequantizes per-head FP8 KV scales",
            "multiply_kv_cache_by_scale_" in pkg.kv_scale_utils
            and "multiply_kv_cache_by_scale_" in pkg.minicpm_backend
            and "k_scale_per_head" in pkg.minicpm_backend
            and "v_scale_per_head" in pkg.minicpm_backend
            and "num_heads=layer.tp_k_head_num" in pkg.minicpm_backend
            and "num_heads=layer.tp_v_head_num" in pkg.minicpm_backend,
            "",
        ))

    rows = []
    for line in pkg.calib_jsonl.splitlines():
        if len(rows) >= num_samples:
            break
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    nonempty_questions = sum(1 for row in rows if isinstance(row.get("question"), str) and row["question"].strip())
    tasks = Counter(str(row.get("task", "<missing>")) for row in rows)
    checks.append((
        f"first {num_samples} calibration rows have question",
        nonempty_questions == min(num_samples, len(rows)) and len(rows) > 0,
        f"rows={len(rows)} nonempty_questions={nonempty_questions} tasks={dict(sorted(tasks.items()))}",
    ))

    ok = True
    print(f"== {path} ==")
    print(f"expected: dtype={expected_dtype} safe_max={expected_safe_max} num_samples={num_samples}")
    for label, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'} {label}" + (f" ({detail})" if detail else ""))
        ok = ok and passed
    print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()

    ok = True
    for path in args.paths:
        try:
            ok = audit(path) and ok
        except Exception as exc:
            print(f"== {path} ==")
            print(f"FAIL audit exception: {exc}")
            print()
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
