#!/usr/bin/env python3
"""Quantize MiniCPM-SALA to W4A16 via llm-compressor's GPTQModifier.

Mirrors the OFFICIAL `quantize_to_w4a16.py` recipe (which W4A16_README.md
explicitly recommends) but plugs in our hardened infrastructure:
  - perf_public_set.jsonl calibration with chat-template + left-truncation
    (NOT wikitext-2-raw-v1 default, which doesn't match SOAR eval distribution)
  - H4 tokenizer overwrite (llmcompressor.save re-serializes chat_template)
  - auto_map.AutoConfig strip (Hard Constraint row 44)
  - compressed-tensors quantization_config block

Compared to quantize_llmcompressor_awq.py (sibling AWQ variant):
  - This uses GPTQModifier alone (recipe = [GPTQModifier(...)])
  - AWQ uses recipe = [AWQModifier(...), QuantizationModifier(...)]
  - GPTQModifier is the official W4A16_README recommendation
  - GPTQ may handle scale_emb=12 outliers worse than AWQ's channel scaling;
    P3 ablation will tell us which is better for SALA

Recipe matches `quantize_to_w4a16.py` (the repo's official script):
  GPTQModifier(targets="Linear", scheme="W4A16",
               ignore=ignore_list, dampening_frac=0.01)

Output is compressed-tensors format (same as AWQ). SGLang loads with:
  --quantization compressed-tensors --dtype bfloat16

NOT IN SCOPE here (intentional):
  - mixed-precision (lightning skip): handled separately by P6
    (tools/apply_lightning_skip_overlay.py)
  - rotation methods (SpinQuant/QuaRot): research recommends defer
  - bit width = 4 only

Wall time: ~60-90 minutes on RTX PRO 6000 / similar.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path


def _stub_transformers_compat() -> None:
    """Compat shim for transformers 4.x ↔ 5.x naming differences.

    Mirrors quantize_llmcompressor_awq.py for symmetry. No-op if transformers
    already has PreTrainedConfig.
    """
    import transformers
    if not hasattr(transformers, "PreTrainedConfig"):
        if hasattr(transformers, "PretrainedConfig"):
            transformers.PreTrainedConfig = transformers.PretrainedConfig


_stub_transformers_compat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantize MiniCPM-SALA to W4A16 via llm-compressor GPTQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Original BF16 model directory")
    p.add_argument("--output", required=True, help="Quantized output directory")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument(
        "--calib-jsonl",
        default="",
        help="Calibration JSONL (perf_public_set.jsonl). Each row has a "
             "`question` field; `gold` is appended if present.",
    )
    p.add_argument("--num-calib", type=int, default=256)
    p.add_argument(
        "--max-calib-len",
        type=int,
        default=8192,
        help="Tail of each prompt is kept (truncation_side='left').",
    )
    p.add_argument(
        "--scheme",
        default="W4A16",
        help="Quantization scheme. W4A16 (symmetric) matches the official "
             "quantize_to_w4a16.py recipe; W4A16_ASYM is also accepted by "
             "GPTQModifier if you want to compare against AWQ_ASYM.",
    )
    p.add_argument(
        "--ignore",
        default="lm_head",
        help="Comma-separated module name patterns to skip (lm_head always included)",
    )
    p.add_argument(
        "--mlp-only",
        action="store_true",
        help="Quantize only MLP modules, keep attention BF16. Mirrors 1849's "
             "hybrid-attention safety. Recommended for SALA's lightning attn.",
    )
    p.add_argument(
        "--dampening-frac",
        type=float,
        default=0.01,
        help="GPTQ Hessian dampening (matches official quantize_to_w4a16.py). "
             "Raise to 0.1 if numerical issues; 0.01 is research default.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without quantizing.",
    )
    return p.parse_args()


# ============================================================================
# Calibration loading — IDENTICAL to AWQ variant for clean A/B comparison
# ============================================================================
def load_calibration_prompts(jsonl_path: str, num_calib: int, seed: int) -> list[str]:
    """Read perf_public_set.jsonl, cycle deterministically to reach num_calib."""
    if not jsonl_path or not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"calibration jsonl required: --calib-jsonl <path>; got {jsonl_path!r}"
        )

    rows: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            rows.append(row)

    if not rows:
        raise RuntimeError(f"no usable rows in {jsonl_path}")

    prompts: list[str] = []
    for r in rows:
        q = r.get("question") or r.get("prompt") or ""
        g = r.get("gold") or r.get("answer") or ""
        prompts.append(q + ("\n" + g if g else ""))

    print(f"[calib] loaded {len(prompts)} prompts from {jsonl_path}", flush=True)

    if len(prompts) < num_calib:
        original = list(prompts)
        i = 0
        while len(prompts) < num_calib:
            prompts.append(original[i % len(original)])
            i += 1
        print(f"[calib] source has {len(original)} usable rows; cycled deterministically "
              f"to satisfy --num-calib={num_calib}", flush=True)

    tasks: dict[str, int] = {}
    for r in rows[:num_calib]:
        t = r.get("task") or r.get("category") or "?"
        tasks[t] = tasks.get(t, 0) + 1
    if tasks:
        print(f"[calib] task distribution (first {min(len(rows), num_calib)} rows): {tasks}",
              flush=True)

    return prompts[:num_calib]


def tokenize_calibration(prompts: list[str], tokenizer, max_len: int):
    """Same tokenization as AWQ variant: chat template + left-truncate."""
    tokenizer.truncation_side = "left"

    encoded: list[dict] = []
    for prompt in prompts:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            text = prompt

        ids = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )
        encoded.append(ids)

    return encoded


# ============================================================================
# Module filtering for MLP-only mode (mirrors AWQ variant)
# ============================================================================
def make_targets_and_ignore(args: argparse.Namespace) -> tuple[str, list[str]]:
    """Build (targets, ignore) for GPTQModifier given --mlp-only flag."""
    ignore = [s.strip() for s in args.ignore.split(",") if s.strip()]
    if "lm_head" not in ignore:
        ignore.append("lm_head")

    if args.mlp_only:
        ignore.extend([
            "re:.*self_attn.*",
            "re:.*o_gate$",
            "re:.*z_proj$",
            "re:.*o_norm$",
            "re:.*q_norm$",
            "re:.*k_norm$",
        ])
    return "Linear", ignore


# ============================================================================
# Post-quant fixes — IDENTICAL to AWQ variant
# ============================================================================
def copy_runtime_assets(input_dir: Path, output_dir: Path) -> None:
    """H4 fix: overwrite tokenizer + chat-template + custom modeling .py from base BF16."""
    for path in input_dir.iterdir():
        if path.is_dir():
            continue
        if path.name in ("model.safetensors.index.json", "config.json"):
            continue
        if path.suffix in {".json", ".model", ".txt", ".py"} or path.name == "tokenizer.json":
            target = output_dir / path.name
            shutil.copy2(path, target)
            print(f"[copy_runtime_assets] overwrote {path.name}", flush=True)


def write_sglang_compatible_config(
    output_dir: Path, input_dir: Path, bits: int, group_size: int, scheme: str
) -> None:
    """Hard Constraint row 44 + row 45 + compressed-tensors quant config."""
    input_config_path = input_dir / "config.json"
    output_config_path = output_dir / "config.json"

    if not input_config_path.exists():
        print("[config] WARN: input config.json missing", file=sys.stderr)
        if output_config_path.exists():
            return
        raise RuntimeError(f"no config.json in input dir {input_dir}")

    with input_config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # Row 44: strip auto_map.AutoConfig
    auto_map = config.get("auto_map", {})
    if isinstance(auto_map, dict) and "AutoConfig" in auto_map:
        removed = auto_map.pop("AutoConfig")
        if not auto_map:
            config.pop("auto_map", None)
        else:
            config["auto_map"] = auto_map
        print(f"[config] removed auto_map.AutoConfig={removed!r} (row 44)", flush=True)

    # Row 45: strip derived read-only properties
    for k in (
        "has_sparse_attention",
        "has_lightning_layers",
        "full_attention_layer_ids",
        "sparse_layer_ids",
        "lightning_layer_ids",
        "mamba2_cache_params",
    ):
        if k in config:
            config.pop(k, None)
            print(f"[config] stripped derived property {k} (row 45)", flush=True)

    # compressed-tensors quant config
    quant_method = "compressed-tensors"
    if scheme.upper().startswith("W4A16"):
        symmetric = not scheme.upper().endswith("_ASYM")
        quant_cfg = {
            "config_groups": {
                "group_0": {
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": bits,
                        "type": "int",
                        "symmetric": symmetric,
                        "strategy": "group",
                        "group_size": group_size,
                    },
                }
            },
            "format": "pack-quantized",
            "global_compression_ratio": None,
            "ignore": ["lm_head"],
            "kv_cache_scheme": None,
            "quant_method": quant_method,
            "quantization_status": "compressed",
        }
    else:
        quant_cfg = {"quant_method": quant_method, "scheme": scheme}

    config["quantization_config"] = quant_cfg

    with output_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[config] wrote {output_config_path}", flush=True)


# ============================================================================
# Main entry
# ============================================================================
def main() -> int:
    args = parse_args()

    print("=" * 70)
    print(f"  GPTQ via llm-compressor (scheme={args.scheme})")
    print(f"  Input:        {args.input}")
    print(f"  Output:       {args.output}")
    print(f"  Group size:   {args.group_size}")
    print(f"  Bits:         {args.bits}")
    print(f"  Dampening:    {args.dampening_frac}")
    print(f"  MLP-only:     {args.mlp_only}")
    print(f"  Calib:        {args.calib_jsonl}")
    print(f"  num_calib:    {args.num_calib}, max_len: {args.max_calib_len}")
    print("=" * 70, flush=True)

    if args.dry_run:
        targets, ignore = make_targets_and_ignore(args)
        print(f"\n[dry-run] targets={targets!r}")
        print(f"[dry-run] ignore={ignore!r}")
        return 0

    # Late imports
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor import oneshot
    except ImportError as e:
        print(f"[fatal] llmcompressor not installed: {e}", file=sys.stderr)
        return 2

    try:
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError as e:
        print(f"[fatal] GPTQModifier import failed: {e}", file=sys.stderr)
        return 2

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("[1/5] Loading tokenizer + model (bfloat16, trust_remote_code)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(input_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(input_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # 2. Calib
    print("[2/5] Loading calibration data...", flush=True)
    prompts = load_calibration_prompts(args.calib_jsonl, args.num_calib, args.seed)
    encoded = tokenize_calibration(prompts, tokenizer, args.max_calib_len)
    print(f"        ready: {len(encoded)} calibration samples", flush=True)

    # 3. Recipe — pure GPTQModifier (matches official quantize_to_w4a16.py)
    print("[3/5] Building recipe (pure GPTQModifier, official W4A16 recipe)...", flush=True)
    targets, ignore_list = make_targets_and_ignore(args)
    print(f"        targets: {targets}")
    print(f"        ignore:  {ignore_list}")

    recipe = GPTQModifier(
        targets=targets,
        scheme=args.scheme,
        ignore=ignore_list,
        dampening_frac=args.dampening_frac,
    )
    print(f"        recipe: GPTQModifier(scheme={args.scheme}, dampening_frac={args.dampening_frac})",
          flush=True)

    # 4. Run oneshot
    print("[4/5] Running oneshot GPTQ quantization (~60-90 min)...", flush=True)
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=encoded,
        recipe=recipe,
        max_seq_length=args.max_calib_len,
        num_calibration_samples=len(encoded),
        output_dir=str(output_dir),
        save_compressed=True,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Post-quant fixes
    print("[5/5] Post-quant fixes (H4 tokenizer + config.json auto_map strip)...", flush=True)
    copy_runtime_assets(input_dir, output_dir)
    write_sglang_compatible_config(
        output_dir=output_dir,
        input_dir=input_dir,
        bits=args.bits,
        group_size=args.group_size,
        scheme=args.scheme,
    )

    # Save tokenizer once + overwrite with base (H4 belt-and-suspenders)
    tokenizer.save_pretrained(str(output_dir))
    copy_runtime_assets(input_dir, output_dir)

    print("=" * 70)
    print(f"  GPTQ quantization DONE")
    print(f"  Output: {output_dir}")
    print(f"  Next: SGLang with --quantization compressed-tensors --dtype bfloat16")
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
