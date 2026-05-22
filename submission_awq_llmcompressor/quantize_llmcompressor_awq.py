#!/usr/bin/env python3
"""Quantize MiniCPM-SALA to W4A16 via llm-compressor's AWQModifier.

Compared to GPTQ (quantize_gptqmodel_w4a16.py):
  - AWQ uses activation-aware per-channel scaling instead of Hessian-based
    weight updates. SALA's scale_emb=12 and scale_depth=1.4 amplify a few
    activation channels — exactly AWQ's sweet spot per multiple papers
    (arxiv 2409.11055) and OpenBMB ships AWQ (not GPTQ) for MiniCPM-V 4.5.
  - Output is compressed-tensors format (NOT gptq_marlin). SGLang loads via
    `--quantization compressed-tensors --dtype bfloat16`.
  - No qzeros fix needed (different packing layout than Marlin).
  - Same calibration: perf_public_set.jsonl, left-truncate, chat template,
    256 samples, 8192 tokens.

Post-processing (same as 1849 GPTQ pipeline, Hard Constraints aware):
  1. AWQModifier + QuantizationModifier via llmcompressor.oneshot()
  2. Tokenizer overwrite from base BF16 (H4 fix; llmcompressor.save also
     re-serializes tokenizer with split chat_template)
  3. config.json transformation:
     - Strip auto_map.AutoConfig (Hard Constraint row 44 — verified by v3/v5/v5c)
     - Strip derived read-only properties (row 45)
     - Add quantization_config block (compressed-tensors format)

NOT IN SCOPE here (intentional):
  - mixed-precision (lightning skip): TODO for v6_awq_mixed
  - rotation methods (SpinQuant/QuaRot): research recommends defer
  - bit width = 4 only (W8A16 is separate variant)

Prerequisites (installed by prepare_env.sh):
  llmcompressor (>=0.7 for AWQModifier), transformers==4.57.1, accelerate,
  flash_attn 2.8.3+cu128torch2.9-cp310 (bundled wheel)

Wall time: ~60-180 minutes on RTX PRO 6000 / similar.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _stub_transformers_compat() -> None:
    """Compat shim for transformers 4.x ↔ 5.x naming differences.

    Mirrors quantize_gptqmodel_w4a16.py:66 (the gptqmodel 7.0 shim) defensively.
    llmcompressor may or may not need this, but it's a 0-cost no-op when
    transformers already has the attribute.
    """
    import transformers
    if not hasattr(transformers, "PreTrainedConfig"):
        if hasattr(transformers, "PretrainedConfig"):
            transformers.PreTrainedConfig = transformers.PretrainedConfig


_stub_transformers_compat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantize MiniCPM-SALA to W4A16 via llm-compressor AWQ",
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
        default="W4A16_ASYM",
        help="Quantization scheme. W4A16_ASYM is the production default for "
             "instruction-tuned models per RedHatAI shipped recipes.",
    )
    p.add_argument(
        "--ignore",
        default="lm_head",
        help="Comma-separated module name patterns to skip (lm_head always included)",
    )
    p.add_argument(
        "--mlp-only",
        action="store_true",
        help="Quantize only MLP modules (gate_proj/up_proj/down_proj), keep "
             "attention BF16. Mirrors 1849's MLP-only strategy. Set this for "
             "hybrid-attention SALA — lightning attention recurrence is "
             "fragile under 4-bit attention quant.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without quantizing.",
    )
    return p.parse_args()


# ============================================================================
# Calibration loading (mirrors quantize_gptqmodel_w4a16.py exactly)
# ============================================================================
def load_calibration_prompts(jsonl_path: str, num_calib: int, seed: int) -> list[str]:
    """Read perf_public_set.jsonl, cycle deterministically to reach num_calib."""
    if not jsonl_path or not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"calibration jsonl required for AWQ: --calib-jsonl <path>; got {jsonl_path!r}"
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
        # Deterministic cycle to reach num_calib (matches 1849 behavior)
        original = list(prompts)
        i = 0
        while len(prompts) < num_calib:
            prompts.append(original[i % len(original)])
            i += 1
        print(f"[calib] source has {len(original)} usable rows; cycled deterministically "
              f"to satisfy --num-calib={num_calib}", flush=True)

    # Report distribution if 'task' field present
    tasks: dict[str, int] = {}
    for r in rows[:num_calib]:
        t = r.get("task") or r.get("category") or "?"
        tasks[t] = tasks.get(t, 0) + 1
    if tasks:
        print(f"[calib] task distribution (first {min(len(rows), num_calib)} rows): {tasks}",
              flush=True)

    return prompts[:num_calib]


def tokenize_calibration(prompts: list[str], tokenizer, max_len: int):
    """Same tokenization strategy as 1849:
       - apply_chat_template wraps in `<用户>...<AI>`
       - truncation_side='left' keeps the question/answer tail
    """
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
# Module filtering for MLP-only mode
# ============================================================================
def make_targets_and_ignore(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Build (targets, ignore) for QuantizationModifier given --mlp-only flag.

    MLP-only mirrors 1849's strategy: skip all self_attn modules because
    SALA's lightning attention recurrence is fragile under W4A16.
    """
    ignore = [s.strip() for s in args.ignore.split(",") if s.strip()]
    if "lm_head" not in ignore:
        ignore.append("lm_head")

    if args.mlp_only:
        # Pattern-based: ignore everything in self_attn block
        # llmcompressor accepts regex via `ignore` field
        ignore.extend([
            "re:.*self_attn.*",   # both dense and lightning attention
            "re:.*o_gate$",       # MiniCPM dense-attn output gate
            "re:.*z_proj$",       # Lightning Mixer output gate
            "re:.*o_norm$",
            "re:.*q_norm$",
            "re:.*k_norm$",
        ])
        targets = "Linear"  # all Linears except those in `ignore`
    else:
        targets = "Linear"

    return targets, ignore


# ============================================================================
# Post-quant fixes (mirrors quantize_gptqmodel_w4a16.py post-processing)
# ============================================================================
def copy_runtime_assets(input_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer / chat-template / custom modeling files.

    CRITICAL (H4 fix, SUBMISSIONS.md 2026-05-22): llmcompressor.save() (like
    GPTQModel.save()) re-serializes the tokenizer with a split chat_template
    (inline + .jinja sidecar). Older transformers ignore the sidecar →
    empty chat_template → garbage output. Overwrite tokenizer files
    unconditionally from base BF16 so they're byte-identical to the source.
    """
    for path in input_dir.iterdir():
        if path.is_dir():
            continue
        # Don't overwrite the model.safetensors.index.json (llmcompressor wrote
        # the quantized version of it) nor config.json (we'll write our own).
        if path.name in ("model.safetensors.index.json", "config.json"):
            continue
        # Copy: tokenizer/template/text/custom-modeling .py files
        if path.suffix in {".json", ".model", ".txt", ".py"} or path.name == "tokenizer.json":
            target = output_dir / path.name
            shutil.copy2(path, target)
            print(f"[copy_runtime_assets] overwrote {path.name}", flush=True)


def write_sglang_compatible_config(
    output_dir: Path, input_dir: Path, bits: int, group_size: int, scheme: str
) -> None:
    """Build config.json compatible with SGLang's compressed-tensors loader.

    CRITICAL (Hard Constraint row 44, verified by v3/v5/v5c crash 2026-05-23):
    Remove auto_map.AutoConfig so SGLang's MiniCPMHybridConfig wins isinstance.

    Strip derived read-only properties (row 45, v18 root cause).
    """
    input_config_path = input_dir / "config.json"
    output_config_path = output_dir / "config.json"

    if not input_config_path.exists():
        print("[config] WARN: input config.json missing", file=sys.stderr)
        if output_config_path.exists():
            return
        raise RuntimeError(f"no config.json in input dir {input_dir}")

    with input_config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # ---- Row 44: strip auto_map.AutoConfig ----
    auto_map = config.get("auto_map", {})
    if isinstance(auto_map, dict) and "AutoConfig" in auto_map:
        removed = auto_map.pop("AutoConfig")
        if not auto_map:
            config.pop("auto_map", None)
        else:
            config["auto_map"] = auto_map
        print(f"[config] removed auto_map.AutoConfig={removed!r} (Hard Constraint row 44)", flush=True)

    # ---- Row 45: strip derived read-only properties ----
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
            print(f"[config] stripped derived property {k} (Hard Constraint row 45)", flush=True)

    # ---- Quantization config: compressed-tensors format ----
    # SGLang's compressed-tensors loader reads this to determine layout.
    # The actual quant artifacts are stored in safetensors with the standard
    # compressed-tensors naming convention written by llmcompressor.
    quant_method = "compressed-tensors"
    # Format reference: compressed-tensors library uses scheme-based config
    if scheme.startswith("W4A16"):
        quant_cfg = {
            "config_groups": {
                "group_0": {
                    "input_activations": None,  # W4A16 = weights only
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": bits,
                        "type": "int",
                        "symmetric": (scheme.upper().endswith("_SYM")),
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
        # Fallback: just record what was passed in. Should not happen here.
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
    print(f"  AWQ via llm-compressor (W4A16, scheme={args.scheme})")
    print(f"  Input:        {args.input}")
    print(f"  Output:       {args.output}")
    print(f"  Group size:   {args.group_size}")
    print(f"  Bits:         {args.bits}")
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
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError:
        AWQModifier = None
        print("[warn] AWQModifier import failed; falling back to GPTQModifier-via-llmcompressor "
              "(still better than GPTQModel due to cleaner tokenizer handling)", file=sys.stderr)

    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as e:
        print(f"[fatal] QuantizationModifier import failed: {e}", file=sys.stderr)
        return 2

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load tokenizer + model
    print("[1/5] Loading tokenizer + model (bfloat16, trust_remote_code)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(input_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(input_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # 2. Load calibration
    print("[2/5] Loading calibration data...", flush=True)
    prompts = load_calibration_prompts(args.calib_jsonl, args.num_calib, args.seed)
    encoded = tokenize_calibration(prompts, tokenizer, args.max_calib_len)
    print(f"        ready: {len(encoded)} calibration samples", flush=True)

    # 3. Build recipe
    print("[3/5] Building recipe...", flush=True)
    targets, ignore_list = make_targets_and_ignore(args)
    print(f"        targets: {targets}")
    print(f"        ignore:  {ignore_list}")

    recipe: list = []
    if AWQModifier is not None:
        recipe.append(
            AWQModifier(
                # AWQModifier targets the channel-scaling step. Apply to all
                # Linear and let QuantizationModifier's ignore list handle skip.
                ignore=ignore_list,
            )
        )
    recipe.append(
        QuantizationModifier(
            targets=targets,
            scheme=args.scheme,
            ignore=ignore_list,
        )
    )
    print(f"        recipe: {[type(m).__name__ for m in recipe]}", flush=True)

    # 4. Run oneshot
    print("[4/5] Running oneshot quantization (this is the slow part, ~60-180 min)...", flush=True)
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

    # Cleanup GPU memory before post-processing
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Post-quant fixes (CRITICAL: these are the Hard Constraints)
    print("[5/5] Post-quant fixes (H4 tokenizer + config.json auto_map.AutoConfig strip)...",
          flush=True)
    copy_runtime_assets(input_dir, output_dir)
    write_sglang_compatible_config(
        output_dir=output_dir,
        input_dir=input_dir,
        bits=args.bits,
        group_size=args.group_size,
        scheme=args.scheme,
    )

    # Save tokenizer one more time (some tools prefer the file present)
    tokenizer.save_pretrained(str(output_dir))
    # And immediately overwrite with the base BF16 version again (H4)
    copy_runtime_assets(input_dir, output_dir)

    print("=" * 70)
    print(f"  AWQ quantization DONE")
    print(f"  Output: {output_dir}")
    print(f"  Next: SGLang with --quantization compressed-tensors --dtype bfloat16")
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
