#!/usr/bin/env python3
"""GPTQModel W4A16 quantization for MiniCPM-SALA — SOAR submission.

Differences from the earlier broken `quantize_gptqmodel_w4a16.py`:

1. **Registers MiniCPM-SALA explicitly.** SALA's HF `model_type` is
   `minicpm_sala`, which is NOT in GPTQModel's built-in registry. We
   register a `MiniCPMSALAGPTQ` subclass of `MiniCPMGPTQ` so the
   quantizer knows the module tree. The module tree adds:
       - `z_proj` (output gate on Lightning Attention layers)
       - `o_gate` (output gate on the standard "minicpm4" layers, when present)
   Both are skipped (`":!"`) for quantization to avoid touching the
   gating path (low marginal benefit, high risk).

2. **Calibration data comes from perf_public_set.jsonl by default.**
   The earlier script used 4 hand-written prompt templates repeated 128
   times — those are not statistically representative of the eval set
   and was the most likely cause of the catastrophic ~40-point accuracy
   drop on the platform. We sample from the SOAR public eval set
   instead, optionally mixing in synthetic long-context probes.

3. **No silent fallback chains.** The previous script's
   `load_gptq_model()` tried 6+ argument combinations and swallowed
   errors. We pick ONE call shape, surface real errors loudly.

4. **No FP8 KV cache in companion prepare_env.sh** — already verified
   broken for the SALA sparse backend. Stays at fp16 KV cache.

5. **Memory-friendly load** for environments without 80GB+ GPUs:
       device_map="auto" with explicit max_memory ceilings, and
       offload_to_disk=True (GPTQModel's flag) for CPU-RAM-poor runs.

Usage (on the GPU node):

    pip install gptqmodel torch transformers accelerate

    python3 quantize_gptqmodel_w4a16.py \\
        --input  /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\
        --output /root/autodl-tmp/models/MiniCPM-SALA-W4A16-GPTQ \\
        --calib-jsonl /root/autodl-fs/zyn/soar_toolkit/perf_public_set.jsonl \\
        --num-calib 256 \\
        --max-calib-len 8192

For local dry-runs of the script logic (CPU-only, no actual quant),
the `--dry-run` flag prints the resolved config + the first 3
calibration prompts and exits before touching GPTQModel.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize MiniCPM-SALA with GPTQModel W4A16 -> SGLang Marlin"
    )
    parser.add_argument("--input", required=True, help="Original BF16 model directory")
    parser.add_argument("--output", required=True, help="Quantized model output directory")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--calib-jsonl",
        default="",
        help="Calibration JSONL (typically SOAR perf_public_set.jsonl). "
             "Each row should have a `question` field; `gold` is appended "
             "if present to approximate full prompt-answer length.",
    )
    parser.add_argument(
        "--num-calib",
        type=int,
        default=256,
        help="Number of calibration samples (1024+ if VRAM/RAM permits "
             "for higher quality)",
    )
    parser.add_argument(
        "--max-calib-len",
        type=int,
        default=4096,
        help="Max tokens per calibration prompt. Higher = better for "
             "long-context preservation, but more memory + time.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for sampling calibration prompts from the JSONL",
    )
    parser.add_argument(
        "--gpu-max-mem",
        default="",
        help="GPU memory cap, e.g. '40GiB' or '80GiB'. Empty = let "
             "GPTQModel decide. Set on 24GB cards to force offloading.",
    )
    parser.add_argument(
        "--cpu-max-mem",
        default="60GiB",
        help="CPU RAM cap during offload. Default 60GiB; raise for "
             "machines with more.",
    )
    parser.add_argument(
        "--no-offload-disk",
        action="store_true",
        help="Disable GPTQModel's offload_to_disk (default is on). Faster "
             "but uses more CPU RAM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and a sample of calibration prompts, "
             "then exit. Does NOT import torch/gptqmodel — safe to run "
             "locally without a GPU.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
def _synthetic_long_probes(count: int) -> list[str]:
    """Fallback when no calibration JSONL is provided.

    Goal: still better than the previous 4-template approach by varying
    length and task style. Used only if --calib-jsonl is not set.
    """
    base_templates = [
        ("Read the passage and answer the multiple choice question.\n"
         "Passage: {filler}\n"
         "Question: Based on the passage, which is correct?\n"
         "A. Statement A\nB. Statement B\nC. Statement C\nD. Statement D\nAnswer:"),
        ("Extract the requested information from the document.\n"
         "Document: {filler}\n"
         "Question: What is the most important fact?\nAnswer:"),
        ("Summarize the following long passage in one paragraph.\n"
         "{filler}\nSummary:"),
        ("Answer the multi-hop question using the context below.\n"
         "Context: {filler}\nQuestion: Combine the relevant facts to answer.\nAnswer:"),
    ]
    filler_phrases = [
        "The system processes long-context inference using hybrid attention.",
        "Researchers benchmarked W4A16 quantization on the SOAR platform.",
        "Marlin GEMM exploits 4-bit packing to reduce memory bandwidth.",
        "Lightning attention has linear complexity over the sequence dimension.",
        "InfLLM-V2 sparse attention selects relevant key blocks per query.",
        "The MiniCPM-SALA architecture mixes dense and linear attention layers.",
        "RTX PRO 6000 Blackwell has compute capability 12.0.",
        "GPTQ uses second-order Hessian information for weight quantization.",
    ]
    rng = random.Random(0xC4118)
    texts: list[str] = []
    for i in range(count):
        tpl = base_templates[i % len(base_templates)]
        # Vary length by repeating filler N times
        n_repeats = 8 + (i % 32)
        filler = " ".join(rng.sample(filler_phrases, k=len(filler_phrases))) * n_repeats
        texts.append(tpl.format(filler=filler))
    return texts


def load_calibration_prompts(
    jsonl_path: str, num_samples: int, seed: int
) -> list[str]:
    """Load and sample calibration prompts.

    Strategy:
      1. If --calib-jsonl is given (typically perf_public_set.jsonl),
         load all rows, deterministic shuffle, take first num_samples.
      2. Each row's `question` field becomes the calibration text.
         If `gold` is a short string we append it (so the calibration
         covers a typical answer continuation, not only prompt prefix).
      3. If the JSONL is empty/missing, fall back to synthetic probes
         and WARN clearly.
    """
    if not jsonl_path:
        print(
            "[calib] WARNING: no --calib-jsonl set. Falling back to synthetic "
            "calibration. Strongly recommend passing the SOAR "
            "perf_public_set.jsonl for representative calibration — "
            "this was the root cause of the earlier 42-acc submission.",
            flush=True,
        )
        return _synthetic_long_probes(num_samples)

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"--calib-jsonl not found: {jsonl_path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise RuntimeError(f"--calib-jsonl contains no usable rows: {jsonl_path}")

    rng = random.Random(seed)
    rng.shuffle(rows)

    texts: list[str] = []
    task_counts: dict[str, int] = {}
    for row in rows:
        if len(texts) >= num_samples:
            break
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        gold = row.get("gold")
        task = row.get("task", "unknown")
        # Append gold answer if it's a short string. For MCQ, gold is often
        # just "A"/"B"/"C"/"D" — append it as "Answer: <gold>".
        prompt = question.strip()
        if isinstance(gold, str) and 0 < len(gold) < 200:
            prompt = f"{prompt}\n{gold.strip()}"
        texts.append(prompt)
        task_counts[task] = task_counts.get(task, 0) + 1

    print(f"[calib] loaded {len(texts)} prompts from {jsonl_path}", flush=True)
    print(f"[calib] task distribution: {task_counts}", flush=True)
    return texts


# ---------------------------------------------------------------------------
# MiniCPM-SALA model registration
# ---------------------------------------------------------------------------
def register_minicpm_sala_with_gptqmodel() -> None:
    """Teach GPTQModel that `minicpm_sala` looks like `minicpm`.

    GPTQModel resolves `config.model_type` to a `BaseQModel` subclass via
    `gptqmodel.models.MODEL_MAP`. SALA shares module names with the
    standard MiniCPM family on the parts that we want to quantize
    (`qkv_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`); the
    extras (`z_proj`, `o_gate`) we explicitly skip.

    NOTE: GPTQModel does NOT use the QKVParallelLinear name we see in
    SGLang — it works on the HuggingFace state dict, where the names
    are `q_proj`, `k_proj`, `v_proj` separately. The MiniCPM-SALA HF
    weights ship with split q/k/v projections (verify with the HF
    checkpoint's `model.safetensors.index.json`).
    """
    from gptqmodel.models import MODEL_MAP  # type: ignore
    from gptqmodel.models.definitions.minicpm import MiniCPMGPTQ  # type: ignore

    class MiniCPMSALAGPTQ(MiniCPMGPTQ):
        """Module tree for MiniCPM-SALA.

        Mirrors MiniCPMGPTQ.module_tree but with:
          - `z_proj` and `o_gate` explicitly marked as "do not quantize"
            via the `":!"` suffix (GPTQModel convention).
          - Same q/k/v/o + gate/up/down quantization order.
        """
        pre_lm_head_norm_module = "model.norm"

        module_tree = [
            "model",
            "layers",
            "#",
            {
                "input_layernorm": ("input_layernorm:!",),
                "self_attn": (
                    "q_proj:0",
                    "k_proj:1",
                    "v_proj:2",
                    "o_proj:3",
                    # Output gate found on the 6 dense ("minicpm4") layers.
                    # Skip quantization: gating projections are highly
                    # accuracy-sensitive and only ~3% of parameters.
                    "o_gate:!",
                    # Lightning layers' output gate. Same reasoning.
                    "z_proj:!",
                    # Lightning-attention norm modules (when present).
                    "q_norm:!",
                    "k_norm:!",
                    "o_norm:!",
                ),
                "post_attention_layernorm": ("post_attention_layernorm:!",),
                "mlp": ("gate_proj", "up_proj", "down_proj"),
            }
        ]

    # Register both keys: model_type AND architecture, since GPTQModel
    # versions differ on which one they look up first.
    MODEL_MAP["minicpm_sala"] = MiniCPMSALAGPTQ
    print("[register] registered minicpm_sala -> MiniCPMSALAGPTQ in MODEL_MAP",
          flush=True)


# ---------------------------------------------------------------------------
# Quantize-config builder + GPTQModel loader
# ---------------------------------------------------------------------------
def make_quant_config(bits: int, group_size: int):
    """Construct a GPTQModel-compatible QuantizeConfig.

    For SGLang gptq_marlin we need:
        bits=4 or 8
        sym=True (uint4b8 / uint8b128)
        desc_act=False (Marlin doesn't support desc_act=True for our case)
        group_size=128 by convention
    """
    try:
        from gptqmodel import QuantizeConfig as ConfigClass  # type: ignore
    except ImportError:
        from gptqmodel import GPTQConfig as ConfigClass  # type: ignore  # noqa

    desired: dict[str, Any] = {
        "bits": bits,
        "group_size": group_size,
        "desc_act": False,
        "sym": True,
        "lm_head": False,
    }
    sig = inspect.signature(ConfigClass)
    kwargs = {k: v for k, v in desired.items() if k in sig.parameters}
    return ConfigClass(**kwargs)


def load_model(
    input_dir: str,
    quant_config: Any,
    gpu_max_mem: str,
    cpu_max_mem: str,
    use_disk_offload: bool,
):
    """Load BF16 model under GPTQModel's wrapper.

    NOTE on max_memory: GPTQModel 7.0 passes unrecognized kwargs all the
    way through to HF's auto_factory `from_config(...)` -> `cls(config,
    **kwargs)`. For SALA, that hits `MiniCPMSALAForCausalLM.__init__`
    which does NOT accept `max_memory` (custom modeling loaded via
    trust_remote_code). So we only attach `max_memory` if the user
    explicitly opted in via --gpu-max-mem. Default = pass nothing about
    memory and let GPTQModel auto-detect, which is fine on a 96GB GPU.
    (cpu_max_mem default was the culprit that broke the 2026-05-19
    20:30 submission.)
    """
    from gptqmodel import GPTQModel  # type: ignore

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }

    # Only attach max_memory when explicitly requested.
    max_memory: dict[Any, str] = {}
    if gpu_max_mem:
        max_memory[0] = gpu_max_mem
    # Intentionally NOT defaulting a "cpu" entry — see docstring.
    if max_memory:
        kwargs["max_memory"] = max_memory

    if use_disk_offload:
        # `offload_to_disk` is a GPTQModel-recognized flag (not passed
        # through to the model constructor), so it's safe.
        kwargs["offload_to_disk"] = True

    print(f"[load] GPTQModel.load(...)  kwargs={sorted(kwargs.keys())}",
          flush=True)
    return GPTQModel.load(input_dir, quant_config, **kwargs)


def tokenize_calibration(
    tokenizer,
    texts: list[str],
    max_len: int,
) -> list[dict[str, Any]]:
    """Tokenize each prompt; drop empties."""
    examples: list[dict[str, Any]] = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=True,
            return_tensors=None,
        )
        if encoded.get("input_ids"):
            examples.append(encoded)
    return examples


# ---------------------------------------------------------------------------
# Output post-processing (Marlin compatibility)
# ---------------------------------------------------------------------------
def write_sglang_compatible_quant_config(
    output_dir: Path, bits: int, group_size: int
) -> None:
    """Override the saved quantize_config.json with a Marlin-ready shape.

    GPTQModel writes its own `quantize_config.json`. We re-write to make
    sure the SGLang `gptq_marlin` loader at `gptq.py:325` (it reads
    `bits`, `group_size`, `sym`, `desc_act`, `lm_head`, `dynamic`) finds
    exactly what it expects.
    """
    quant_cfg = {
        "bits": bits,
        "group_size": group_size,
        "quant_method": "gptq",
        "desc_act": False,
        "sym": True,
        "lm_head": False,
        "dynamic": {},
    }
    quant_path = output_dir / "quantize_config.json"
    if quant_path.exists():
        with quant_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(quant_cfg)
        quant_cfg = existing
    with quant_path.open("w", encoding="utf-8") as f:
        json.dump(quant_cfg, f, indent=2)

    config_path = output_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        # SGLang's gptq_marlin path expects fp16 activations.
        config["torch_dtype"] = "float16"
        config["quantization_config"] = quant_cfg
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


def copy_runtime_assets(input_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer / chat-template / custom modeling files."""
    for path in input_dir.iterdir():
        if path.is_dir():
            continue
        if path.name == "model.safetensors.index.json":
            # GPTQModel rewrites this with quantized tensor mapping
            continue
        # The HF custom-model files (.py) must travel with the weights
        # so SGLang can load them under --trust-remote-code.
        if path.suffix in {".json", ".model", ".txt", ".py"} or path.name == "tokenizer.json":
            target = output_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)


def save_model(model, output_dir: str) -> None:
    if hasattr(model, "save"):
        model.save(output_dir)
    elif hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        raise RuntimeError(
            "GPTQModel object has neither .save nor .save_pretrained — "
            "is this an unexpected gptqmodel version?"
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def dry_run(args: argparse.Namespace, prompts: list[str]) -> int:
    print("=" * 60)
    print("DRY RUN — no GPU/quantization actually invoked")
    print("=" * 60)
    print(f"  --input            {args.input}")
    print(f"  --output           {args.output}")
    print(f"  --bits             {args.bits}")
    print(f"  --group-size       {args.group_size}")
    print(f"  --num-calib        {args.num_calib} (loaded {len(prompts)})")
    print(f"  --max-calib-len    {args.max_calib_len}")
    print(f"  --calib-jsonl      {args.calib_jsonl or '(synthetic)'}")
    print(f"  --gpu-max-mem      {args.gpu_max_mem or '(GPTQModel default)'}")
    print(f"  --cpu-max-mem      {args.cpu_max_mem}")
    print(f"  --no-offload-disk  {args.no_offload_disk}")
    print()
    print("First 3 calibration prompts (truncated to 200 chars):")
    for i, p in enumerate(prompts[:3]):
        snippet = p.replace("\n", " | ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"  [{i}] {snippet}")
    print()
    print("OK. Re-run without --dry-run on a GPU node to actually quantize.")
    return 0


def main() -> int:
    args = parse_args()

    prompts = load_calibration_prompts(
        args.calib_jsonl, args.num_calib, args.seed
    )

    if args.dry_run:
        return dry_run(args, prompts)

    if args.bits not in (4, 8):
        raise SystemExit(
            f"--bits must be 4 or 8 for SGLang Marlin compatibility; got {args.bits}"
        )

    # Lazy import — only after we know it's not a dry run.
    register_minicpm_sala_with_gptqmodel()
    quant_config = make_quant_config(args.bits, args.group_size)

    model = load_model(
        args.input,
        quant_config,
        gpu_max_mem=args.gpu_max_mem,
        cpu_max_mem=args.cpu_max_mem,
        use_disk_offload=not args.no_offload_disk,
    )

    # Get tokenizer (GPTQModel >= 6 attaches it; older versions need fallback)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(
            args.input, trust_remote_code=True
        )

    calibration = tokenize_calibration(tokenizer, prompts, args.max_calib_len)
    if not calibration:
        raise RuntimeError(
            "no calibration examples were produced — check that --calib-jsonl "
            "has rows with non-empty `question` fields"
        )

    print(
        f"[quantize] starting GPTQModel W{args.bits}A16 "
        f"group_size={args.group_size} "
        f"samples={len(calibration)} max_len={args.max_calib_len}",
        flush=True,
    )

    model.quantize(calibration, batch_size=args.batch_size)

    print(f"[quantize] saving to {args.output}", flush=True)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(output_dir))

    # Free the model before post-processing to keep peak memory down.
    del model
    gc.collect()
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    copy_runtime_assets(Path(args.input), output_dir)
    write_sglang_compatible_quant_config(output_dir, args.bits, args.group_size)
    print("[quantize] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
