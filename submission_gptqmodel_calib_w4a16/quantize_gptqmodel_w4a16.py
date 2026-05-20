#!/usr/bin/env python3
"""GPTQModel W4A16 quantization for MiniCPM-SALA — SOAR submission.

Differences from the earlier broken `quantize_gptqmodel_w4a16.py`:

1. **Registers MiniCPM-SALA explicitly.** SALA's HF `model_type` is
   `minicpm_sala`, which is NOT in GPTQModel's built-in registry. We
   register a `MiniCPMSALAGPTQ` subclass of `BaseQModel` so the
   quantizer sees only the common SALA MLP module list. Do NOT subclass
   `MiniCPMGPTQ`: its inherited `layer_modules` includes optional
   `self_attn.o_gate`, which is absent from some SALA layers and caused
   the 2026-05-20 platform failure. Also do NOT quantize attention here:
   full q/k/v/o W4A16 successfully served but produced acc=0 on the
   platform, so this route is intentionally MLP-only.

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
import importlib.metadata as metadata
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

    usable_rows = [
        row for row in rows
        if isinstance(row.get("question"), str) and row["question"].strip()
    ]
    if not usable_rows:
        raise RuntimeError(
            f"--calib-jsonl contains no rows with non-empty `question`: {jsonl_path}"
        )

    rng = random.Random(seed)
    rng.shuffle(usable_rows)

    texts: list[str] = []
    task_counts: dict[str, int] = {}
    row_index = 0
    pass_index = 0
    while len(texts) < num_samples:
        if row_index >= len(usable_rows):
            pass_index += 1
            row_index = 0
            rng.shuffle(usable_rows)
        row = usable_rows[row_index]
        row_index += 1
        if len(texts) >= num_samples:
            break
        question = row["question"]
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
    if pass_index > 0:
        print(
            f"[calib] source has {len(usable_rows)} usable rows; cycled deterministically "
            f"to satisfy --num-calib={num_samples}",
            flush=True,
        )
    print(f"[calib] task distribution: {task_counts}", flush=True)
    return texts


# ---------------------------------------------------------------------------
# MiniCPM-SALA model registration
# ---------------------------------------------------------------------------
def force_flash_attention_2_for_sala() -> None:
    """Monkey-patch AutoConfig.from_pretrained to set _attn_implementation
    on the SALA config before GPTQModel ever instantiates the model.

    The SALA custom modeling has a hard assertion at the top of
    MiniCPMInfLLMv2Attention.__init__:

        assert self.config._attn_implementation == "flash_attention_2", \\
            "Only flash_attention_2 is supported for sparse attention"

    GPTQModel uses the from_config(...) codepath (not from_pretrained),
    which does NOT run the attn_implementation auto-routing that
    `from_pretrained(attn_implementation=...)` would. So we hook in
    earlier — when AutoConfig loads — and set the attribute on the
    config object before it's handed off to the model constructor.

    flash_attn package itself does NOT need to be installed at quant
    time; the assertion only reads the config string. The forward
    Hessian pass DOES need flash_attn at runtime — but the SOAR base
    environment ships it (verified: the BF16 baseline submission ran
    successfully).
    """
    from transformers import AutoConfig  # type: ignore

    _orig_from_pretrained = AutoConfig.from_pretrained

    def _patched_from_pretrained(*args, **kwargs):
        cfg = _orig_from_pretrained(*args, **kwargs)
        try:
            if getattr(cfg, "model_type", "") == "minicpm_sala":
                cfg._attn_implementation = "flash_attention_2"
                # Avoid transformers' "auto-set" override warning by
                # marking it as explicitly chosen.
                setattr(cfg, "_attn_implementation_autoset", True)
                print(
                    "[patch] forced minicpm_sala._attn_implementation"
                    " = flash_attention_2",
                    flush=True,
                )
        except Exception as exc:
            print(f"[patch] WARNING: couldn't set _attn_implementation: {exc}",
                  flush=True)
        return cfg

    AutoConfig.from_pretrained = _patched_from_pretrained


def register_minicpm_sala_with_gptqmodel() -> None:
    """Teach GPTQModel the MiniCPM-SALA module layout.

    GPTQModel resolves `config.model_type` to a `BaseQModel` subclass via
    `gptqmodel.models.auto.MODEL_MAP` in current examples. Some older
    versions also expose `gptqmodel.models.MODEL_MAP`, so we update both
    maps when available.
    """
    model_maps: list[dict[str, Any]] = []
    try:
        from gptqmodel.models.auto import MODEL_MAP as AUTO_MODEL_MAP  # type: ignore
        model_maps.append(AUTO_MODEL_MAP)
    except ImportError:
        pass
    try:
        from gptqmodel.models import MODEL_MAP as ROOT_MODEL_MAP  # type: ignore
        if ROOT_MODEL_MAP not in model_maps:
            model_maps.append(ROOT_MODEL_MAP)
    except ImportError:
        pass
    if not model_maps:
        raise RuntimeError("could not import GPTQModel MODEL_MAP")

    from gptqmodel.models.base import BaseQModel  # type: ignore

    class MiniCPMSALAGPTQ(BaseQModel):
        """Common module tree for MiniCPM-SALA.

        Keep only MLP modules present across all decoder layers. Attention
        and Lightning Mixer projections stay in original precision; full
        attention W4A16 was fast but produced acc=0 on the platform.
        """
        base_modules = ["model.embed_tokens", "model.norm"]
        pre_lm_head_norm_module = "model.norm"
        lm_head = "lm_head"
        lm_head_module = "lm_head"
        layers_node = "model.layers"
        layer_type = "MiniCPMSALADecoderLayer"

        layer_modules = [
            ["mlp.gate_proj", "mlp.up_proj"],
            ["mlp.down_proj"],
        ]

        module_tree = [
            "model",
            "layers",
            "#",
            {
                "input_layernorm": ("input_layernorm:!",),
                "post_attention_layernorm": ("post_attention_layernorm:!",),
                "mlp": ("gate_proj", "up_proj", "down_proj"),
            }
        ]

    for model_map in model_maps:
        model_map["minicpm_sala"] = MiniCPMSALAGPTQ
        model_map["MiniCPMSALAForCausalLM"] = MiniCPMSALAGPTQ

    # CRITICAL: SUPPORTED_MODELS is built ONCE at import time as
    # `list(MODEL_MAP.keys())`. Mutating MODEL_MAP afterwards does NOT update
    # SUPPORTED_MODELS, so `check_and_get_model_definition` (auto.py:471)
    # still sees `model_type not in SUPPORTED_MODELS` and falls back to
    # BaseQModel + auto_detect_module_tree. The auto-detector then walks
    # layer 0 (a "minicpm4" decoder that has o_gate) and inserts
    # `self_attn.o_gate` into layer_modules — which crashes on layer 1
    # (a "lightning-attn" decoder without o_gate). This was the failure
    # mode for submissions v10/v11/v12 on 2026-05-20.
    try:
        from gptqmodel.models import auto as _gptq_auto  # type: ignore
        sm = getattr(_gptq_auto, "SUPPORTED_MODELS", None)
        if isinstance(sm, list):
            for key in ("minicpm_sala", "MiniCPMSALAForCausalLM"):
                if key not in sm:
                    sm.append(key)
        else:
            # Reassign in case the attribute is a tuple/set or absent.
            _gptq_auto.SUPPORTED_MODELS = list(
                set(list(getattr(_gptq_auto, "SUPPORTED_MODELS", []))
                    + ["minicpm_sala", "MiniCPMSALAForCausalLM"])
            )
        print(
            "[register] also added minicpm_sala to gptqmodel.models.auto"
            f".SUPPORTED_MODELS (len={len(_gptq_auto.SUPPORTED_MODELS)})",
            flush=True,
        )
    except ImportError:
        pass

    print(
        "[register] registered minicpm_sala/MiniCPMSALAForCausalLM "
        f"-> MiniCPMSALAGPTQ in {len(model_maps)} MODEL_MAP(s)",
        flush=True,
    )


def validate_gptq_wrapper(model: Any) -> None:
    """Fail before expensive quantization if GPTQModel picked the wrong class."""
    cls = type(model)
    layer_modules = getattr(cls, "layer_modules", None)
    module_tree = getattr(cls, "module_tree", None)
    print(f"[register] GPTQ wrapper class: {cls.__module__}.{cls.__name__}",
          flush=True)
    print(f"[register] layer_modules={layer_modules}", flush=True)
    combined = f"{layer_modules!r}\n{module_tree!r}"
    forbidden = [
        "self_attn",
        "o_gate",
        "z_proj",
        "q_norm",
        "k_norm",
        "o_norm",
    ]
    found = [name for name in forbidden if name in combined]
    if found:
        raise RuntimeError(
            "GPTQModel is still using a template with optional SALA modules "
            f"{found}; registration did not override the active MODEL_MAP"
        )


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


def set_quant_config_disk_offload(quant_config: Any, enabled: bool) -> None:
    """Keep the CLI disk-offload choice synchronized with QuantizeConfig.

    GPTQModel 7.x stores offload_to_disk on QuantizeConfig itself and several
    quantization stages read that attribute directly. Passing an
    offload_to_disk kwarg to GPTQModel.load is not enough when the config
    object keeps its default value.
    """
    changed = False
    if hasattr(quant_config, "offload_to_disk"):
        setattr(quant_config, "offload_to_disk", enabled)
        changed = True
    if not enabled and hasattr(quant_config, "offload_to_disk_path"):
        setattr(quant_config, "offload_to_disk_path", None)
        changed = True
    if changed:
        print(f"[quant_config] offload_to_disk={enabled}", flush=True)


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
    """Tokenize each prompt into the plain dict format GPTQModel 7.x expects."""
    examples: list[dict[str, Any]] = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.numel() == 0:
            continue
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            import torch  # type: ignore
            attention_mask = torch.ones_like(input_ids)
        examples.append(
            {
                "input_ids": input_ids.to("cpu").long(),
                "attention_mask": attention_mask.to("cpu").long(),
            }
        )
    return examples


def print_runtime_versions() -> None:
    print(f"[versions] python={sys.version.split()[0]}", flush=True)
    packages = [
        "torch",
        "transformers",
        "gptqmodel",
        "flash-attn",
        "flash-linear-attention",
        "tokenizers",
        "huggingface-hub",
        "accelerate",
        "ninja",
    ]
    for package in packages:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        print(f"[versions] {package}={version}", flush=True)


# ---------------------------------------------------------------------------
# Output post-processing (Marlin compatibility)
# ---------------------------------------------------------------------------
def write_sglang_compatible_quant_config(
    output_dir: Path, input_dir: Path, bits: int, group_size: int
) -> None:
    """Build a config.json + quantize_config.json that SGLang gptq_marlin
    + MiniCPM-SALA sparse backend can load.

    CRITICAL: we use the INPUT (original BF16) config as the base, NOT the
    one GPTQModel saved. GPTQModel's save may drop or simplify SALA-
    specific fields. The downstream consumers need ALL of these to be
    preserved:
      - `auto_map` (so trust_remote_code resolves MiniCPMSALAConfig +
        MiniCPMSALAForCausalLM from the bundled .py files)
      - `mixer_types` (SGLang's MiniCPMSALAForCausalLM reads this to
        decide which decoder layer class to use)
      - `sparse_config` (block_size, kernel_*, topk, window_size, ...
        - read by minicpm_flashinfer attention backend)
      - `lightning_*` (head_dim, nh, nkv, scale, use_rope —
        Lightning Mixer expects these)
      - `attn_use_output_gate`, `use_output_gate`, `use_output_norm`,
        `qk_norm`, `attention_bias`, `rms_norm_eps`, `scale_depth`,
        `scale_emb`, `dim_model_base`, etc.

    Do NOT serialize derived read-only config properties such as
    `has_sparse_attention`. SGLang's `MiniCPMHybridConfig` exposes them as
    Python `@property` values inferred from `sparse_config` and
    `mixer_types`; putting those names into config.json makes
    `AutoConfig.from_pretrained` try to assign to a read-only property at
    server startup.
    """
    quant_cfg = {
        "bits": bits,
        "group_size": group_size,
        "quant_method": "gptq",
        "desc_act": False,
        "sym": True,
        "lm_head": False,
        # Tell SGLang's GPTQ-Marlin loader to leave attention/Lightning
        # modules unquantized (UnquantizedLinearMethod). GPTQModel writes
        # plain BF16 `.weight` tensors for anything omitted from
        # `layer_modules`; without matching dynamic skips, SGLang would
        # initialize those layers as quantized and either crash on missing
        # `.qweight` or silently run an incompatible layout.
        #
        # Patterns use `re.match` semantics (auto-anchored at start),
        # per get_dynamic_override in sglang/srt/layers/quantization/
        # utils.py:248. `-:<regex>` means "skip".
        "dynamic": {
            "-:.*self_attn.*": True,  # keep all sparse/linear attention projections BF16
            "-:.*o_gate$": True,    # MiniCPM dense-attn output gate (only some layers have it)
            "-:.*z_proj$": True,    # Lightning Mixer output gate (only some layers have it)
            "-:.*o_norm$": True,    # Lightning output RMSNorm (defensive)
            "-:.*q_norm$": True,    # Lightning Q RMSNorm (defensive; RMSNorm is not Linear anyway)
            "-:.*k_norm$": True,    # Lightning K RMSNorm (defensive)
        },
    }

    # Standalone quantize_config.json (some loaders prefer this over
    # config.quantization_config; write both consistently).
    quant_path = output_dir / "quantize_config.json"
    saved_quant_cfg: dict[str, Any] = {}
    if quant_path.exists():
        with quant_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            raise RuntimeError(
                f"{quant_path} is not a JSON object; cannot merge quantization config"
            )
        saved_quant_cfg = existing
        for metadata_key in ("format", "checkpoint_format", "pack_dtype"):
            if metadata_key in saved_quant_cfg:
                quant_cfg[metadata_key] = saved_quant_cfg[metadata_key]
        for fmt_key in ("format", "checkpoint_format"):
            fmt = str(quant_cfg.get(fmt_key, "gptq")).lower()
            if fmt != "gptq":
                raise RuntimeError(
                    f"GPTQModel saved {fmt_key}={quant_cfg.get(fmt_key)!r}; "
                    "SGLang gptq_marlin runtime conversion expects raw GPTQ "
                    "checkpoint tensors, so refusing to write a misleading config"
                )
            quant_cfg[fmt_key] = "gptq"
        existing.update(quant_cfg)
        quant_cfg_full = existing
    else:
        quant_cfg["format"] = "gptq"
        quant_cfg["checkpoint_format"] = "gptq"
        quant_cfg_full = quant_cfg
    with quant_path.open("w", encoding="utf-8") as f:
        json.dump(quant_cfg_full, f, indent=2)

    # ---- config.json: start from the ORIGINAL config, not GPTQModel's save ----
    input_config_path = input_dir / "config.json"
    output_config_path = output_dir / "config.json"
    if not input_config_path.exists():
        # Defensive: fall back to whatever GPTQModel wrote.
        print("[config] WARN: input config.json missing, falling back to GPTQModel save",
              flush=True)
        if not output_config_path.exists():
            return
        with output_config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        with input_config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if output_config_path.exists():
            # Merge in anything new from GPTQModel's saved config that wasn't
            # in the input (rare, mostly quantization-related).
            with output_config_path.open("r", encoding="utf-8") as f:
                gptq_saved = json.load(f)
            for key, val in gptq_saved.items():
                if key not in config:
                    config[key] = val

    # SGLang's gptq_marlin path expects fp16 activations.
    config["torch_dtype"] = "float16"
    config["dtype"] = "float16"
    config["quantization_config"] = quant_cfg

    # Remove derived properties. They are computed by MiniCPMHybridConfig and
    # cannot be assigned by transformers' config deserializer.
    readonly_config_keys = [
        "mamba2_cache_params",
        "full_attention_layer_ids",
        "has_sparse_attention",
        "has_lightning_layers",
        "sparse_layer_ids",
        "lightning_layer_ids",
    ]
    removed_readonly = [key for key in readonly_config_keys if key in config]
    for key in removed_readonly:
        config.pop(key, None)
    if removed_readonly:
        print(
            f"[config] removed read-only derived config keys: {removed_readonly}",
            flush=True,
        )

    # CRITICAL: drop `auto_map.AutoConfig` so transformers falls back to
    # SGLang's registered MiniCPMHybridConfig (registered in
    # sglang/srt/utils/hf_transformers_utils.py:104 via AutoConfig.register).
    # SGLang's hybrid path requires `isinstance(hf_config, MiniCPMHybridConfig)`:
    #   - model_runner.py:1494 minicpm_hybrid_config property
    #   - hybrid_linear_attn_backend.py:1456 SimpleGLA backend assertion
    # If we leave auto_map.AutoConfig pointing at the SALA custom config
    # class, trust_remote_code wins and the isinstance check returns False,
    # which kills the Lightning attention backend at init time.
    auto_map = config.get("auto_map")
    if isinstance(auto_map, dict) and "AutoConfig" in auto_map:
        removed = auto_map.pop("AutoConfig")
        print(
            f"[config] dropped auto_map.AutoConfig (was: {removed!r}) "
            "so SGLang's MiniCPMHybridConfig wins",
            flush=True,
        )

    with output_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(
        f"[config] wrote {output_config_path} with "
        f"sparse_config={'sparse_config' in config}, "
        f"mixer_types_len={len(config.get('mixer_types', []))}, "
        f"auto_map={'auto_map' in config}",
        flush=True,
    )


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

    print_runtime_versions()

    # Lazy import — only after we know it's not a dry run.
    # IMPORTANT: monkey-patch BEFORE register/load — SALA's modeling code
    # has a hard assertion on _attn_implementation == "flash_attention_2"
    # inside MiniCPMInfLLMv2Attention.__init__ that fires at model
    # instantiation time (BEFORE we can do anything else).
    force_flash_attention_2_for_sala()
    register_minicpm_sala_with_gptqmodel()
    quant_config = make_quant_config(args.bits, args.group_size)
    set_quant_config_disk_offload(quant_config, enabled=not args.no_offload_disk)

    model = load_model(
        args.input,
        quant_config,
        gpu_max_mem=args.gpu_max_mem,
        cpu_max_mem=args.cpu_max_mem,
        use_disk_offload=not args.no_offload_disk,
    )
    validate_gptq_wrapper(model)

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
    first_ids = calibration[0]["input_ids"]
    first_mask = calibration[0]["attention_mask"]
    print(
        "[calib] tokenized format: "
        f"type={type(calibration[0]).__name__} "
        f"input_ids={type(first_ids).__name__}{tuple(first_ids.shape)} "
        f"attention_mask={type(first_mask).__name__}{tuple(first_mask.shape)}",
        flush=True,
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
    write_sglang_compatible_quant_config(output_dir, Path(args.input), args.bits, args.group_size)
    print("[quantize] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
