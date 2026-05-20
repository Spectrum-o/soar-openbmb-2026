#!/usr/bin/env python3
"""GPTQModel W4A16 quantization for MiniCPM-SALA — v17-minconfig variant.

Forked from the 2026-05-20 v17 submission (acc=0). Reverts the three
v17-specific changes that the RTN submission (acc=42) did NOT make, while
leaving the module set and everything else identical to v17 so an A/B with
v17 attributes the regression precisely.

P0-A. Minimal config.json rewrite. v17 dropped `auto_map.AutoConfig`
      (likely caused SALA-specific fields like `scale_emb` to be ignored
      by SGLang's MiniCPMHybridConfig fallback) and statically wrote
      `has_sparse_attention=True` (collides with the @property on
      MiniCPMHybridConfig). This variant only adds `quantization_config`
      and `torch_dtype`. Nothing else in config.json is touched.

P0-B. Tail-truncation of calibration prompts. v17 used the HF default
      `truncation_side="right"`, keeping only the FIRST 4096 tokens of
      ~30K-token perf_public_set rows (i.e. haystack filler, with the
      actual question structure cut off). This variant sets
      `truncation_side="left"` and applies the chat template so the
      kept window is the tail (question + answer format) that serving
      actually sees.

P0-C. Default `--max-calib-len` 4096 -> 8192.

Module set is unchanged: full attention (q/k/v/o_proj) + MLP, with
o_gate / z_proj / *_norm left BF16 (matching v17's `dynamic` skip).
This is NOT MLP-only.

If THIS variant clears acc ≥ 60, the config rewrites and/or the
truncation bug were the regression vs RTN. If acc is still 0, the
remaining suspects are desc_act=False on outlier-heavy Q/K weights,
and the o_gate/z_proj mixed-precision break.

Usage (on the GPU node):

    python3 quantize_gptqmodel_w4a16.py \\
        --input  /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\
        --output /root/autodl-tmp/models/MiniCPM-SALA-W4A16-GPTQ \\
        --calib-jsonl /path/to/perf_public_set.jsonl \\
        --num-calib 256 \\
        --max-calib-len 8192

`--dry-run` prints the resolved config and the first 3 calibration
prompts; no GPU needed.
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


def _stub_transformers_for_gptqmodel_7() -> None:
    """gptqmodel 7.0 was designed against transformers 5.x; SALA pins 4.57.1.
    Flip gptqmodel's own early-return sentinel so its causal_conv1d
    hub-kernel compat patch never runs (it needs transformers 5.x APIs).
    Also alias the renamed PreTrainedConfig. MUST run before importing gptqmodel.
    """
    import transformers
    if not hasattr(transformers, "PreTrainedConfig"):
        transformers.PreTrainedConfig = transformers.PretrainedConfig
    try:
        import transformers.integrations.hub_kernels as _hk
        _hk._gptqmodel_local_causal_conv1d_kernel = True
    except ImportError:
        pass


_stub_transformers_for_gptqmodel_7()


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
        default=8192,
        help="Max tokens per calibration prompt. Higher = better for "
             "long-context preservation, but more memory + time. "
             "v17 used 4096 with HF-default right-truncation, which "
             "kept only haystack filler; this variant left-truncates "
             "(keeps the tail with the question), so 8192 is safe.",
    )
    parser.add_argument(
        "--calib-window-mode",
        default="tail",
        choices=["tail", "multi-adaptive"],
        help="How to slice prompts into calibration samples. "
             "'tail' (default): keep the last --max-calib-len tokens of "
             "each prompt via tokenizer truncation — matches v21/v22 "
             "behavior. 'multi-adaptive': split each prompt into 1-3 "
             "non-overlapping --max-calib-len windows depending on its "
             "full token length (short prompts as-is; medium tail-only; "
             "long prompts tail+mid; super-long add one random mid "
             "window). Multi-adaptive can roughly 2-3x the calibration "
             "sample count — keep --num-calib <= 512 to stay under the "
             "90-min prepare_model timeout.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Skip apply_chat_template() during calibration tokenization. "
             "Use this for ablation: SOAR perf_public_set evaluation "
             "feeds raw `question` strings to the model, but the default "
             "calibration wraps in `<用户>...<AI>`. v21 (chat template ON "
             "+ left-trunc + 8K) got acc=49 across 150 samples — well "
             "below v18-era acc=63 on 30 samples with chat template OFF "
             "+ right-trunc + 4K. This flag enables A/B isolating the "
             "chat-template variable.",
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

        Keep only modules present across all decoder layers. GPTQModel will
        quantize the modules listed here; optional gates/norms are left in
        their original precision by being omitted from the quantization tree.
        """
        base_modules = ["model.embed_tokens", "model.norm"]
        pre_lm_head_norm_module = "model.norm"
        lm_head = "lm_head"
        lm_head_module = "lm_head"
        layers_node = "model.layers"
        layer_type = "MiniCPMSALADecoderLayer"

        layer_modules = [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["mlp.gate_proj", "mlp.up_proj"],
            ["mlp.down_proj"],
        ]

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
                ),
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
    forbidden = ["o_gate", "z_proj", "q_norm", "k_norm", "o_norm"]
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


def _slice_windows_for_prompt(
    n_tokens: int,
    max_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Return (start, end) windows for one prompt under multi-adaptive policy.

    Input-aware split of a prompt of length ``n_tokens`` into 1-3
    non-overlapping ``max_len``-sized windows:

        n <= L:              [(0, n)]                       short: all
        L  <  n <= 4L:       [(n-L, n)]                     medium: tail only
        4L <  n <= 12.5L:    tail + centered mid             long
        n  >  12.5L:         tail + mid + 1 random non-overlap window

    where L = max_len. The HEAD window is deliberately NEVER taken: the
    first 8K of a 30K perf_public_set row is pure haystack filler whose
    activations would mis-direct GPTQ's per-channel scale selection.
    """
    L = max_len
    if n_tokens <= L:
        return [(0, n_tokens)]
    if n_tokens <= 4 * L:
        return [(n_tokens - L, n_tokens)]
    windows = [(n_tokens - L, n_tokens)]
    mid_start = (n_tokens - L) // 2
    windows.append((mid_start, mid_start + L))
    if n_tokens > int(12.5 * L):
        lo = L
        hi = n_tokens - 2 * L
        if hi > lo:
            start = rng.randint(lo, hi - 1)
            windows.append((start, start + L))
    return windows


def tokenize_calibration(
    tokenizer,
    texts: list[str],
    max_len: int,
    window_mode: str = "tail",
    seed: int = 42,
    disable_chat_template: bool = False,
) -> list[dict[str, Any]]:
    """Tokenize each prompt into the plain dict format GPTQModel 7.x expects.

    Differences from v17:
      1. `truncation_side = "left"` so we keep the TAIL of long prompts.
         perf_public_set rows have the actual question structure at the
         end (haystack/needle layout, MCQ choices at the end, etc.).
         v17 used the HF default `truncation_side="right"`, which kept
         only the first 4096 tokens of ~30K-token rows — i.e. pure
         filler. GPTQ's Hessian was then fit to filler activations,
         which is the leading hypothesis for v17's acc=0.
      2. Apply the chat template if the tokenizer defines one. SALA is
         instruction-tuned, so serving sees `<用户>...<AI>` wrappers;
         calibrating on bare `question` strings makes the Hessian see
         a different token distribution at sequence boundaries than
         what the deployed model actually consumes.

    ``window_mode``:
      - ``"tail"`` (default): one window per prompt — the last ``max_len``
        tokens, via tokenizer truncation. Byte-equivalent to the
        pre-2026-05-21 behavior; v21/v22 quant artifacts were produced
        with this.
      - ``"multi-adaptive"``: 1-3 non-overlapping ``max_len``-sized windows
        per prompt, sized by ``_slice_windows_for_prompt``. Tail is always
        kept; mid is added on long prompts. Returned windows are shuffled
        with ``seed`` to keep GPTQ batches diverse. Inflates total sample
        count for long prompts — keep ``--num-calib`` <= 512 to stay under
        the 90-min prepare_model timeout.
    """
    if window_mode not in ("tail", "multi-adaptive"):
        raise ValueError(
            f"--calib-window-mode must be 'tail' or 'multi-adaptive', "
            f"got {window_mode!r}"
        )

    original_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"

    if disable_chat_template:
        use_chat_template = False
        print("[calib] chat template DISABLED via --no-chat-template", flush=True)
    else:
        use_chat_template = (
            hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None)
        )
        if use_chat_template:
            print("[calib] applying chat template to calibration prompts", flush=True)
        else:
            print("[calib] tokenizer has no chat_template; using raw prompts",
                  flush=True)

    def _render(text: str) -> str:
        if not use_chat_template:
            return text
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            print(f"[calib] chat template render failed ({exc}); "
                  "falling back to raw text",
                  flush=True)
            return text

    examples: list[dict[str, Any]] = []

    if window_mode == "tail":
        # ORIGINAL PATH — preserves byte-for-byte behavior of v21/v22 quant.
        for text in texts:
            rendered = _render(text)
            encoded = tokenizer(
                rendered,
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
    else:
        # multi-adaptive: tokenize without truncation, slice into windows.
        import torch  # type: ignore
        rng = random.Random(seed)
        buckets = {
            "short(<=L)": 0,
            "tail-only(<=4L)": 0,
            "tail+mid(<=12.5L)": 0,
            "tail+mid+rand(>12.5L)": 0,
        }
        for text in texts:
            rendered = _render(text)
            encoded = tokenizer(
                rendered,
                truncation=False,
                padding=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = encoded.get("input_ids")
            if input_ids is None or input_ids.numel() == 0:
                continue
            n = int(input_ids.shape[-1])
            slices = _slice_windows_for_prompt(n, max_len, rng)
            for s, e in slices:
                w = input_ids[:, s:e].contiguous()
                am = torch.ones_like(w)
                examples.append(
                    {
                        "input_ids": w.to("cpu").long(),
                        "attention_mask": am.to("cpu").long(),
                    }
                )
            if n <= max_len:
                buckets["short(<=L)"] += 1
            elif n <= 4 * max_len:
                buckets["tail-only(<=4L)"] += 1
            elif n <= int(12.5 * max_len):
                buckets["tail+mid(<=12.5L)"] += 1
            else:
                buckets["tail+mid+rand(>12.5L)"] += 1
        rng.shuffle(examples)
        print(
            f"[calib] multi-adaptive: {len(texts)} prompts -> "
            f"{len(examples)} windows (L={max_len}); per-bucket: {buckets}",
            flush=True,
        )

    tokenizer.truncation_side = original_side
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
    """Write quant configs with MINIMAL changes to the original config.json.

    This mirrors the RTN-scalefix submission (acc=42), which only set
    `torch_dtype=float16` and added a `quantization_config` block — and
    crucially did NOT remove `auto_map.AutoConfig` or write
    `has_sparse_attention` statically. The v17 submission did both of
    those, plausibly causing acc=0 by either:
      - losing SALA-specific config fields (scale_emb, scale_depth,
        dim_model_base, ...) when MiniCPMHybridConfig took over, or
      - colliding with the @property descriptor on MiniCPMHybridConfig.

    The `dynamic` skip patterns for o_gate / z_proj / *_norm are still
    needed because GPTQModel leaves them unquantized (they are not in
    `layer_modules`) — without the skip pattern, SGLang's gptq_marlin
    loader would expect `.qweight` for those modules and KeyError on the
    `.weight` it finds in safetensors.
    """
    quant_cfg = {
        "bits": bits,
        "group_size": group_size,
        "quant_method": "gptq",
        "desc_act": False,
        "sym": True,
        "lm_head": False,
        # Skip patterns for modules GPTQModel left unquantized. Same as v17.
        # `-:<regex>` means "use UnquantizedLinearMethod" in SGLang's loader.
        "dynamic": {
            "-:.*o_gate$": True,
            "-:.*z_proj$": True,
            "-:.*o_norm$": True,
            "-:.*q_norm$": True,
            "-:.*k_norm$": True,
        },
    }

    # Standalone quantize_config.json (some loaders prefer this over
    # config.quantization_config; write both consistently).
    quant_path = output_dir / "quantize_config.json"
    if quant_path.exists():
        with quant_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(quant_cfg)
        quant_cfg_full = existing
    else:
        quant_cfg_full = quant_cfg
    with quant_path.open("w", encoding="utf-8") as f:
        json.dump(quant_cfg_full, f, indent=2)

    # config.json: start from the ORIGINAL SALA config and ONLY add the
    # two fields needed for SGLang+Marlin to load the quantized weights.
    # Do NOT touch auto_map, do NOT add has_sparse_attention, do NOT merge
    # in GPTQModel's saved config (which strips SALA-specific fields).
    input_config_path = input_dir / "config.json"
    output_config_path = output_dir / "config.json"
    if not input_config_path.exists():
        raise RuntimeError(
            f"input config.json missing at {input_config_path}; refusing to "
            "fall back to GPTQModel's saved config (it drops SALA fields)"
        )
    with input_config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # The ONLY two edits. Same minimal set as the RTN-scalefix submission.
    config["torch_dtype"] = "float16"
    config["quantization_config"] = quant_cfg

    with output_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(
        f"[config] wrote {output_config_path} (minimal-rewrite mode): "
        f"auto_map_kept={'auto_map' in config}, "
        f"has_sparse_attention_in_json={'has_sparse_attention' in config}, "
        f"scale_emb={config.get('scale_emb')}, "
        f"dim_model_base={config.get('dim_model_base')}",
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


def fix_qzeros_for_marlin(output_dir: Path) -> None:
    """Rewrite GPTQModel's qzeros 0x77777777 (=7 per slot) -> 0x88888888 (=8).
    gptqmodel 7.0.0 + sym=True writes qzero=7 per 4-bit slot; SGLang Marlin
    dequant `(q - qzero) * scale` then adds a per-weight `+1*scale` bias
    that accumulates to acc=0. Patch every shard's qzeros in place.
    """
    import torch
    from safetensors.torch import load_file, save_file

    BAD = 0x77777777
    GOOD_SIGNED = -2004318072  # int32 form of 0x88888888

    total_fixed = 0
    for shard in sorted(output_dir.glob("model-*.safetensors")):
        state = load_file(str(shard))
        n_fixed = 0
        for key, tensor in list(state.items()):
            if not key.endswith(".qzeros") or tensor.dtype != torch.int32:
                continue
            unique = tensor.flatten().unique().tolist()
            if unique == [BAD]:
                state[key] = torch.full_like(tensor, GOOD_SIGNED)
                n_fixed += 1
            elif unique == [GOOD_SIGNED]:
                pass
            else:
                print(f"[qzeros-fix] WARN: {key} unique={unique[:5]}; leaving alone",
                      flush=True)
        if n_fixed:
            save_file(state, str(shard))
            print(f"[qzeros-fix] rewrote {n_fixed} qzeros tensors in {shard.name}",
                  flush=True)
            total_fixed += n_fixed
    print(f"[qzeros-fix] total {total_fixed} qzeros 0x77777777 -> 0x88888888",
          flush=True)


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
    print(f"  --calib-window-mode {args.calib_window_mode}")
    print(f"  --no-chat-template  {args.no_chat_template}")
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
    if args.calib_window_mode == "multi-adaptive":
        print()
        print(f"Window slicing simulation (multi-adaptive, "
              f"L={args.max_calib_len}):")
        sim_rng = random.Random(args.seed)
        test_ns = (
            args.max_calib_len // 2,
            args.max_calib_len,
            2 * args.max_calib_len,
            4 * args.max_calib_len,
            8 * args.max_calib_len,
            13 * args.max_calib_len,
            20 * args.max_calib_len,
        )
        for n in test_ns:
            windows = _slice_windows_for_prompt(n, args.max_calib_len, sim_rng)
            ratio = f"{n / args.max_calib_len:.1f}L"
            spans = ", ".join(f"[{s:>6}:{e:>6}]" for s, e in windows)
            print(f"  N={n:7d} ({ratio:>5}) -> {len(windows)} window(s): {spans}")
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

    calibration = tokenize_calibration(
        tokenizer, prompts, args.max_calib_len,
        window_mode=args.calib_window_mode,
        seed=args.seed,
        disable_chat_template=args.no_chat_template,
    )
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
    fix_qzeros_for_marlin(output_dir)
    print("[quantize] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
