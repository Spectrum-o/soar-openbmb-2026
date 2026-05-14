#!/usr/bin/env python3
"""
Quantize MiniCPM-SALA to W4A16 (GPTQ) for inference via sglang's compressed-tensors backend.

Champion's approach (SOAR Week 4 - 智算一队):
  - GPTQ algorithm (Hessian-based weight quantization)
  - W4A16 scheme (4-bit weights, 16-bit activations)
  - Marlin kernel at inference (~4x bandwidth savings, ideal for memory-bound decode)
  - Calibration dataset matched to eval distribution -> significant accuracy improvement

This script handles step 1 (quantize). After this runs, point sglang at the output
directory with --quantization compressed-tensors (or let it auto-detect from config).

Prerequisites:
  pip install llmcompressor>=0.4 datasets

Typical wall time on a single RTX PRO 6000 / A100: 60-180 minutes for a 9B model
depending on calibration size and max_seq_length.

Quick start (defaults: 512 samples of wikitext at 2048 tokens):
  python quantize_to_w4a16.py \\
      --model /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\
      --output /root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16

Custom calibration (recommended once SOAR eval distribution is known):
  python quantize_to_w4a16.py \\
      --model ... --output ... \\
      --calib-dataset HuggingFaceFW/fineweb \\
      --calib-split sample-10BT \\
      --num-samples 1024 \\
      --max-seq-len 4096
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantize MiniCPM-SALA to W4A16 GPTQ via llm-compressor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Path or HF id of the BF16 source model")
    p.add_argument("--output", required=True,
                   help="Directory to save the W4A16 model (will be created)")
    p.add_argument("--calib-dataset", default="wikitext",
                   help="HF datasets path for calibration")
    p.add_argument("--calib-config", default="wikitext-2-raw-v1",
                   help="Subset/config name within --calib-dataset (use empty string if N/A)")
    p.add_argument("--calib-split", default="train",
                   help="Split name within --calib-dataset")
    p.add_argument("--calib-text-column", default="text",
                   help="Column name containing the text to tokenize")
    p.add_argument("--num-samples", type=int, default=512,
                   help="Number of calibration samples")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Max tokens per calibration sample")
    p.add_argument("--group-size", type=int, default=128,
                   help="Quantization group size (128 is standard for Marlin)")
    p.add_argument("--ignore", default="lm_head",
                   help="Comma-separated module name patterns to skip (always includes lm_head)")
    p.add_argument("--dampening-frac", type=float, default=0.01,
                   help="GPTQ Hessian dampening (raise if numerical issues)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config and exit without quantizing")
    return p.parse_args()


def main():
    args = parse_args()

    # Late imports so --help works without the dependencies installed.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError as e:
        print(f"[fatal] missing dependency: {e}", file=sys.stderr)
        print("install with: pip install 'llmcompressor>=0.4' datasets transformers", file=sys.stderr)
        sys.exit(2)

    ignore_list = [s.strip() for s in args.ignore.split(",") if s.strip()]
    if "lm_head" not in ignore_list:
        ignore_list.append("lm_head")

    print("=" * 60)
    print(f"Source model:  {args.model}")
    print(f"Output dir:    {args.output}")
    print(f"Calibration:   {args.calib_dataset}/{args.calib_config or '-'}/{args.calib_split}")
    print(f"               {args.num_samples} samples, max_seq_len={args.max_seq_len}")
    print(f"Scheme:        W4A16, group_size={args.group_size}")
    print(f"Ignore:        {ignore_list}")
    print("=" * 60)

    if args.dry_run:
        return

    os.makedirs(args.output, exist_ok=True)

    print("[1/4] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # MiniCPM-SALA's remote modeling code asserts attn_implementation == "flash_attention_2"
    # (modeling_minicpm_sala.py:1328) because its InfLLMv2 sparse attention is only
    # wired through flash_attention_2; eager/sdpa abort. Requires flash-attn installed.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print("[2/4] Loading and tokenizing calibration data...")
    if args.calib_config:
        raw = load_dataset(args.calib_dataset, args.calib_config, split=args.calib_split)
    else:
        raw = load_dataset(args.calib_dataset, split=args.calib_split)

    # Drop empty rows then take the first num_samples
    raw = raw.filter(lambda r: r[args.calib_text_column].strip() != "")
    raw = raw.shuffle(seed=42).select(range(min(args.num_samples, len(raw))))

    def tokenize_fn(sample):
        return tokenizer(
            sample[args.calib_text_column],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )

    tokenized = raw.map(tokenize_fn, remove_columns=raw.column_names)
    print(f"        ready: {len(tokenized)} samples")

    print("[3/4] Running GPTQ quantization (this is the slow part)...")
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=ignore_list,
        dampening_frac=args.dampening_frac,
    )

    oneshot(
        model=model,
        dataset=tokenized,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=len(tokenized),
        output_dir=args.output,
        save_compressed=True,
    )

    print("[4/4] Saving tokenizer alongside model...")
    tokenizer.save_pretrained(args.output)

    print("=" * 60)
    print(f"DONE. Quantized model at: {args.output}")
    print("Next: bash run_sala_w4a16.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
