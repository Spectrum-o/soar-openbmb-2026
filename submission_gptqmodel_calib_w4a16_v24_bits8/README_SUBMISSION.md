# v24_bits8 — fork of v23 with W8A16 instead of W4A16

> **PRECONDITION**: only submit if v23 platform returns `acc_ori = 0`
> AND we've ruled out H1+H4 (tokenizer + qzeros) AND we suspect the
> 4-bit kernel itself is the issue. This is a SANITY-CHECK variant.

## The diff vs v23

In `prepare_model.sh`, append `--bits 8` to `EXTRA_ARGS`:

```diff
- EXTRA_ARGS+=(--no-offload-disk)
+ EXTRA_ARGS+=(--no-offload-disk --bits 8)
```

## Why this might matter

If W4A16 produces garbage on the platform but W8A16 produces correct
output, then the failure is in the 4-bit Marlin kernel or the 4-bit
quantization pipeline specifically, not in our config + tokenizer
infrastructure. This is a strong diagnostic signal — most quant fixes
target the 4-bit-specific path, and 8-bit is much more tolerant.

W8A16 typical accuracy on SALA: > 95% of BF16 baseline (vs W4A16
~50% on our broken local config). So if v24_bits8 lands ~80+, we
know the platform pipeline is healthy and the residual W4A16 gap is
purely quantization noise.

If v24_bits8 ALSO returns acc=0 on platform, then the failure is in
the GPTQ pipeline itself OR an environment factor that's bit-agnostic
(e.g., tokenizer, modeling code, dtype handling) — NOT in the 4-bit
kernel.

## Variant dir structure

```
submission_gptqmodel_calib_w4a16_v24_bits8/
├── README_SUBMISSION.md             ← tracked, this file
├── prepare_model.sh                 ← tracked, adds --bits 8
├── prepare_env.sh                   → symlink to v23
├── quantize_gptqmodel_w4a16.py      → symlink to v23
├── perf_public_set.jsonl            → symlink to v23
├── flash_attn-...whl                → symlink to v23
└── sglang                           → symlink to v23
```

## Pack + submit

```bash
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v24_bits8 \
    --check-only

python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v24_bits8 \
    --output soar_gptqmodel_calib_w4a16_mlp_only_submission_20260522_v24_bits8.tar.gz
```

## What's NOT changed

- All H1+H4 fixes from v23 (hardened qzeros, tokenizer overwrite)
- qzeros repair covers both W4 `0x77777777 -> 0x88888888` and
  W8 `0x7f7f7f7f -> 0x80808080`; without the W8 case this variant
  can fail in `prepare_model` before SGLang starts.
- MLP-only quant scope (same `dynamic` skip patterns)
- SGLang args identical
- Calibration recipe identical

The ONLY thing flipped is `--bits 4` → `--bits 8`. The quantize script
already supports both via `parser.add_argument("--bits", type=int, default=4)`
and `make_quant_config(bits, group_size)`.

## Expected outcomes vs diagnosis

| Platform result | What it means | Next action |
|---|---|---|
| `acc_ori >= 70` | The 4-bit kernel is the issue; W8A16 is the rescue | Ship W8A16 as the production submission; accept the 2x memory cost; iterate on W4A16 fixes in parallel |
| `acc_ori >= 30` but < 70 | Improvement but still gated by something else | Same pipeline issues affect both 4 and 8-bit; check log for FATAL / WARN |
| `acc_ori = 0` | Issue is bit-agnostic (modeling, tokenizer, dtype, etc.) | Move to v24_no_dtype_key or v24_pin_transformers |
| Quant takes 2x longer than expected | Normal (W8A16 ~ 1.5-2x quant time of W4A16); ok if within 90 min timeout | n/a |

## Cost note

W8A16 tarball serves a model with 2x the weight memory of W4A16
(~13GB instead of ~6.5GB). On RTX PRO 6000 96GB, this is fine. But
on a smaller-memory platform (e.g. A100 40GB without quant), it'd
OOM. Verified RTX PRO 6000 has enough VRAM per SUBMISSIONS.md.
