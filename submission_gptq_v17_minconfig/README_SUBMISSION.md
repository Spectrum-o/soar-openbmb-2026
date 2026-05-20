# SOAR W4A16 Submission — v17-minconfig

A targeted experiment to find the cause of v17's `acc=0.0` regression
(v17 went from RTN-scalefix `acc=42` down to `0`, even though GPTQ
should be ≥ RTN at the same bit-width).

## Hypothesis

v17 introduced **three** changes that RTN-scalefix did NOT make. Any
one (or combination) could explain the regression:

| Hypothesis | What v17 changed | Why this can produce acc=0 |
|---|---|---|
| H1 | Removed `auto_map.AutoConfig` from `config.json` | Forces SGLang's `MiniCPMHybridConfig` to deserialize the config; if `MiniCPMHybridConfig.__init__` doesn't accept SALA-specific fields like `scale_emb` (≈12), they are silently dropped. Missing `scale_emb` makes logits 12× too large → softmax fully saturated → wrong-but-confident outputs. |
| H2 | Statically wrote `has_sparse_attention: true` into `config.json` | `MiniCPMHybridConfig` exposes this as a `@property` (derived from `sparse_config` / `mixer_types`). Setting it via `setattr` collides with the descriptor — HF swallows the `AttributeError`, leaving the attribute in an undefined state. |
| H3 | Calibration `tokenizer(..., max_length=4096)` with default `truncation_side="right"` | perf_public_set rows have ~30K-token median with the question structure at the END. v17's GPTQ Hessian was fit to 4K tokens of pure haystack filler, then "compensated" weights in the wrong direction (worse than naive RTN). |

## What this variant changes vs v17

**Reverted (matches RTN-scalefix, the acc=42 baseline that didn't have these bugs):**

- **Minimal `config.json` rewrite.** Only adds `quantization_config` and
  sets `torch_dtype=float16`. Does NOT touch `auto_map`, does NOT write
  `has_sparse_attention`, does NOT merge in GPTQModel's saved config.
  This is the exact pattern from `quantize_gptq_rtn_sym.py` (RTN-scalefix).
- **Tail-truncation of calibration.** `tokenizer.truncation_side = "left"`
  + applies chat template if the tokenizer has one.
- **Default `--max-calib-len` 4096 → 8192.** Tail-window is the part GPTQ
  needs to see (question/answer format); 8K covers it comfortably with
  ~2× the per-sample Hessian cost.

**Kept identical to v17 (intentional — to keep the A/B clean):**

- `layer_modules = [q/k/v + o_proj, gate/up + down]` — full attention + MLP.
  NOT MLP-only (v18/v19 was the MLP-only experiment).
- `dynamic` skip patterns for `o_gate` / `z_proj` / `*_norm`.
- GPTQModel registration, `force_flash_attention_2_for_sala`, all loader plumbing.
- `prepare_env.sh`, bundled flash-attn wheel, `SGLANG_SERVER_ARGS`.

## How to read the result

| Outcome on platform | What it tells us |
|---|---|
| acc ≥ 60 | At least one of H1/H2/H3 was the regression. **Next step**: split into per-fix variants (config-only fix, calib-only fix) to attribute. |
| acc still 0 | H1/H2/H3 are not the regression. **Remaining suspects**: desc_act=False outlier-channel damage on Q/K (GPTQ-specific failure mode), or the `o_gate`/`z_proj` mixed-precision break. |
| acc ~ 40-60 | Partial recovery — likely calibration is the dominant fix and Q/K outliers are residual. |

## Files

| File | Source |
|---|---|
| `quantize_gptqmodel_w4a16.py` | Forked from `soar_gptqmodel_calib_w4a16_submission_20260520_v17.tar.gz`; patches in 3 places (write_sglang_compatible_quant_config, tokenize_calibration, --max-calib-len default) |
| `prepare_env.sh` | Identical to v17 / current MLP-only variant |
| `prepare_model.sh` | Same as v17 except `MAX_CALIB_LEN` default 4096 → 8192 |
| `perf_public_set.jsonl` | Bundled calibration source (150 SOAR public rows) |
| `flash_attn-...whl` | Symlink to MLP-only variant's bundled wheel |
| `sglang/` | Symlink to MLP-only variant's SGLang source |

## Local verification (no GPU)

```bash
# dry-run shows the calibration prompts and resolved config
python3 quantize_gptqmodel_w4a16.py \
    --input /any --output /any \
    --calib-jsonl perf_public_set.jsonl --num-calib 5 --dry-run
```

The config-writer and tail-truncation logic were independently verified
locally with fake configs/tokenizers (see chat log 2026-05-20). The
remaining GPU-only validation is the actual GPTQ Hessian forward pass.

## Open follow-ups (do NOT do here)

- **P1**: also restore `o_gate` / `z_proj` quantization (requires
  `layer_modules_strict=False` or a post-pass RTN). Separate variant.
- **P2**: multi-window slicing of long prompts (head + middle + tail
  windows per row). Separate variant.
- **P3**: position-id augmentation for RoPE coverage. Separate variant.
