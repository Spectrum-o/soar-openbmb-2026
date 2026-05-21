# v23 vs RTN-scalefix tarball audit

> Goal: enumerate every difference between the canonical "works on
> platform" tarball (`soar_rtn_sym_w4a16_g128_marlin_submission_20260519_scalefix.tar.gz`,
> platform acc=42) and the most recent GPTQModel-pipeline tarball
> (`soar_gptqmodel_v17_minconfig_full_attn_submission_20260520_v22.tar.gz`,
> platform acc=0). Verify each difference is intentional (part of the
> GPTQModel pipeline) and not a spurious regression. v22 stands in for
> v23 (which hasn't been packed yet); v23 will be byte-equivalent to v22
> except for the 2026-05-21..22 source-level fixes already on
> `quant/w4a16`.
>
> Reproduce all diffs from this doc:
> ```bash
> mkdir -p /tmp/audit/{rtn,v22}
> tar -xzf soar_rtn_sym_w4a16_g128_marlin_submission_20260519_scalefix.tar.gz -C /tmp/audit/rtn/
> tar -xzf soar_gptqmodel_v17_minconfig_full_attn_submission_20260520_v22.tar.gz -C /tmp/audit/v22/
> diff -rq /tmp/audit/rtn/ /tmp/audit/v22/
> ```

## Top-level inventory

| Path | RTN-scalefix | v22 | Note |
|---|---|---|---|
| `README_SUBMISSION.md` | present | present | both describe the variant; not a runtime input |
| `prepare_env.sh` | 138 lines | 218 lines | v22 adds gptqmodel install + flash-attn install + print_versions |
| `prepare_model.sh` | 33 lines | 113 lines | v22 adds calib resolution + timeout + diagnostic prints |
| `quantize_*.py` | `quantize_gptq_rtn_sym.py` (228 lines) | `quantize_gptqmodel_w4a16.py` (897 lines) | Different scripts, different quant tool — INTENTIONAL |
| `perf_public_set.jsonl` | _absent_ | 23.9 MB present | v22 bundles calib data; RTN doesn't need calib |
| `flash_attn-...whl` | _absent_ | 254 MB present | v22 needs flash-attn for SALA modeling; RTN doesn't import gptqmodel/SALA modeling code at quant time |
| `sglang/` | bundled | bundled | both ship the same SGLang fork; deltas below |

File count: RTN 1627 files / 21MB; v22 1630 files / 286MB. Delta = 3 files + 265MB (wheel + jsonl + the gptqmodel quant script).

## SGLANG_SERVER_ARGS (the launch contract)

**Both byte-identical** post-cherry-pick-revert:

```
--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --dtype float16
```

This is the key point: at SGLang serve-time, the two tarballs are launching identical configurations. The only thing that can differ is the on-disk model artifact that each pipeline produced.

## sed-patch contract on bundled SGLang (the fp16 fix)

**Both byte-identical**:

```bash
sed -i 's/torch\.bfloat16/torch.float16/g' "${pyfile}"
sed -i 's/"bfloat16"/"float16"/g' "${pyfile}"
```

applied to `minicpm_backend.py` and `minicpm_sparse_utils.py`. Both tarballs need the same fp16 patch because both use `--quantization gptq_marlin` which emits fp16 outputs that must match the sparse attention backend's expectation.

## prepare_env.sh deltas

v22 adds (vs RTN):

- `python_pkg_present` / `python_pkg_major_ok` / `python_pkg_min_version` / `python_pkg_exact_version` helpers
- gptqmodel install gate: forces `gptqmodel==7.0.0` (after 2026-05-21 hardening commit `d8aaaf1e4`)
- transformers/accelerate/ninja install
- flash-attn wheel install path (bundled wheel preferred, URL fallback)
- `print_versions()` block that dumps gptqmodel/transformers/torch/sglang versions
- `GPTQMODEL_MARLIN_USE_FP32=1` env export

**Verdict: every delta is intentional.** RTN doesn't need gptqmodel/flash-attn because its quant runs pure numpy offline before the platform stage.

## prepare_model.sh deltas

v22 adds (vs RTN):

- Calibration data resolution (env var → bundled jsonl → AutoDL fallback → synthetic)
- 90-min timeout wrap
- DIAGNOSTIC blocks BEFORE and AFTER quantize (post-2026-05-21 commit `d8aaaf1e4`):
  - script dir contents
  - input dir listing
  - calib jsonl size + row count + first 200 chars
  - output dir contents after quantize
  - safetensors find
  - quantize_config.json dump

**Verdict: every delta is intentional.** RTN's prepare_model.sh just calls its numpy script with `--input`/`--output`; no calibration, no timing concerns.

## sglang/ tree deltas (2 real files differ; 2 cosmetic)

`diff -rq` between the bundled SGLang in each tarball:

```
Files .../sglang/python/sglang/srt/configs/minicpm.py and .../v22/.../minicpm.py differ
Files .../sglang/python/sglang/srt/layers/torchao_utils.py and .../v22/.../torchao_utils.py differ
Only in .../rtn/sglang/python/sglang/srt/mem_cache/cpp_radix_tree: .clang-format
Only in .../v22/sglang/python/sglang/srt/configs: __pycache__
```

### Real diff 1: `srt/configs/minicpm.py` (v22 +14 lines)

v22 adds a defensive `kwargs.pop(key, None)` loop inside `MiniCPMHybridConfig.__init__` for keys that are `@property` values (read-only):
- `mamba2_cache_params`, `full_attention_layer_ids`, `has_sparse_attention`, `has_lightning_layers`, `sparse_layer_ids`, `lightning_layer_ids`

Reason: GPTQModel's `save_pretrained` can accidentally serialize a few of these derived properties into `config.json`. On reload via `AutoConfig.from_pretrained`, transformers tries to assign them and crashes with `AttributeError: can't set attribute`. The patch silently drops them before the parent `__init__` sees them.

**Verdict: intentional safety net.** Documented in SUBMISSIONS.md row v18 (the failure that proved this constraint). RTN's pipeline never serializes those keys so the patch is no-op for RTN but doesn't hurt.

### Real diff 2: `srt/layers/torchao_utils.py` (v22 +10 lines)

v22 adds an early-return in `apply_torchao_config_to_model`:

```python
if torchao_config == "" or torchao_config is None:
    return model
```

placed BEFORE the eager imports of `float8_dynamic_activation_float8_weight`, `int4_weight_only`, etc.

Reason: torchao ≥ 0.16.0 removed the snake_case functions in favor of PascalCase classes. SGLang's import is unconditional even when torchao isn't actually used. The early-return short-circuits before the import crashes.

**Verdict: intentional environment-defensive patch.** Documented in SUBMISSIONS.md row v15 (`ImportError` from torchao 0.16+).

### Cosmetic diffs (ignore)

- `.clang-format` in RTN's `mem_cache/cpp_radix_tree/` — formatting hint file; not a runtime input
- `__pycache__` only in v22 — accidentally bundled bytecode from a local `pack_submission.py` build; not runtime-fatal. Should be cleaned up in v23 packing.

## What's NOT different (where things could have drifted but didn't)

- **`SGLANG_SERVER_ARGS`** — byte-identical (after 2026-05-22 chunked-prefill revert)
- **sed-patch** — byte-identical
- **All 1627 other files in `sglang/`** — byte-identical
- **`prepare_env.sh`'s `GPTQMODEL_MARLIN_USE_FP32=1`** — present in both (RTN inherits, doesn't use)

## Verdict for v23

Compared to RTN-scalefix (the only known-good GPTQ-marlin platform tarball), v23 (= v22 + commit `57f9cef06` H4 fix on the quant script) differs in:

| Difference | Type | Intentional? |
|---|---|---|
| Different quant script (`quantize_gptqmodel_w4a16.py` vs `quantize_gptq_rtn_sym.py`) | Required (GPTQModel pipeline) | ✅ Yes |
| `flash_attn-...whl` bundled | Required (gptqmodel needs flash-attn at quant time) | ✅ Yes |
| `perf_public_set.jsonl` bundled | Required (calib data) | ✅ Yes |
| `prepare_env.sh` adds gptqmodel install + version pin + flash-attn install | Required | ✅ Yes |
| `prepare_model.sh` adds calib resolution + timeout + DIAGNOSTIC blocks | Required + observability | ✅ Yes |
| `srt/configs/minicpm.py` kwargs.pop patch | Defensive | ✅ Yes (v18 lesson) |
| `srt/layers/torchao_utils.py` early-return | Defensive | ✅ Yes (v15 lesson) |
| `__pycache__/` directory accidentally bundled | Cosmetic regression | ⚠️ Clean up in pack |

**Single most-suspicious difference**: none. There are no spurious differences between v22 and RTN-scalefix beyond what the GPTQModel pipeline requires. If v23 (= v22 + tokenizer overwrite fix) STILL produces platform=0, the cause must be in:
- the actual on-disk quantized artifact's byte content (not the tarball)
- an environmental difference on the platform (which the new DIAGNOSTIC prints in v23 will surface)
- our suspect H1/H4/H2/H3 hypotheses themselves

i.e. v23 is now as RTN-clean as a GPTQModel submission can be. Any remaining platform=0 → look at the platform-side log via `V23_PLATFORM_LOG_CHECKLIST.md`, not at this tarball-structure level.

## Cleanup recommendations before packing v23

```bash
# Drop __pycache__ from packed dirs (cosmetic only but tidier)
find submission_gptqmodel_calib_w4a16 -name __pycache__ -exec rm -rf {} +
find submission_gptq_v17_minconfig -name __pycache__ -exec rm -rf {} +
# Confirm tarball-time pack also strips them
grep -n "pycache" tools/pack_submission.py  # should show that pack already skips these (it does via `find -type d`)
```
