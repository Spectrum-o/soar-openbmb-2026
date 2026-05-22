# v24_pin_transformers — force transformers==4.57.1

> **PRECONDITION**: only submit if v23 platform returns `acc_ori = 0`
> with `[qzeros-fix] OK` in the log AND the `[versions] transformers=X.Y.Z`
> line shows a different version than AutoDL's 4.57.1. If platform's
> transformers matches AutoDL's, this variant tests nothing.
>
> 2026-05-22 update: the platform log for `v24_bits8` showed
> `Transformers version 5.9.0 is used for model type minicpm_sala` and warned
> that MiniCPM-SALA may be incompatible with transformers >= 5.0. This variant
> is now the direct test of that warning: force the platform back to the repo's
> pinned `transformers==4.57.1`.

## The diff vs v23

In `prepare_env.sh`, replace the gptqmodel install gate:

```diff
- if python_pkg_exact_version gptqmodel "${GPTQMODEL_PIN}" && \
-    python_pkg_min_version transformers 4.45 && \
+ if python_pkg_exact_version gptqmodel "${GPTQMODEL_PIN}" && \
+    python_pkg_exact_version transformers "${TRANSFORMERS_PIN:-4.57.1}" && \
     python_pkg_present accelerate && python_pkg_present ninja; then
      ...
  else
-    install_with_cn_fallbacks "gptqmodel==${GPTQMODEL_PIN}" \
-        "transformers>=4.45" accelerate ninja
+    install_with_cn_fallbacks --force-reinstall "gptqmodel==${GPTQMODEL_PIN}" \
+        "transformers==${TRANSFORMERS_PIN:-4.57.1}" accelerate ninja
  fi
```

Single behavior change: platform's transformers gets force-reinstalled to
4.57.1, matching the local/reference stack and this repo's `python/pyproject.toml`.

## Why this might matter

There are two separate reasons this can matter:

1. MiniCPM-SALA has known version sensitivity. The local stack and this repo
   pin `transformers==4.57.1`; the platform currently shows 5.9.0 and SGLang
   itself warns about potential MiniCPM-SALA/RoPE incompatibility under
   transformers >=5.0.
2. GPTQModel touches tokenizer/config during save. Even though the H4 fix now
   overwrites tokenizer files from the BF16 source model, using the same
   transformers version as local removes a large parsing/rendering variable
   around `chat_template`, config aliases, and custom MiniCPM-SALA code.

## Variant dir structure

```
submission_gptqmodel_calib_w4a16_v24_pin_transformers/
├── README_SUBMISSION.md     ← tracked, this file
├── prepare_env.sh           ← tracked, force-reinstall transformers
├── prepare_model.sh         → symlink to v23
├── quantize_gptqmodel_w4a16.py  → symlink to v23
├── perf_public_set.jsonl    → symlink to v23
├── flash_attn-...whl        → symlink to v23
└── sglang                   → symlink to v23
```

## Pack + submit

```bash
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v24_pin_transformers \
    --check-only

python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v24_pin_transformers \
    --output soar_gptqmodel_calib_w4a16_submission_20260522_v24_pin_transformers.tar.gz
```

## Risk: transformers + gptqmodel compatibility

gptqmodel 7.0.0 requires `transformers >= 4.45`. 4.57.1 satisfies this.
But if there's a hidden API incompatibility we don't know about, this
variant could crash early at quant time. Watch for ImportError or
AttributeError in the platform log's first 5 minutes.

If gptqmodel fails on 4.57.1 specifically, try
`TRANSFORMERS_PIN=4.46.0` env override (the version just BEFORE sidecar
support — would also force inline-only reading, similar effect to
forcing newer).

## Expected outcomes

| Platform result | What it means | Next action |
|---|---|---|
| `acc_ori >= 30` | transformers 5.9.0 was a major part of the failure | Keep the exact pin in the main GPTQModel submission path |
| `acc_ori = 0` with `[versions] transformers=4.57.1` and clean log | Not primarily the transformers version | Focus on GPTQModel artifact format, full-linear vs MLP-only, or calibration |
| Pip install fails | 4.57.1 conflicts with platform base env | Try `TRANSFORMERS_PIN=4.46.0` |
