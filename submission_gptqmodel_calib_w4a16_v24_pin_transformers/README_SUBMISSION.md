# v24_pin_transformers — fork of v23 forcing transformers==4.57.1

> **PRECONDITION**: only submit if v23 platform returns `acc_ori = 0`
> with `[qzeros-fix] OK` in the log AND the `[versions] transformers=X.Y.Z`
> line shows a different version than AutoDL's 4.57.1. If platform's
> transformers matches AutoDL's, this variant tests nothing.

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

Single behavior change: platform's transformers gets force-reinstalled
to 4.57.1 (the version where v21 achieved local acc=49 on AutoDL).

## Why this might matter

`transformers v4.47.0` (PR #33957, 2024-12-05) added `chat_template.jinja`
sidecar file support. AutoDL has `transformers==4.57.1` so it reads the
sidecar correctly. If the platform's transformers is older than 4.47
(quite likely — many production envs lag releases by months), it
silently ignores the sidecar and reads only the inline chat_template
in `tokenizer_config.json`. The H4 fix (commit `57f9cef06`) overwrites
the tokenizer files with base BF16's, which only has inline, so H4
SHOULD address this. But H4 only fires when the platform's
quantize_gptqmodel_w4a16.py runs `copy_runtime_assets`. If that flow
has an issue on the platform that we don't know about, the tokenizer
overwrite may not actually take effect.

This variant addresses H4 from a DIFFERENT angle: even if the tokenizer
overwrite fails, forcing transformers to 4.57.1 means the platform
reads the sidecar just like AutoDL, which gives the same result.

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
    --suffix _v24_pin_transformers --output-dir .
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
| `acc_ori >= 30` | transformers version was the issue (or H4 wasn't taking effect) | Cherry-pick into v23 source as `transformers==4.57.1` pin |
| `acc_ori = 0` with `[versions] transformers=4.57.1` and clean log | Not the version | Move to v24_bits8 or v24_no_dtype_key |
| Pip install fails | 4.57.1 conflicts with platform base env | Try `TRANSFORMERS_PIN=4.46.0` |
