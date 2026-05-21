# v24_no_dtype_key — fork of v23 testing the `dtype` config-key hypothesis

> **PRECONDITION**: do NOT submit until v23 platform result is known.
> v24_no_dtype_key tests a specific Branch-3 hypothesis from
> V24_PLAN.md and is only meaningful if v23 came back with
> `acc_ori = 0` AND `[qzeros-fix] OK` in the platform log (i.e., H1+H4
> fixes worked but acc was still 0 → some OTHER factor broke output).

## The one-line diff vs v23

In `quantize_gptqmodel_w4a16.py`'s `write_sglang_compatible_quant_config`:

```diff
  config["torch_dtype"] = "float16"
- config["dtype"] = "float16"
+ # variant: omit dtype= key; only torch_dtype= is written
  config["quantization_config"] = quant_cfg
```

## Why this might matter

`torch_dtype` is the historical HF transformers config key for the
default tensor dtype. transformers 5.x introduced `dtype` as the new
canonical key, deprecating `torch_dtype`. v21/v22 wrote BOTH (defensive
forward-compat), but:

- The successful RTN-scalefix tarball (platform acc=42) wrote ONLY
  `torch_dtype` (verified via tar -xzO).
- The failed v17/v21/v22 tarballs wrote BOTH.

Older platform transformers (< 4.45) ignores `dtype` cleanly. Newer
platform transformers (>= 4.45 but < some version) may read `dtype`
AS the activation dtype, overriding model behavior. Either way, the
RTN path that worked didn't set it; this variant replicates that.

If this variant succeeds where v23 fails, the `dtype=float16` key was
silently miscomputing something on the platform side. Fix would then
be: stop writing this key in the main quantize script.

## Variant dir structure (symlink-shared with v23)

```
submission_gptqmodel_calib_w4a16_v24_no_dtype_key/
├── README_SUBMISSION.md             ← this file (tracked)
├── quantize_gptqmodel_w4a16.py      ← tracked, ONE-LINE diff vs v23
├── prepare_env.sh                   → symlink to v23
├── prepare_model.sh                 → symlink to v23
├── perf_public_set.jsonl            → symlink to v23
├── flash_attn-...whl                → symlink to v23
└── sglang                           → symlink to v23
```

Pack-time `cp -L` dereferences each symlink; tarball is byte-equivalent
to a v23 tarball except for the one omitted line in quantize_gptqmodel_w4a16.py.

## Pack + submit

```bash
cd /root/soar/sglang
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v24_no_dtype_key \
    --check-only
# expect: OK

python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16_v24_no_dtype_key \
    --suffix _v24_no_dtype_key --output-dir .
```

## Expected outcomes vs decision

| Platform result | What it means | Next action |
|---|---|---|
| `acc_ori >= 30` | `dtype` key WAS the platform=0 cause | Push the fix back into v23's quantize script (canonical fix), ship v25 with chunked-prefill 65K + dtype-key-removed |
| `acc_ori = 0` with bench timing ~ v22 | Not the cause; H1+H4 + `dtype` key all eliminated | Move to v24_pin_transformers or v24_bits8 |
| Crash before bench | Some new issue from removing `dtype` key (e.g., transformers refuses model load) | Revert change; test on AutoDL first |

## Source-of-truth note

This variant's quantize_gptqmodel_w4a16.py is a COPY of v23's (not a
symlink) because the diff is in code. Any future fix to v23's
quantize_gptqmodel_w4a16.py must be manually mirrored here, or
this variant must be re-forked from v23.
