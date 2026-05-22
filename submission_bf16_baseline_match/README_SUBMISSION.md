# SOAR BF16 baseline-match submission (v4 of safetynet line)

**Purpose:** disambiguate whether the safetynet v2/v3 failure was caused by
our extra SGLang flags (`--max-prefill-tokens`, `--enable-mixed-chunk`) or
by something else.

## Diff from SOAR baseline

**ZERO functional changes.** This submission is identical to the SOAR-Toolkit's
default `SGLANG_SERVER_ARGS` and an identity passthrough for the model:

```
SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse"
```

`prepare_model.sh` only symlinks (or copies) the model — no transformation.

## Expected outcome

- If submission completes with **score ≈ 19.13** (matches BF16 baseline) → the identity package and official default args are healthy; v2/v3 failure was likely caused by one of the added flags, so add them back one at a time
- If submission fails the SAME way as v2/v3 → the problem is NOT the flags; need to investigate further

## Diff from safetynet v3 (which failed)

| | safetynet v3 | **this (v4)** |
|---|---|---|
| `--chunked-prefill-size` | 32768 | 8192 (baseline default) |
| `--max-prefill-tokens 32768` | included | REMOVED |
| `--enable-mixed-chunk` | included | REMOVED |
| Bundled SGLang | no | no (same) |

## Tarball

`soar_bf16_baseline_match_submission_20260520_v4.tar.gz` (~2 KB)
