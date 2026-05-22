# BF16 safetynet audit (2026-05-22 evening)

> Written by Claude on `parallel/non-gptqmodel-paths` branch while user is away.
> Purpose: lay out the state of the BF16 path so user can submit confidently when back.

## Existing tarballs (repo root)

| Tarball | Size | Built | Status | What it does |
|---|---|---|---|---|
| `soar_bf16_chunk32k_safetynet_submission_20260519.tar.gz` | 3.2 MB | 5/19 | unknown | v1 — included bundled SGLang |
| `soar_bf16_chunk32k_safetynet_submission_20260520_v2.tar.gz` | 3.2 MB | 5/20 | ⛔ crashed 13s | v2 — bundled SGLang incompatible with platform base env |
| **`soar_bf16_chunk32k_safetynet_submission_20260520_v3.tar.gz`** | **2.0 KB** | 5/20 | 📦 **never submitted** | **v3 — NO bundled SGLang. The actual scoring candidate.** |
| `soar_bf16_baseline_match_submission_20260520_v4.tar.gz` | 1.6 KB | 5/20 | 📦 never submitted | v4 — identity baseline (would reproduce 19.13). Diagnostic only |

## v3 content verification

Extracted and inspected:

```
prepare_env.sh                    # exports SGLANG_SERVER_ARGS, no install
prepare_model.sh                  # symlink-only, no quantization
README_SUBMISSION.md
```

No bundled SGLang. Uses platform's base env SGLang (the same one BF16 baseline 19.13 uses, so known to support `minicpm_flashinfer`).

`SGLANG_SERVER_ARGS` in v3:
```
--disable-radix-cache
--attention-backend minicpm_flashinfer
--chunked-prefill-size 32768
--max-prefill-tokens 32768
--enable-mixed-chunk
--skip-server-warmup
--dense-as-sparse
```

Diff from BF16 baseline default (19.13 territory):
- `--chunked-prefill-size`: 8192 → **32768** (per `project_chunked_prefill_win` memory: +59% throughput, -60% TTFT locally on SALA)
- `--max-prefill-tokens 32768` added (must match chunked size; without it sglang's hidden default 16384 caps the win)
- `--enable-mixed-chunk` added (SARATHI piggyback for decode under prefill)

## Predicted final_score

Per `submission_bf16_chunk32k_safetynet/README_SUBMISSION.md`:
> Expected score: 20-26 final_score

If the local +59% throughput projection scales linearly to the platform's scoring formula, **~30 final_score is plausible**. This matches user's stated rank ~20 target.

## ⚠ Important: do NOT re-pack v3 from the variant dir

`submission_bf16_chunk32k_safetynet/` contains a `sglang/` subdirectory (~3 MB
bundled source). The v3 tarball was created by an out-of-band process that
**excluded** this directory. If anyone runs:

```bash
python3 tools/pack_submission.py --variant submission_bf16_chunk32k_safetynet
```

…they would get the v2-style bundle that already crashed in 13s on the
platform (SUBMISSIONS.md row 104).

**v3 must be treated as read-only.** Keep the existing tarball as the source
of truth. If you need to rebuild from the dir, manually exclude `sglang/`:

```bash
# How v3 was likely built (reconstruct safely)
tar -czf soar_bf16_chunk32k_v3_new.tar.gz \
    -C submission_bf16_chunk32k_safetynet \
    --exclude='sglang' \
    prepare_env.sh prepare_model.sh README_SUBMISSION.md
```

## Recommended submission order

**When 1849 platform result comes back, regardless of outcome:**

1. **Submit v3 first** — `soar_bf16_chunk32k_safetynet_submission_20260520_v3.tar.gz`. This is the highest-EV move toward the rank-20 goal. Expected final_score 22-30.

2. **If v3 returns 0 or fails** — submit v4 (`soar_bf16_baseline_match_submission_20260520_v4.tar.gz`) to diagnose. v4 is identical to BF16 baseline; if v4 reproduces 19.13, the v3 failure was caused by one of the added flags (chunked-prefill 32K / max-prefill-tokens 32K / mixed-chunk). Peel them back one at a time.

3. **If v3 succeeds with ~25 final_score** — proceed with the op_fusion variant (see HANDOFF doc) to push score higher.

## What this audit does NOT verify

- Whether platform's current base env SGLang (as of 2026-05-22) still supports `--attention-backend minicpm_flashinfer` — the v2 vs v3 distinction relied on the assumption that base env SGLang can serve MiniCPM-SALA. v2's crash was attributed to bundled SGLang conflict, not lack of MiniCPM support, but this assumption was never independently verified.
- Whether mixed-chunk is stable on SALA's sparse attention backend — local microbench showed +59% throughput, but that was on 5/14, code state may have drifted.

These are reasons to submit v4 (identity baseline reproducer) as a diagnostic if v3 fails — it cleanly separates "platform contract works" from "our extra flags work".
