# v24 submission — perf-extended fork of v23

> **PRECONDITION**: do NOT submit this variant until v23 platform result
> shows `acc_ori >= 30`. v24 ships chunked-prefill 65K, which v23 was
> deliberately submitted WITHOUT to isolate the H1+H4 fix attribution.
> Submitting v24 before v23 has come back will confound the result.

## What this variant is

A symlink-based fork of `submission_gptqmodel_calib_w4a16/` (the v23
source) with **one and only one functional difference**: SGLang launches
with the chunked-prefill 65K config measured at +83% throughput / -63%
TTFT on `config/chunked-prefill-tuned`.

```
submission_gptqmodel_calib_w4a16_v24/
├── README_SUBMISSION.md           ← this file (tracked, v24-specific)
├── prepare_env.sh                 ← tracked, v24-specific (chunked-prefill 65K)
├── prepare_model.sh               → symlink to v23
├── quantize_gptqmodel_w4a16.py    → symlink to v23 (same hardened fix_qzeros + H4 copy_runtime_assets)
├── perf_public_set.jsonl          → symlink to v23 (same round-robin reordering)
├── flash_attn-...whl              → symlink to v23
└── sglang                         → symlink to v23 (same bundled fork)
```

Pack-time `cp -L` dereferences every symlink, so the produced tarball
contains the actual file content — pack output is byte-identical to
what packing the v23 directory would produce **except** for
prepare_env.sh.

## The one-line diff vs v23

```diff
- export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --dtype float16"
+ export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 65536 --max-prefill-tokens 65536 --mem-fraction-static 0.80 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --dtype float16"
```

Three flag changes, all inherited from `config/chunked-prefill-tuned`:
- `--chunked-prefill-size 65536` (was 8192)
- `--max-prefill-tokens 65536` (was unspecified — SGLang default 16384 silently capped previous setting)
- `--mem-fraction-static 0.80` (was unspecified — SGLang auto-calc can go negative under large chunks)

## Expected platform result

If v23 came back with `acc_ori = X`, v24 should land at:

- `acc_ori ≈ X` (chunked-prefill is a perf flag, not a quality flag —
  Exp H confirmed local 49.00 → 48.78 = noise)
- `benchmark_duration` drops materially: S1 50%-60% faster, S8 / Smax
  similarly. Source-branch measurement on `config/chunked-prefill-tuned`:
  267→489 tok/s = +83% throughput; 34049ms→12529ms TTFT = -63%.

If `acc_ori` regresses materially from v23 (>2pp), the chunked-prefill
config interacts badly with this artifact (was NOT seen in Exp H locally;
would indicate a platform-specific issue).

## How to pack

```bash
cd /root/soar/sglang   # or wherever the repo is
python3 tools/pack_submission.py --variant submission_gptqmodel_calib_w4a16_v24 \
    --check-only
# expect: [check] /path/...v24: OK (no problems)

python3 tools/pack_submission.py --variant submission_gptqmodel_calib_w4a16_v24 \
    --suffix _v24 --output-dir .
# produces: soar_gptqmodel_calib_w4a16_v24_submission_<date>_v24.tar.gz
```

## Op-fusion: optional, deferred to v25

`perf/op-fusion` adds another ~10-15% decode speedup via RMSNorm+residual
fusion + dropping the FP32 upcast around RoPE. It is bit-exact in
CPU-mock but has not been GPU-validated. To layer op-fusion into v25:

1. GPU-validate first (in a separate worktree, not on v24 source):
   ```bash
   git worktree add /tmp/fusion_test origin/perf/op-fusion
   cd /tmp/fusion_test
   git apply /root/soar/sglang/experiments/op_fusion_verify.patch
   MINICPM_FUSION_VERIFY=1 bash run_sala.sh   # check norm_diff < 1e-3 for ~50 layers
   ```
2. If validation passes, create a new submission_*_v25 variant dir that:
   - Inherits from v24 via symlinks the same way v24 inherits from v23
   - Has a non-symlinked sglang/python/sglang/srt/models/minicpm.py
     replaced with op-fusion's version

Not done in this variant — keep v24 = perf flags only, single change for
attribution.

## Source of truth

The Python/shell logic in v24 is symlinked to `submission_gptqmodel_calib_w4a16/`.
Any future bug fix to `quantize_gptqmodel_w4a16.py`, `prepare_model.sh`,
or the bundled `sglang/` automatically flows to v24 via the symlink. The
only two v24-specific files (`README_SUBMISSION.md` + `prepare_env.sh`)
should NOT be modified without a corresponding update to v23 to keep
the diff minimal.
