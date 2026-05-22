# SOAR BF16 + Op-Fusion Submission (stretch variant)

> ⚠️ **Not for blind submission.** Requires GPU validation step first.
> Prepared 2026-05-22 on branch `parallel/non-gptqmodel-paths`.

## What this is

A stretch variant beyond `soar_bf16_chunk32k_safetynet_submission_20260520_v3.tar.gz`:

| Knob | v3 safetynet | **this variant** |
|---|---|---|
| Quantization | None (identity BF16) | None (identity BF16) |
| `--chunked-prefill-size` | 32768 | **65536** |
| `--max-prefill-tokens` | 32768 | **65536** |
| `--mem-fraction-static` | (default) | **0.80** |
| `--enable-mixed-chunk` | yes | yes |
| **RMSNorm+residual fusion** | no | **yes** (笔记 04) |
| **Drop FP32 upcast around RoPE** | no | **yes** (笔记 04) |
| Bundled SGLang source | no | no (overlay only) |

## Expected impact

- v3 (chunked-prefill 32K + mixed-chunk): predicted ~25-30 final_score (per `experiments/BF16_SAFETYNET_AUDIT_20260522.md`)
- This variant adds chunked-prefill 65K + op-fusion → expected **30-37 final_score**
- Diminishing returns past 32K → 65K, primary further win comes from op-fusion saving ~64 kernel launches per decode step (32 layers × 2 norms)

## How op-fusion is delivered (not as bundled SGLang)

The v2 safetynet (`soar_bf16_chunk32k_safetynet_submission_20260520_v2.tar.gz`)
bundled the entire `sglang/python/` directory and **crashed in 13s** on the
platform (SUBMISSIONS.md row 104). v3 fixed it by not bundling. This variant
**continues the no-bundle approach**, but ships ONLY the patched
`minicpm.py` as an overlay at `sglang_overlay/minicpm.py`.

At `prepare_env.sh` time:
1. Locate platform's installed `sglang/srt/models/minicpm.py`
2. Back it up to `minicpm.py.bak.op_fusion`
3. Copy our overlay over it
4. Run `import sglang.srt.models.minicpm` smoke test; if it fails, restore backup and FATAL

This minimizes the v2-style "bundled SGLang conflicts with base env" risk surface.

## Op-fusion correctness provenance

- Code from `origin/perf/op-fusion` branch, commit `79b0c20f4` "perf: fuse RMSNorm+residual and drop FP32 upcast around RoPE"
- **Pure-torch equivalence verified** by commit `87c22ec88` (`test/srt/...`) — CPU-side bit-exact equivalence to the unfused implementation
- **GPU runtime correctness NOT verified on the platform's exact env** — this is the remaining risk

## ⚠ Required validation before submitting

Before submitting this tarball, on the AutoDL GPU node:

```bash
# 1. Apply the verify patch to the overlay file
cd /root/autodl-tmp/zyn/sglang
git fetch origin
git apply experiments/op_fusion_verify.patch    # see header comment

# 2. Launch with MINICPM_FUSION_VERIFY=1 and any test prompt
MINICPM_FUSION_VERIFY=1 bash run_sala.sh &
SERVER_PID=$!
sleep 60
curl -X POST http://127.0.0.1:31111/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"minicpm-sala","messages":[{"role":"user","content":"hello"}]}'

# 3. Inspect stderr — every layer should print:
#    [fusion-verify L0 norm1] norm_diff=0.000e+00 res_diff=0.000e+00
# If any diff exceeds 1e-3 in bf16, the GPU fusion path has divergence
# from the unfused reference. DO NOT submit if so.

kill $SERVER_PID
```

If `MINICPM_FUSION_VERIFY=1` shows all diffs within tolerance → safe to pack
and submit:

```bash
tar -czf soar_bf16_op_fusion_$(date +%Y%m%d_%H%M).tar.gz \
    -C submission_bf16_op_fusion \
    README_SUBMISSION.md prepare_env.sh prepare_model.sh sglang_overlay
```

(Note: the `tar` command above is intentional — `pack_submission.py` does
not currently know about the `sglang_overlay/` directory pattern. Use the
manual tar command above.)

## Why NOT to ship without validation

The op-fusion path uses sglang's fused `add_rmsnorm` CUDA kernel. On the
specific combination of:
- Platform's torch 2.9.1+cu128
- Platform's CUDA driver
- SALA's hybrid sparse/linear forward pass
- Long context inputs (the actual SOAR eval has 30K+ token prompts)

…the fused kernel could have numerical drift that the CPU test doesn't catch.
A 5h slot lost to a fusion bug is much more expensive than a 30-min local
validation.

## If GPU validation passes

This variant should reach **30-37 final_score**, putting you confidently in
the **rank 18-20 band**.

Stack ordering recommendation:
1. Submit v3 safetynet first (already-packed, lowest risk) — establishes ~25-30 floor
2. After v3 result is back AND positive, GPU-validate this variant
3. Submit this variant for the additional +5-7 final_score
