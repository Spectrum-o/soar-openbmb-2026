# FP8 KV Cache on SALA — DEAD END (Blackwell SM 12.0 binary missing) 2026-05-25

> Final root cause found 2026-05-25 00:06 after 4 patch iterations
> (Path B/D/E v1/E v2/E v3). The blocker is NOT Python plumbing.
> sgl_kernel's FlashAttention FP8 kernel only ships a Hopper (SM 9.0)
> binary; SOAR's RTX 6000D and AutoDL's RTX PRO 6000 Blackwell are both
> SM 12.0 (Blackwell) — no kernel image. **Do not attempt FP8 KV on this
> hardware again without first recompiling sgl_kernel for Blackwell.**

## Confirmation log line

From `scripts/logs/local_eval_server_*chunk32k_fp8kv_*.log` (smoke 2026-05-25 00:06):

```
[2026-05-25 00:06:52] Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!
[2026-05-25 00:06:52] Using KV cache dtype: torch.float8_e4m3fn
[2026-05-25 00:06:52] Capture cuda graph begin.
CUDA error (/sgl-kernel/build/_deps/repo-flash-attention-src/hopper/flash_fwd_launch_template.h:166): no kernel image is available for execution on the device
```

`/hopper/` in the path is the smoking gun — the only FP8-capable FA backend bundled with our sgl_kernel build is `flash-attention/hopper` (SM 9.0). On SM 12.0 the kernel image is absent → CUDA loader throws "no kernel image is available."

Also noteworthy: SGLang's own startup prints `"no scaling factors provided. Defaulting to scaling factors of 1.0"`. So Path D's effort to attach `BaseKVCacheMethod` to RadixAttention was unnecessary — SGLang already falls back to 1.0 internally. We were patching the wrong layer for 3 hours.

## Patch trail (all reverted on 2026-05-25 ~00:10)

| Patch | What | Outcome |
|---|---|---|
| Path B (e7a962905) | inline scale=1.0 in minicpm_backend.py | abandoned before this session |
| Path D (136c4ec3c) | GPTQMarlinConfig.get_quant_method → RadixAttention → BaseKVCacheMethod | redundant; SGLang has built-in fallback |
| Path E v1 (this session, attempted in scripts/gpu_smoke_phase3.sh) | dequant `key_cache/value_cache` after `get_kv_buffer` | wrong call site; sparse path doesn't go through get_kv_buffer-based decode |
| Path E v2 (manual, minicpm_sparse_utils.py:514) | dequant `q/k/k2` in `compressed_attention` before `infllmv2_attn_stage1` | fixed sparse-stage-1 crash; revealed the real FA call further down |
| Path E v3 (manual, minicpm_backend.py:1148+) | force descale `torch.ones((bs, tp_k_head_num // 2))` of fresh float32 | wrong shape; FA wants different `num_heads_k` |
| Path E v4 (manual, descale shape = `2 * bs`) | match SALA's sparse head_group double | hit CUDA "no kernel image" at flash_fwd_launch_template.h:166 — Blackwell SM 12.0 has no FP8 FA kernel binary |
| fp8_e5m2 → fp8_e4m3 (variant prepare_env.sh) | toolkit says e5m2; we tried e4m3 to satisfy SGLang's earlier "only supports fp16, bf16, fp8_e4m3" check | accepted by Python plumbing; still blocked by Blackwell binary |

All reverted via `git checkout --`. Tree clean.

## Why the official toolkit recommends this anyway

`https://soar.openbmb.cn/toolkit` 路径一 lists W4A16 + Marlin + FP8 KV. That recommendation assumes:

1. Hopper GPU (e.g. H100, H800, H20, RTX 6000 Ada) — has FP8 FA kernel binaries
2. Standard transformer attention path (not SALA's hybrid InfLLMv2 + lightning split)

SOAR platform's RTX 6000D is **Blackwell SM 12.0**, not Hopper. The kernel that would dequant fp8 K/V doesn't exist for our hardware. Toolkit doc doesn't call this out — probably written for general guidance, not SOAR-specific hardware.

The toolkit's hint *"Lightning Attention 层使用独立线性注意力状态，优化路径不同"* is real but secondary — that's a code-path concern. The blocker is one level deeper at the binary level.

## What would actually unblock FP8 KV on Blackwell

1. **Wait for upstream sgl_kernel Blackwell FP8 FA** — flash-attention 2.x added Hopper FP8 in v2.8, Blackwell support depends on FlashAttention release cadence. Track `Dao-AILab/flash-attention` issues.
2. **Compile sgl_kernel from source with SM 12.0 target + Blackwell FP8 PTX** — non-trivial; the sgl_kernel cmake doesn't auto-target arches above SM 9.0 for FP8 paths. Several day's work + may hit further ABI breakage.
3. **Bypass FA entirely for fp8 K/V** — write a Python-level dequant path that converts fp8 KV cache to bf16 immediately after `get_kv_buffer`, then calls FA with bf16 K/V. Loses ALL bandwidth benefit. Equivalent to bf16 KV — pointless.

None of these are tonight-work. **The cost-benefit on SOAR's specific hardware says: don't pursue FP8 KV.**

## What to do instead

Throughput optimization paths that don't depend on Blackwell-FP8 FA:

- **W4A16 Marlin Linear quant** — already shipped (chunk32k_safe got platform `acc_ori=80.31, final_score=22.9` on 2026-05-25). The Marlin GEMM kernel IS Blackwell-compatible (SM 9.0+ PTX).
- **chunk32k_opfusion REAL** — local smoke 82.67 with overlay actually applied (this session E1). Worth submitting to platform; expected small throughput win from RMSNorm/RoPE fusion.
- **Full-attn (q/k/v/o) W4A16** — high-risk smoke. E3 attempt this session crashed at `validate_gptq_wrapper`'s forbidden-strings check (the MLP-only quantize script proactively rejects any layer_modules containing `self_attn`). To run: skip/relax that validator + handle lightning layers' missing self_attn modules. ~1 day engineering.
- **Speculative decoding** — orthogonal axis, not yet tried. SGLang has built-in speculative; could give 1.5-2x decode speedup.
- **Marlin tile/warp tuning per RTX 6000D** — toolkit 路径一 step 4. Would need profiling + sgl-kernel rebuild but the binary IS available.

## File:line citations (don't waste time re-reading)

- `python/sglang/srt/layers/attention/minicpm_backend.py:1148-1155` — dense forward_decode fp8 dispatch + descale prep
- `python/sglang/srt/layers/attention/minicpm_backend.py:1196-1204` — head_group reshape (`// 2`) for sparse path
- `python/sglang/srt/layers/attention/minicpm_backend.py:1244-1248` — FA call site
- `python/sglang/srt/layers/attention/minicpm_attention_kernels.py:139` — sgl_kernel flash_attn_with_kvcache wrapper
- `sglang_minicpm_sala_env/lib/.../sgl_kernel/flash_attn.py:223` — Python entry; check at C++ level
- `python/sglang/srt/layers/attention/minicpm_sparse_utils.py:514` — sparse stage-1 CUDA call (`infllmv2_attn_stage1`)
- `python/sglang/srt/layers/quantization/kv_cache.py:30-79` — BaseKVCacheMethod (create_weights, process_weights_after_loading)
- `python/sglang/srt/layers/quantization/gptq.py:364` — GPTQMarlinConfig.get_quant_method (Path D injection point)
- `python/sglang/srt/models/minicpm.py:155` — SALA RadixAttention construction
- `python/sglang/srt/layers/attention/attention_registry.py:213-244` — HybridLinearAttnBackend dispatch

## Bottom line

**FP8 KV cache is a hardware-binary dead end on SOAR's Blackwell GPU as of 2026-05-25.** Stop iterating on Path D/E variants. Reallocate budget to op-fusion / full-attn quant / speculative decoding / Marlin tuning.

Commit history of attempts:
- `e7a962905` Path B abandoned (chunk32k_fp8kv dropped from default queue)
- `136c4ec3c` Path D added (insufficient; this doc supersedes)
- `1a96283d7` Phase 3 orchestrator including Path E v1
- 2026-05-25 ~00:10 Path E v2/v3/v4 manual debugging — all reverted, no commit
