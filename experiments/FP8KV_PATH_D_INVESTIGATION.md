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

---

## Path X update — 2026-05-26: platform run confirms dead end, corrects the root cause

The 2026-05-25 conclusion above ("Blackwell SM 12.0 binary missing in sgl_kernel") was a **single-source inference from one CUDA-error string** (`/hopper/flash_fwd_launch_template.h`). That conclusion was wrong about the layer, right about the verdict. Path X (commit `f02334285`, variant `submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer`) tested the hypothesis "rebuild flashinfer with `FLASHINFER_CUDA_ARCH_LIST=12.0f` + switch attention backend to `minicpm_flashinfer`". Platform 2026-05-25 23:46 → 2026-05-26 00:07 (~21 min). Result: **FAILED at CUDA-graph capture**, but with a different error than the local `/hopper/` smoke.

### New platform error (the real root cause)

```
File ".../flashinfer/jit/attention/modules.py", line 978, in gen_batch_prefill_module
    assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"
AssertionError: fp8 tensor core is not supported in fa2 backend
```

Backtrace path: `minicpm_backend.init_forward_metadata_capture_cuda_graph` → `flashinfer_wrapper.begin_forward` → `plan` → `get_batch_prefill_module` → `gen_batch_prefill_module(backend="fa2", ...)`. The assertion fires at flashinfer's **JIT module-generation step**, BEFORE any cubin is loaded — i.e. before SM 12.0 vs. SM 9.0 matters. The `FLASHINFER_CUDA_ARCH_LIST=12.0f` rebuild was irrelevant; the FA2 backend categorically refuses FP8 regardless of GPU arch.

### What this corrects in the prior claim

| Claim (2026-05-25 doc) | What Path X showed |
|---|---|
| "sgl_kernel's FA FP8 kernel only ships Hopper binary" | True for `minicpm_flashattn` route (sgl_kernel FA), but `minicpm_flashinfer` route doesn't even reach the cubin level — flashinfer FA2 has no FP8 path at all |
| "Need to recompile sgl_kernel for Blackwell" | Would fix the `/hopper/` route but not flashinfer; FA2 has no FP8 path on any arch |
| "Wait for upstream Blackwell FP8 FA" | Refined: need flashinfer **FA3** backend for sm_120. FA3 has FP8 support but is currently Hopper-only (`/hopper/` dir in flashinfer too) |

### Both attention backends for SALA reject FP8 on Blackwell

| Backend (`--attention-backend`) | Internal kernel | Result on SOAR (Blackwell SM 12.0) |
|---|---|---|
| `minicpm_flashattn` | sgl_kernel FA (Hopper FA3 dir) | CUDA "no kernel image" — binary missing |
| `minicpm_flashinfer` | flashinfer FA2 (this run) | Python assertion — FP8 not supported in FA2 |

There is no third SALA-compatible attention backend in this codebase. **FP8 KV is structurally infeasible on Blackwell + SALA absent upstream flashinfer FA3 sm_120 support** (not yet present as of 2026-05-26).

### Single open unknown

Does upstream flashinfer FA3 compile for sm_120? If a future flashinfer release lands FA3-on-Blackwell, Path X becomes "switch backend selector to FA3" (one-line change in flashinfer's backend chooser) plus the existing SM120 build steps. Until then, FP8 KV stays parked.

### Memory updates needed

- `feedback_sala_hard_constraints.md` — "FP8 KV needs flashinfer SM 12.0 build" is wrong; the SM120 build doesn't help. Correct rule: "FP8 KV blocked on Blackwell+SALA — neither minicpm_flashattn (sgl_kernel FA hopper-only) nor minicpm_flashinfer (FA2 no FP8) works".
- `reference_kv_quant_path_guide.md` — Path A (SM120 rebuild) tested and failed; the remaining live option is upstream flashinfer FA3 sm_120 work (track flashinfer-ai PRs, not in-tree work).

### Bottom line (revised)

FP8 KV on SALA stays dead. The earlier conclusion was right; the reasoning is now corrected. **Stop spending time on flashinfer rebuilds or sgl_kernel SM 12.0 attempts** — both attention backends used by SALA refuse FP8 for independent reasons. Resume the non-FP8-KV throughput agenda (op-fusion v2 platform retest, full-attn W4A16 retry under fix stack, Marlin tile tuning, speculative decoding).

Platform log artefact: this submission's entrypoint log includes the full traceback; key 12 lines are reproduced above.

---

## Path X v2 update — 2026-05-26: the real fix, prepared as overlay

The Path X v1 platform run gave a precise enough error to point at the actual bug — and on closer reading of flashinfer's own source, the bug is on OUR side, not flashinfer's.

### What flashinfer's gen_batch_prefill_module actually says

From the upstream comment + assertion at `flashinfer/jit/attention/modules.py:978`:

> `fp8_enabled = dtype_q in [torch.float8_e4m3fn, torch.float8_e5m2]`
>
> [comment] *KV-only quantization is independent of the `fp8_enabled` flag, meaning KV-quantized paths can still flow through FA2 even though tensor-core FP8 compute cannot.*

So flashinfer FA2 **does support fp8 KV cache** — it just doesn't support fp8 *tensor cores* (which require Hopper FA3). The blocker only fires when `dtype_q` is fp8. Pass bf16 Q + fp8 KV and FA2 is happy: it dequants KV internally during the attention compute.

### Where SALA pollutes dtype_q with fp8

Two sites in this fork's MiniCPM backend incorrectly propagate the KV cache dtype into Q:

1. **`python/sglang/srt/layers/attention/minicpm_attention_kernels.py:189`**
   ```python
   self.q_data_type = self.kv_cache_dtype          # bug
   ```
   Upstream's stock flashinfer backend at `python/sglang/srt/layers/attention/flashinfer_backend.py:911`:
   ```python
   self.q_data_type = model_runner.dtype           # correct — bf16 from model config
   ```

2. **`python/sglang/srt/layers/attention/minicpm_backend.py:933` and `:1153`**
   ```python
   q = q.to(self.kv_cache_dtype)                   # cast to fp8 before kernel dispatch
   q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
   k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
   ```
   The accompanying comment block says this is for the **sgl_kernel FA3** path which expects `q.dtype == kv.dtype`. For the **flashinfer FA2** path the cast is harmful — flashinfer FA2 wants bf16 Q with fp8 KV, and casting Q to fp8 trips the `fp8_enabled` assertion at module-gen time.

### The overlay fix

`submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer/apply_pathx_q_bf16_overlay.py` — applied during `prepare_env.sh`, idempotent, marker-based:

| Site | Patch |
|---|---|
| `minicpm_attention_kernels.py:189` | `self.q_data_type = self.kv_cache_dtype` → `self.q_data_type = model_runner.dtype` |
| `minicpm_backend.py:933` and `:1153` | wrap the 3-line cast block with `if getattr(self, 'attention_kernel_type', None) != 'flashinfer':` |

The sgl_kernel FA3 path (`minicpm_flashattn`) is untouched — it still gets the q→fp8 cast it expects. Only `minicpm_flashinfer` skips the cast and passes bf16 Q + fp8 KV to flashinfer.

### Status

- Overlay script written + locally smoke-tested on file copies (correct text replacement at both sites, idempotent re-run is a no-op).
- `bash scripts/full_preflight.sh --variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer` → 14/14 hard constraints PASS, latent-assertion lint OK, pack-check OK.
- **AutoDL smoke required next**: confirm overlay applies on the actual bundled sglang, server starts past CUDA-graph capture, 5-sample eval gives acc_ori > 70.
- **Only then platform submit** — burning a 5h slot before AutoDL smoke is the failure mode the SUBMISSIONS.md hard-constraints memory warns against.

### Caveat — what could still go wrong

Path X v2 fixes the JIT-time assertion. It does NOT prove FP8 KV will work end-to-end. Open risks:

- flashinfer FA2's internal KV-dequant kernel may have its own dtype assumptions that surface only at forward-time, not plan-time.
- Even if FA2 runs, accuracy may degrade beyond the 80-gate (no scale factors → defaults to 1.0; SALA's `scale_emb=12` may push KV magnitudes into fp8 saturation, especially for fp8_e5m2's ±57344 range).
- SALA-specific paths (sparse stage-1 via `infllmv2_attn_stage1`) are NOT touched by the overlay. They read `q.dtype` for branching — if they assume q matches kv (fp8) they'll mis-dispatch. The existing memory `feedback_sala_hard_constraints.md` mentions a sparse-utils Q-dequant patch (`minicpm_sparse_utils.py:514`) that was tested on AutoDL but never landed in the tree. Path X v2 may need to be paired with that patch.

If Path X v2 also fails, the next iteration would target the sparse path. If it succeeds, the source patch should be promoted out of overlay form (per `decide-angle` discussion — overlay-first, source-after-platform-pass).
