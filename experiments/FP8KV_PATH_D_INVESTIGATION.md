# FP8 KV Cache on SALA — Path D investigation, dead-end as of 2026-05-24

> Written after the 2026-05-24 evening session burned a smoke slot on chunk32k_fp8kv
> Path D. Records what failed, why, what to instrument next session, and an
> alternative (Path E) that side-steps the scale-plumbing question.

## TL;DR

SOAR official toolkit (https://soar.openbmb.cn/toolkit) lists **W4A16 + Marlin + FP8 KV
Cache** as 路径一 (the canonical optimization stack). On SALA's hybrid backend, the
`--kv-cache-dtype fp8_e5m2` flag does allocate the KV pool as fp8 — but FlashAttention
rejects fp8 K/V at CUDA graph capture (`RuntimeError: FlashAttention only support fp16
and bf16 data type`). Two patches have been attempted; both insufficient.

- **Path B** (commit `e7a962905`, abandoned): inline scale=1.0 fallback inside
  `minicpm_backend.py` forward_extend/forward_decode. Didn't fix the real problem
  (no quant_method on RadixAttention so layer.k_scale stayed None).
- **Path D** (commit `136c4ec3c`, smoke-tested + crashed 2026-05-24 20:21):
  3-line patch on `python/sglang/srt/layers/quantization/gptq.py:364` —
  `GPTQMarlinConfig.get_quant_method` returns `BaseKVCacheMethod` for
  `RadixAttention`. Theory: RadixAttention.__init__ would then auto-attach
  `layer.k_scale` via `create_weights`, forward_decode's scale path would activate.
  In practice: same `RuntimeError: FlashAttention only support fp16 and bf16`.

Submitting `submission_*_chunk32k_fp8kv` to the platform **WILL crash** until SALA-side
work lands. Currently dropped from `scripts/gpu_smoke_remaining.sh` queue.

## Architectural facts

Verified by reading source 2026-05-24:

| Component | Location | Note |
|---|---|---|
| SALA model | `python/sglang/srt/models/minicpm.py:564` (`MiniCPMSALAForCausalLM`) | NOT loaded via `trust_remote_code`. The HF modeling file is the standalone-HF path. |
| Backend dispatch | `python/sglang/srt/layers/attention/attention_registry.py:244` | Returns `HybridLinearAttnBackend(full_attn, linear_attn)` |
| Dense attn backend | `minicpm_flashinfer` / `minicpm_flashattn` → `MiniCPMSparseBackend` | `python/sglang/srt/layers/attention/minicpm_backend.py:147` |
| Linear (lightning) attn | `SimpleGLAAttnBackend` | No KV cache; linear-recurrent state. |
| RadixAttention | `python/sglang/srt/layers/radix_attention.py:43` | Per-layer module owning `self.k_scale`/`self.v_scale` |
| Quant method attach | `radix_attention.py:85-88` | `quant_method = quant_config.get_quant_method(self, prefix); quant_method.create_weights(self)` |
| KV scale create | `python/sglang/srt/layers/quantization/kv_cache.py:30-44` | `layer.k_scale = Parameter(-1.0)` initial |
| Post-load default | `kv_cache.py:47-67` | Both scales <0 → fallback to `1.0` |
| Decode fp8 branch | `minicpm_backend.py:1148-1155` | `if kv_cache_dtype_str != "auto" and layer.head_dim <= 256:` casts q to fp8, sets `k_descale`/`v_descale` IF `layer.k_scale is not None` |

The toolkit doc explicitly warns: *"Lightning Attention 层使用独立线性注意力状态，优化路径不同"*. The linear (lightning) layers don't go through KV cache at all — only dense layers benefit from FP8 KV.

## Observed Path D crash signature

`submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv` smoke 2026-05-24 20:21:

```
[server_args] kv_cache_dtype='fp8_e5m2', attention_backend='minicpm_flashattn'
[20:21:47] Using KV cache dtype: torch.float8_e5m2     ← pool IS fp8
[20:21:48] Capture cuda graph begin.
...
File "/root/autodl-tmp/zyn/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 723, in capture_one_batch_size
RuntimeError: FlashAttention only support fp16 and bf16 data type
Exception: Capture cuda graph failed: FlashAttention only support fp16 and bf16 data type
[20:21:49] Received sigquit from a child process. It usually means the child failed.
```

Crash during graph capture means forward_decode/forward_extend is invoked on dummy
batches and FA rejects the fp8 K/V. We don't yet know *whether* `layer.k_scale` was
populated, *whether* `k_descale` was passed to the FA call, or *whether* the FA-call
site even forwards descales in the SALA sparse path.

## Hypotheses (ranked)

1. **`process_weights_after_loading` not called for auto-attached BaseKVCacheMethod**
   on RadixAttention. SGLang's loader iterates Linear layers for this hook; if the
   hook isn't invoked for our auto-attached method, k_scale stays at `-1.0` sentinel.
   `layer.k_scale is not None` is True (-1.0 isn't None), so descale path activates,
   but descales are negative → FA rejects or returns garbage.
2. **The FA-call site in `minicpm_backend.py` forward_decode** (after line 1211, not
   read yet) doesn't forward `k_descale`/`v_descale` to the FA invocation — the
   scales are computed but ignored.
3. **Sparse-K path** (`get_topk_for_sparse`, `get_block_table_v3`, lines 1182-1204)
   operates on raw cache tensors that are fp8 → internal kernels don't dequant →
   fp8 tensors leak into FA.
4. **GPTQMarlinConfig.get_quant_method isn't called for RadixAttention** at all —
   maybe SGLang's loader only invokes get_quant_method for `LinearBase` instances,
   never for `RadixAttention`. RadixAttention.__init__ does call it (line 86), so
   this is unlikely, but possible if the model construction path is different for
   SALA's hybrid layout.

## What to instrument next session

Don't try another blind 3-line patch. Add diagnostic prints:

1. **`RadixAttention.__init__`** (`radix_attention.py:85-88`) — for SALA's dense
   layers, log `quant_config.__class__.__name__`,
   `self.quant_method.__class__.__name__ if quant_method else None`. Confirm
   whether `BaseKVCacheMethod` is actually attached for GPTQMarlinConfig with
   Path D applied.
2. **`forward_decode`** (`minicpm_backend.py:1148`) — log `layer.k_scale`,
   `layer.v_scale`, `self.kv_cache_dtype_str`, `layer.head_dim`. Confirm the
   branch is taken with valid (positive) scales.
3. **The actual FA call** (read minicpm_backend.py 1211+) — confirm
   `k_descale`/`v_descale` reach the FA invocation.
4. **`BaseKVCacheMethod.process_weights_after_loading`** (`kv_cache.py:47`) —
   add `print` at function entry. Verify it's called for our auto-attached
   method.

## Path E: side-step the scale plumbing

If instrumentation reveals deep plumbing issues, the pragmatic fallback is to
dequant fp8 → bf16 inside `minicpm_backend.py:1158` after `get_kv_buffer`:

```python
key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
if self.kv_cache_dtype_str.startswith("fp8"):
    key_cache = key_cache.to(torch.bfloat16)
    value_cache = value_cache.to(torch.bfloat16)
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
```

Trade-off: defeats ~50% of fp8 bandwidth win (we dequant immediately after fetching)
but unblocks FA. Useful as proof-of-concept that the SALA hybrid backend can run
under fp8 KV pool at all, before investing more in scale-aware Path D fixes.

## Recommended next steps (in order)

1. Run the instrumented diagnostic smoke. Goal: identify which of the 4 hypotheses
   above is the actual failure.
2. If it's H1 (process_weights_after_loading not called): add a manual call in
   `MiniCPMSparseBackend.__init__` for each RadixAttention layer the runner owns.
3. If it's H2 (FA call doesn't forward descales): patch the FA call site to pass
   them.
4. If H3 or H4: shelf Path D and ship Path E as a working fallback (acc=82-ish,
   throughput improvement reduced but non-zero).
5. **Don't queue chunk32k_fp8kv in any smoke orchestrator until smoke locally
   reaches `acc_ori ≥ 75` for at least 30 samples.** The variant tarball staged
   in `~/OneDrive/soar_submissions/` should NOT be submitted to platform.

## References

- Official toolkit: https://soar.openbmb.cn/toolkit (路径一)
- Path B commit: `e7a962905 fix(smoke): drop chunk32k_fp8kv from default queue — Path B patch insufficient`
- Path D commit: `136c4ec3c fp8kv v2: align with SOAR official toolkit (fp8_e5m2 + Path D patch)`
- Path D crash log: `scripts/logs/local_eval_server_submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_1779625293.log` (search for line "RuntimeError: FlashAttention only support")
- Companion auto-memory: `~/.claude/projects/.../memory/fp8kv_sala_investigation.md`
