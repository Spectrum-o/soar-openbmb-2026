# FP8 KV Cache on MiniCPM-SALA: Investigation & Fix Paths

Diagnosed 2026-05-14. The user attempted `--kv-cache-dtype fp8_e4m3` with two
attention backends and got two different failures. This document explains the
root cause and lists concrete fix paths to try on GPU.

## Symptoms

### Attempt 1: with `--attention-backend minicpm_flashinfer`

```
File ".../flashinfer/jit/attention/modules.py", line 978, in gen_batch_prefill_module
    assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"
AssertionError: fp8 tensor core is not supported in fa2 backend
```

flashinfer's FA2 prefill path explicitly refuses FP8 KV cache. Dead end with
this backend.

### Attempt 2: with `--attention-backend minicpm_flashattn`

```
Exception: Capture cuda graph failed: FlashAttention only support fp16 and bf16 data type
```

This one made it past flashinfer's gate but died during CUDA graph capture of
the decode path.

## Root cause

In `python/sglang/srt/layers/attention/minicpm_backend.py:920-935`:

```python
if (
    self.kv_cache_dtype_str != "auto"
    and layer.head_dim <= 256
    and self.fa_impl_ver != 4
):
    if layer.k_scale is not None:
        descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        k_descale = layer.k_scale.expand(descale_shape)
        v_descale = layer.v_scale.expand(descale_shape)
    q = q.to(self.kv_cache_dtype)   # <-- always converts Q to fp8
```

The condition for entering FP8 path is just `kv_cache_dtype_str != "auto"`.
But the `k_descale` / `v_descale` setup is gated by `layer.k_scale is not None`.

When does `layer.k_scale` get set? Walk back to
`python/sglang/srt/layers/radix_attention.py:79-88`:

```python
self.k_scale = None
...
if quant_config is not None:
    self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
if self.quant_method is not None:
    self.quant_method.create_weights(self)   # <-- this is what makes k_scale exist
```

So `k_scale` is created only when the model is loaded with a `quant_config`
that provides a KV cache quant method. The user passed only `--kv-cache-dtype`,
not `--quantization`, so `quant_config = None`, `quant_method = None`, and
`k_scale` stays None.

Result: `q` is cast to fp8 but no descale tensor is provided to the kernel.
FlashAttention sees fp8 inputs without scales -> rejects them.

`python/sglang/srt/layers/quantization/kv_cache.py:47-79` confirms this is
the expected channel: `BaseKVCacheMethod.create_weights` adds `k_scale` /
`v_scale` parameters initialized to -1.0, and `process_weights_after_loading`
defaults them to 1.0 when the checkpoint doesn't supply real values.

So **the path to make FP8 KV cache work on SALA without modifying code is to
attach a KV cache quant method via the model's quant_config.**

## Fix paths, ranked

### Path A: pass `--quantization fp8` together with `--kv-cache-dtype fp8_*`

This is what every other model in sglang does to enable FP8 KV cache. The
`fp8` quant_config attaches `BaseKVCacheMethod` to every RadixAttention layer,
which adds the scale parameters. Since the base SALA checkpoint has no FP8
weights, sglang should still try to quantize them on-the-fly OR the user must
combine with a different non-fp8 weight quant.

Try this command:

```bash
python3 -m sglang.launch_server \
    --model /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
    --trust-remote-code --disable-radix-cache \
    --attention-backend minicpm_flashattn \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 65536 --max-prefill-tokens 65536 \
    --max-running-requests 32 --mem-fraction-static 0.80 \
    --skip-server-warmup --port 31111 --dense-as-sparse
```

Possible outcomes:
1. **It works**: weights also get fp8-quantized (effectively dynamic per-tensor
   FP8), KV cache stored as fp8, default scale=1.0 used. Compose with this as
   a baseline against W4A16 later.
2. **It fails with "model already quantized" or similar**: model config may
   refuse fp8 weight quant. Move to Path B.
3. **Numerical garbage in outputs**: default scale=1.0 is too crude for SALA.
   Move to Path C.

### Path B: minimal code patch to allow scale=1.0 when k_scale is None

Edit `python/sglang/srt/layers/attention/minicpm_backend.py` lines 924-935
and 1145-1155 to fall back to scale=1.0 when `layer.k_scale is None`:

```python
if (
    self.kv_cache_dtype_str != "auto"
    and layer.head_dim <= 256
    and self.fa_impl_ver != 4
):
    if layer.k_scale is not None:
        descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        k_descale = layer.k_scale.expand(descale_shape)
        v_descale = layer.v_scale.expand(descale_shape)
    else:
        # NEW: default to 1.0 when running fp8 KV cache without a quant config.
        descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        k_descale = torch.ones(descale_shape, dtype=torch.float32, device=q.device)
        v_descale = torch.ones(descale_shape, dtype=torch.float32, device=q.device)
    q = q.to(self.kv_cache_dtype)
    ...
```

Two places to patch (forward_extend and forward_decode). This is the cleanest
"just make it work" fix and matches what `process_weights_after_loading` does
when no scales are in the checkpoint.

### Path C: calibrate KV cache scales properly via llm-compressor

If Path A or B run but produce garbage outputs, the scales need real
calibration. llm-compressor has a `KVCacheModifier` that, given a calibration
dataset, computes per-tensor or per-channel k/v scales and saves them into
the checkpoint.

Sketch:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import KVCacheScaleType, GPTQModifier

recipe = [
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    # Add KV cache calibration. The exact API name depends on llmcompressor
    # version - check llmcompressor.modifiers.quantization at runtime.
]
oneshot(model=..., dataset=..., recipe=recipe, output_dir=...)
```

After this, the produced model has `k_scale`/`v_scale` tensors per layer that
sglang loads automatically via BaseKVCacheMethod. Stack this with W4A16 for
maximum compression.

### Path D: stop trying

The user's workload is heavily prefill-bound (TTFT 70% of E2E). KV cache
quantization mainly helps decode. After the chunked-prefill win (TTFT 34s ->
12.5s), prefill is still the bottleneck. **FP8 KV cache has lower expected
ROI than W4A16** (which also helps prefill via reduced GEMM bandwidth). Skip
FP8 KV cache until W4A16 is shipped and then revisit.

## Recommended sequence

1. **Now**: ship W4A16 (branch `quant/w4a16`). Biggest win for this workload.
2. **After W4A16 lands**: try Path A. 5 minute test, no code changes.
3. **If Path A works**: bench against W4A16-alone to see if FP8 KV adds value.
   - Likely small win on decode latency, modest on memory budget.
4. **If Path A fails or numerical**: Path B (10-line code patch) on a branch.
5. **If you have GPU time left and want the gold standard**: Path C, but
   only after evidence that crude FP8 KV cache helps perf.

## Files referenced

- `python/sglang/srt/layers/attention/minicpm_backend.py:920-935, 1145-1155`
- `python/sglang/srt/layers/radix_attention.py:43, 70-89`
- `python/sglang/srt/layers/quantization/kv_cache.py:1-82`
- (For Path C) llm-compressor `KVCacheModifier` (API name varies by version,
  check actual install before relying on this)
