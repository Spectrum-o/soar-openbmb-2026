# FP8KV + W4A16 precision analysis - 2026-05-31

## Status snapshot - 2026-06-01

Current FP8KV state:

- The active package is
  `soar_fp8kv_DENSEQKV_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260601_101730.tar.gz`
  (`md5=3ad3f92828d5595768d6977a4e839dab`). Local audit passes and
  `tools/decide_fp8kv_next.py --current` points to this package.
- The last completed strong-calibration FP8KV run was
  `soar_fp8kv_REALCALIB_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260531_225344.tar.gz`:
  `acc_ori=77.33`, `final_score=0.0`, `S1=650.26`, `S8=1018.16`,
  `Smax=2308.66`, with about 55 minutes of prepare time. This was real
  `300 x 8K x multi-adaptive` GPTQ replay, not the short 4K-tail shortcut.
- Therefore the low FP8KV quality is no longer explained by weak GPTQ replay.
  The remaining clean scalar-scale hypotheses are scale-source mismatch
  (`101730` DENSEQKV) and then post-quantization scale measurement
  (POSTQ_MULTI8K, not yet packed).
- The prepared `214103`, `215412`, and `221735` tarballs are 4K-tail backups
  only. Do not submit them directly after an 8K result, because that would mix
  the precision hypothesis with a weaker calibration-strength variable.

Current interpretation:

- FP8KV is working as a runtime/load path, but it is not currently a scoring
  win. All completed FP8KV variants are around `acc_ori=77.x`, below the
  usable W4A16 record.
- FP8KV also has no visible end-to-end speed win on this benchmark. The best
  non-FP8 W4A16 record had `S1=648.05`; completed FP8KV runs are
  `S1=647.9-650.26`. This is measurement noise, not a strict speedup.
- The most likely reason is that this MiniCPM-SALA serving path is not
  dominated by dense KV cache bandwidth. FP8KV shrinks dense K/V cache storage,
  but it does not speed up Lightning attention, sparse page-table work,
  FlashInfer wrapper planning, MLP GEMMs, sampling, or evaluator overhead. It
  also adds scale handling on KV writes and attention calls.

Calibration status:

- GPTQ calibration for the current FP8KV queue is still the known-good strong
  shape: public `question` prompts, `NUM_CALIB=300`, `MAX_CALIB_LEN=8192`,
  `CALIB_WINDOW_MODE=multi-adaptive`.
- KV-scale calibration is a second stage. It uses existing public
  `question` rows, left/tail truncation, and a model forward/hook pass to
  measure K/V ranges before injecting `self_attn.attn.k_scale/.v_scale`.
  It is not an external hidden calibration dataset.
- KV-scale calibration runs after GPTQ quantization, qzeros fix, and selective
  BF16 overlay. It can have its own memory cost, but the known `222931` and
  `121439` OOM logs died during GPTQModel replay before the KV-scale stage, so
  those failures should not be blamed on KV-scale calibration.

Next decision:

- If `101730` completes low without explicit range/outlier symptoms, stop the
  prepared scalar-source queue and build a new `POSTQ_MULTI8K` package with the
  same `300 x 8K x multi-adaptive` GPTQ replay. Do not use old 4K POSTQ or
  PERHEAD tarballs as the next comparison.
- If `101730` OOMs, classify the log with the `GPU DIAG` blocks: dirty GPU or
  external memory pressure is a platform-state problem; clean GPU that grows to
  the card limit is a package peak-memory problem.
- If a completed E4M3 result shows clear range/saturation symptoms, compare
  against an E5M2 package with comparable calibration strength. Without those
  symptoms, E4M3 remains the better default precision choice.
- If comparable HP224, DENSEQKV, POSTQ, PERHEAD, and range backups all remain
  below the gate, stop the FP8KV queue and spend effort on non-FP8 W4A16 levers
  such as W4 scale recalibration or strictly accuracy-neutral fusion.

## Current platform facts

- `soar_fp8kv_ATTNSCALE_20260530_172346.tar.gz` completed evaluation, so the
  MiniCPM FP8KV runtime/load path is real. Result: `acc_ori=77.24`,
  `final_score=0.0`, `S1=649.48`, `S8=1016.54`, `Smax=2306.0`.
- `soar_fp8kv_REALCALIB_HP224_20260530_222931.tar.gz` failed in
  `prepare_model.sh` with CUDA OOM during GPTQModel replay. This is a
  prepare-time memory issue, not an FP8KV scale-load issue.
- `soar_fp8kv_REALCALIB_TAIL_HP224_OFFLOAD_FIXED_20260531_121439.tar.gz`
  also failed in `prepare_model.sh` during GPTQModel first-layer MLP replay.
  It used `NUM_CALIB=300`, `MAX_CALIB_LEN=8192`, and
  `CALIB_WINDOW_MODE=multi-adaptive`. The OOM log is mixed evidence: current
  Python reported only about 10.75GiB in use while the 83GiB GPU had only about
  252MiB free, so a dirty platform GPU or allocator-state issue is plausible;
  the package was also near the activation-memory edge. It did not reach FP8KV
  scale calibration.
- `soar_fp8kv_REALCALIB_TAIL4K_HP224_OFFLOAD_FIXED_20260531_141006.tar.gz`
  completed low: `acc_ori=77.53`, `final_score=0.0`, `S1=647.62`,
  `S8=1017.5`, `Smax=2305.19`. It prepared in about 12.5 minutes because GPTQ
  replay was reduced to `150 x 4K x tail`. Treat it as an OOM-avoidance/runtime
  signal, not as a strong precision result.
- `soar_fp8kv_REALCALIB_TAIL4K300_HP224_DIAG_OFFLOAD_FIXED_20260531_174343.tar.gz`
  completed low: `acc_ori=77.24`, `final_score=0.0`, `S1=647.9`,
  `S8=1017.28`, `Smax=2306.48`. It prepared cleanly in about 18m51s with
  `300 x 4K x tail`, so the low score is no longer explained by the 141006
  150-row shortcut. However, it is still much lighter than the earlier
  successful 8K multi-adaptive W4A16 GPTQ run. It matches the old ATTNSCALE
  quality and routes the queue to an 8K multi-adaptive HP224 diagnostic before
  DENSEQKV.
- `soar_fp8kv_REALCALIB_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260531_225344.tar.gz`
  completed low: `acc_ori=77.33`, `final_score=0.0`, `S1=650.26`,
  `S8=1018.16`, `Smax=2308.66`. The actual `prepare_model.sh` defaults were
  `NUM_CALIB=300`, `MAX_CALIB_LEN=8192`, and
  `CALIB_WINDOW_MODE=multi-adaptive`; this was not a renamed 4K package. It
  prepared successfully in about 55m14s and reached inference, so the previous
  low results are not mainly explained by shortened 4K-tail GPTQ replay. It
  also shows FP8KV does not produce a measurable end-to-end speed win on the
  current MiniCPM sparse path.
- `soar_fp8kv_DENSEQKV_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260601_101730.tar.gz`
  is the current next package. It preserves the same 300 x 8K multi-adaptive
  GPTQ replay as `225344` and changes the scale-source hypothesis: dense
  MiniCPM q/k/v are restored to BF16 before KV-scale calibration so collected
  scales better match dense runtime K/V. Audit PASS, md5
  `3ad3f92828d5595768d6977a4e839dab`.
- `soar_fp8kv_REALCALIB_E5M2_20260530_225432.tar.gz` completed low:
  `acc_ori=75.4`, `final_score=0.0`, `S1=647.19`, `S8=1015.29`,
  `Smax=2303.06`. It is useful signal, but it is not equivalent to the later
  fixed `TAIL_E5M2_OFFLOAD` backup: audit shows it reads the SOAR `question`
  field, writes the direct `self_attn.attn.k_scale/.v_scale` names, uses
  `fp8_e5m2`, and uses safe-max 224, but it lacks explicit left/tail-window KV
  calibration and disables GPTQ disk offload. Therefore the low 225432 score
  should route to fixed HP224_OFFLOAD rather than invalidate the current
  offload queue.
- `soar_fp8kv_DENSEQKV_HP224_OFFLOAD_20260531_0115.tar.gz` failed before
  inference because `offload_to_disk` was passed as a `GPTQModel.load(...)`
  kwarg and GPTQModel forwarded it into `MiniCPMSALAForCausalLM.__init__`.
  This is a packaging/load-kwarg failure, not a DENSEQKV accuracy signal.
- `soar_fp8kv_POSTQ_PERHEAD_HP224_BRIDGE_20260531_095129.tar.gz` is an old
  per-head tarball submitted after the fixed queue was prepared. Audit fails on
  the same offload-kwarg bug and on the pre-fix fallback path, so any platform
  failure from 095129 should be interpreted as stale-package signal, not as
  evidence against per-head scaling.

## Latest local audit snapshot

The fixed queue was rechecked locally with
`tools/audit_fp8kv_realcalib_package.py`:

- PASS `soar_fp8kv_REALCALIB_TAIL4K300_HP224_DIAG_OFFLOAD_FIXED_20260531_174343.tar.gz`
  (`md5=b1bcfa37d9764490743065cc3af60306`). This package completed low on
  platform, so it is no longer the current next package.
- PASS `soar_fp8kv_REALCALIB_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260531_225344.tar.gz`
  (`md5=40b692635b4743bc64d5d57db8ce2fde`). This package completed low on
  platform, so it is no longer the current next package.
- PASS `soar_fp8kv_DENSEQKV_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260601_101730.tar.gz`
  (`md5=3ad3f92828d5595768d6977a4e839dab`). This is the current next package:
  it keeps `NUM_CALIB=300`, `MAX_CALIB_LEN=8192`, and
  `CALIB_WINDOW_MODE=multi-adaptive`, while restoring dense MiniCPM q/k/v to
  BF16 before KV-scale calibration.
- PASS `soar_fp8kv_DENSEQKV_TAIL4K300_HP224_DIAG_OFFLOAD_FIXED_20260531_214103.tar.gz`
  (`md5=afe85f92dd3f104954c6368cbac82593`). This is prepared but not current
  next; it is a 4K-tail DENSEQKV backup and should not be submitted before the
  stronger `101730` 8K multi-adaptive DENSEQKV package.
- PASS `soar_fp8kv_POSTQ_TAIL4K300_HP224_DIAG_OFFLOAD_FIXED_20260531_215412.tar.gz`
  (`md5=d048431fcb4ff31f8eb756938ba6da0e`). This is prepared but not current
  next. It belongs to the older 4K-tail branch. After the low `225344` result,
  the clean POSTQ hypothesis must be rebuilt as POSTQ_MULTI8K if `101730`
  DENSEQKV_MULTI8K is still low.
- PASS `soar_fp8kv_POSTQ_PERHEAD_TAIL4K300_HP224_BRIDGE_DIAG_OFFLOAD_FIXED_20260531_221735.tar.gz`
  (`md5=5f18e1aae6d7dda4a365ff8dc49a7080`). This is prepared but not current
  next. It belongs to the older 4K-tail branch. Do not submit it directly after
  `101730`; first build a comparable POSTQ_MULTI8K, then PERHEAD_MULTI8K if the
  post-Q scale-source test is also low.
- Historical PASS but not current next:
  `soar_fp8kv_REALCALIB_TAIL_HP224_OFFLOAD_FIXED_20260531_121439.tar.gz`
  (`md5=a1ab2a273390f87a1a5c8c041a252c62`) because the platform run OOMed;
- PASS `soar_fp8kv_DENSEQKV_HP224_OFFLOAD_FIXED_20260531_123154.tar.gz`
  (`md5=f5da669be848b913bc0181ed6180ccf3`);
- PASS `soar_fp8kv_POSTQ_HP224_OFFLOAD_FIXED_20260531_124940.tar.gz`
  (`md5=2c6d679098c89689b1c0940058cc09ef`);
- PASS `soar_fp8kv_POSTQ_PERHEAD_HP224_BRIDGE_FIXED_20260531_124940.tar.gz`
  (`md5=e746cf4d450f6f418bf8699bcc8f3659`);
- PASS `soar_fp8kv_REALCALIB_TAIL_E5M2_OFFLOAD_FIXED_20260531_130420.tar.gz`
  (`md5=ad603f6183956f73bd0a05f029c9ca3f`);
- expected FAIL
  `soar_fp8kv_POSTQ_PERHEAD_HP224_BRIDGE_20260531_095129.tar.gz`
  (`md5=11db5fa8a6630129a47b70e3238623b0`) because it still passes
  `kwargs["offload_to_disk"]` into `GPTQModel.load(...)` and predates the fixed
  per-head fallback dequant.

Related regression command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_fp8kv_realcalib_tools \
  tests.test_fp8kv_runtime_scale_contract \
  tests.test_fp8kv_runtime_scale_support \
  tests.test_fp8kv_per_head_bridge \
  tests.test_kv_calibrate_head_stats \
  tests.test_summarize_fp8kv_head_waste \
  tests.test_minicpm_opfusion_safety \
  tests.test_minicpm_rms_opfusion_math
```

Result: 81 tests OK. This includes explicit GQA/head-group regression coverage
for the per-head bridge output reconstruction, so the V-scale post-multiply is
checked against MiniCPM's interleaved split-head-group layout rather than only
against all-ones tensors. It also checks that a per-head scale tensor with the
wrong KV-head count raises instead of silently broadcasting to the wrong layout.
The decision helper also has regression coverage for `--current`, `--list`, the
new diagnostic OOM classifier, the `174343 -> 225344 -> 101730 -> build
POSTQ_MULTI8K` routing, and the opfusion safety boundary. The package audit also checks
that the KV scale calibration stage appears after GPTQ quantization and the
selective BF16 overlay, so prepare-time GPTQ replay OOMs are not misattributed
to the later KV calibration stage.

## What the official/local docs constrain

`docs/advanced_features/quantized_kv_cache.md` says FP8 KV requires scale
factors, and missing scales default to 1.0, which can hurt accuracy. It also
states that upstream/documented FP8 scale support is only per-tensor scalar
scale. `docs/advanced_features/attention_backend.md` says the MHA FlashInfer
backend supports FP8 KV cache but not FP4 KV cache, so the current
`minicpm_flashinfer + fp8_e4m3` runtime choice is consistent with the official
backend matrix.

That is the main precision wall for a script-only SOAR package:

- We can choose dtype (`fp8_e4m3` vs `fp8_e5m2`).
- We can choose a scalar safe-max and calibration set/window.
- We can choose whether scales are measured from the BF16 base, a BF16-restored
  dense-qkv artifact, or the final W4A16 artifact.
- Upstream JSON `--quantization-param-path` is still scalar-only
  (`Dict[int, Dict[int, float]]` in `weight_utils.py`), so per-head scale is not
  a pure JSON packaging trick.
- The local runtime now has an experimental checkpoint-injected per-head path:
  `kv_cache.py` preserves rank-1 `k_scale_per_head`/`v_scale_per_head`,
  `memory_pool.py` can divide KV by tensor scales before FP8 storage, and
  `minicpm_backend.py` applies the algebraic Q/output bridge around the paged
  FlashInfer call while `minicpm_attention_kernels.py` passes scalar scale
  `1.0` to FlashInfer. This is a real W4A16+FP8KV fusion candidate, but it is
  still a runtime-code path that must be GPU-smoked before platform submission;
  it is not covered by upstream's documented scalar-only contract.

Operational conclusion after the 225344 result:

- Do not shrink GPTQ calibration just because FP8KV scale calibration is added.
  The FP8KV calibration stage runs after GPTQ quantize/save and after the BF16
  overlay, so it does not stack on top of the GPTQ replay activation peak.
- `225344` answered the 8K question: strong 300 x 8K multi-adaptive GPTQ replay
  still stayed at FP8KV low quality. The next package must keep that GPTQ
  strength and change only the scale-source hypothesis (`101730` DENSEQKV).
- Do not submit old 4K POSTQ/PERHEAD packages after a low 8K DENSEQKV result.
  If `101730` is still low without range symptoms, build a comparable
  POSTQ_MULTI8K package instead.

## Why FP8KV did not preserve the expected speed win

FP8KV is not "the previous W4A16 package plus a free faster cache" on this
MiniCPM-SALA path.

- The documented FP8KV benefit is memory capacity/bandwidth, not guaranteed
  lower end-to-end latency. SGLang's quantized-KV docs explicitly warn that
  dequantization or scale handling can negate throughput unless it is fused in
  the selected attention backend.
- The current server uses `--dense-as-sparse`, so dense attention is forced
  through the MiniCPM sparse/FlashInfer paged-wrapper path. FP8KV only shrinks
  dense-layer K/V storage. It does not speed up Lightning attention, sparse
  page-table construction, FlashInfer wrapper planning, MLP/linear GEMMs, token
  sampling, or evaluation overhead.
- The FP8KV packages also add runtime work that the non-FP8 record package does
  not have: FP8 KV write scaling in `memory_pool.py`, scalar scale forwarding
  through `minicpm_attention_kernels.py`, and, for experimental per-head work,
  query/output scale bridges in `minicpm_backend.py`.
- Current platform timings are essentially tied with the non-FP8 record:
  record W4A16 `S1=648.05`, ATTNSCALE FP8KV `S1=649.48`, 4K300 HP224
  `S1=647.9`, and 8K300 HP224 `S1=650.26`. That means FP8KV is not attacking
  the dominant bottleneck of this benchmark; the speed advantage from W4A16
  projection quantization remains, but FP8KV does not add a second visible win.

Local PyTorch reports:

```text
torch.float8_e4m3fn max = 448.0, eps = 0.125
torch.float8_e5m2    max = 57344.0, eps = 0.25
```

So E4M3 has better mantissa precision and is the first choice unless logs or
outputs show explicit range/outlier saturation. E5M2 is a range backup, not the
default accuracy route.

## Runtime scale flow in code

The scale path is live and symmetric:

1. MiniCPM backend writes K/V with the layer scales:
   `python/sglang/srt/layers/attention/minicpm_backend.py`
   calls `token_to_kv_pool.set_kv_buffer(layer, loc, k, v, layer.k_scale,
   layer.v_scale)`.

2. The KV pool divides by the scalar before casting to FP8:
   `python/sglang/srt/mem_cache/memory_pool.py`
   uses `cache_k.div_(k_scale)` and `cache_v.div_(v_scale)` before
   `cache_*.to(self.dtype)`.

3. FlashInfer receives the same scalar for dequant:
   `python/sglang/srt/layers/attention/minicpm_attention_kernels.py`
   passes `layer.k_scale_float` / `layer.v_scale_float` into
   `wrapper.forward(...)` when `kv_data_type` is FP8.

Therefore the current issue is not an obvious write/read scale mismatch. The
remaining likely loss is scale quality: one scalar has to cover the whole dense
K or V tensor for that layer.

## Why per-head scale is not a simple checkpoint package

The remaining scalar limitation is enforced in the attention wrapper boundary,
not only in documentation:

- `python/sglang/srt/layers/radix_attention.py` initializes
  `k_scale/v_scale/k_scale_float/v_scale_float` as scalar attention-layer
  attributes.
- `python/sglang/srt/layers/quantization/kv_cache.py` creates `k_scale` and
  `v_scale` as single `torch.tensor(-1.0)` parameters for compatibility, while
  local helper code now preserves rank-1 checkpoint scales as
  `k_scale_per_head/v_scale_per_head`.
- The optimized FP8 KV write helper documents `k_scale` and `v_scale` as
  optional scalar scales, and the kernel path materializes one inverse scalar
  pointer for K and one for V.

So a safetensors file containing shape `[num_heads]` scales would not be a
drop-in precision fix. The quantization method and MHA KV write path now have
per-head support hooks, and MiniCPM attention now has an experimental
Q/output bridge. It still needs GPU validation and a production package format
that emits per-head scale tensors. That is runtime work, not a SOAR script-only
package.

This is now guarded by a static audit:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_fp8kv_runtime_scale_contract.py \
  --flashinfer-source /tmp/flashinfer_053/flashinfer_python-0.5.3-py3-none-any.whl
```

The audit checks the local SGLang docs/source plus the FlashInfer 0.5.3 wheel
declared by `python/pyproject.toml`. It currently reports:

- SGLang docs say FP8 KV scale is only per-tensor scalar.
- `BaseKVCacheMethod` still keeps scalar `k_scale/v_scale` fallback parameters,
  but the local checkout now preserves optional rank-1 per-head tensors as
  `k_scale_per_head/v_scale_per_head`.
- `--quantization-param-path` schema is `Dict[int, Dict[int, float]]`, so JSON
  cannot express per-head arrays.
- MiniCPM writes KV through `set_kv_buffer(...)` and now chooses per-head scale
  tensors when `k_scale_per_head/v_scale_per_head` exist.
- MiniCPM FlashInfer calls still use scalar `k_scale/v_scale` kwargs; when the
  per-head bridge is active, the wrapper receives scalar `1.0` and Q/output
  carry the per-head factors around it.
- FlashInfer `single_prefill_with_kv_cache` exposes tensor per-head
  `scale_q/scale_k/scale_v`, but `BatchPrefillWithPagedKVCacheWrapper` and
  `BatchDecodeWithPagedKVCacheWrapper` expose scalar `k_scale/v_scale` and pass
  `None` for internal `scale_k/scale_v`.

Conclusion: current MiniCPM serving uses FlashInfer paged wrappers, not the
single-prefill API that has per-head tensor scale parameters. Per-head scale is
therefore not a pure checkpoint-only fix and not a wrapper-argument-only fix for
decode. The local SGLang loader, KV-write side, and MiniCPM Q/output bridge can
now carry per-head scales, but the paged FlashInfer API itself remains scalar.

There is, however, a lower-risk bridge that may avoid changing FlashInfer
itself. For each query head mapped to a KV head:

```text
K_real = K_fp8 * k_scale[kv_head]
V_real = V_fp8 * v_scale[kv_head]

attention(Q, K_real, V_real)
  == postscale_v(attention(Q * k_scale[kv_head], K_fp8, V_fp8),
                 v_scale[kv_head])
```

So a MiniCPM runtime experiment can:

1. store K/V in FP8 using per-head cache-write scales;
2. pass `k_scale/v_scale=1` or no scalar scale to FlashInfer paged wrappers;
3. multiply `q` by the corresponding per-KV-head K scale before the wrapper;
4. multiply the wrapper output by the corresponding per-KV-head V scale after
   the wrapper.

This math is now captured in `tools/fp8kv_per_head_bridge.py`, with CPU tests
that compare the bridge path against direct GQA attention on dequantized K/V.
The local runtime foundation is also partially implemented:

- `python/sglang/srt/layers/quantization/kv_scale_utils.py` handles scalar or
  rank-1 per-head scale loading/finalization and KV cache scale broadcasting.
- `python/sglang/srt/layers/quantization/kv_cache.py` registers scale loaders
  that preserve rank-1 checkpoint scales while keeping scalar fallback values.
- `python/sglang/srt/mem_cache/memory_pool.py::MHATokenToKVPool.set_kv_buffer`
  can divide shaped `[tokens, heads, dim]` or flat `[tokens, heads * dim]` K/V
  tensors by per-head scales before FP8 storage.
- `tests/test_fp8kv_runtime_scale_support.py` covers scalar compatibility,
  shaped/flat per-head broadcasting, per-head scale finalization, and split
  head-group output reconstruction. The regression uses distinct values for
  token 0/1 and head groups 0/1, so a wrong reshape/interleave order changes
  the expected output. It also rejects per-head scale tensors whose length does
  not match the active KV head count.

This is still experimental. The implementation avoids changing sparse top-k
selection: it computes top-k from the original Q/K/V, then scales Q only for the
FlashInfer attention call and scales output after the grouped result has been
restored to full head order. The next proof point is a GPU smoke test with a
tiny per-head-scale checkpoint.

## What a real non-scalar scale implementation would touch

If the prepared POSTQ package still scores low, the next plausible accuracy
wall is not another safe-max constant. It is the fact that one scalar covers a
whole layer's K tensor and one scalar covers the whole V tensor. A real
per-head/per-block route would need a coordinated runtime change:

1. Loader/schema:
   - `python/sglang/srt/model_loader/weight_utils.py` currently declares JSON
     KV scale values as `Dict[int, Dict[int, float]]`, so the JSON loader can
     only express one float per layer.
   - Checkpoint loading via
     `python/sglang/srt/layers/quantization/kv_cache.py` now accepts scalar or
     rank-1 per-head scales in checkpoint tensors. It still writes scalar
     fallback `k_scale_float/v_scale_float` for existing FlashInfer calls.

2. Attention layer contract:
   - `python/sglang/srt/layers/radix_attention.py` stores
     `k_scale/v_scale/k_scale_float/v_scale_float` as scalar attributes on the
     attention layer.
   - For graph-safe FlashInfer calls, current code passes the Python-float
     versions (`layer.k_scale_float`, `layer.v_scale_float`), not tensors.

3. KV cache write:
   - `python/sglang/srt/mem_cache/memory_pool.py::MHATokenToKVPool.set_kv_buffer`
     divides `cache_k` and `cache_v` by the provided scale before casting to
     FP8. The local helper now supports scalar and rank-1 per-head scale
     broadcasting for both shaped and flattened MHA K/V tensors.
   - A per-token/per-block variant would need extra scale buffers, similar in
     spirit to the existing FP4 pool's `k_scale_buffer/v_scale_buffer`, because
     the scale would become cache-entry metadata rather than a layer constant.

4. MiniCPM FlashInfer path:
   - `python/sglang/srt/layers/attention/minicpm_backend.py` writes K/V through
     `token_to_kv_pool.set_kv_buffer(layer, ..., layer.k_scale, layer.v_scale)`.
   - `python/sglang/srt/layers/attention/minicpm_attention_kernels.py` passes
     `k_scale` and `v_scale` to `BatchDecode/PrefillWithPagedKVCacheWrapper`.
     The current SGLang docs only promise per-tensor FP8 scales, so this wrapper
     call is the hard external contract. If FlashInfer cannot consume vector or
     paged scales, SGLang would have to dequantize selected KV pages before
     attention, apply the Q/output bridge above, or add a new kernel/wrapper
     path.
   - FlashInfer 0.5.3 source checked on 2026-05-31 shows a nuanced state:
     `prefill.py` documents `scale_q/scale_k/scale_v` tensors for per-head FP8
     single-prefill, and internally splits tensor-vs-scalar scale parameters.
     But `BatchDecodeWithPagedKVCacheWrapper.run(...)` still documents
     `k_scale/v_scale` as optional floats, and the upstream FP8 paged decode
     calibration tests use `k_scale = amax.item() / 256`, not a vector. So
     per-head decode may be possible only after a targeted FlashInfer/SGLang
     experiment, not by assuming the current SGLang wrapper path already handles
     it under CUDA graph.

5. Kernel/backend variants:
   - The TRTLLM helper in
     `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
     also documents scalar `k_scale/v_scale` and materializes scalar inverse
     scale pointers. Even outside MiniCPM, the optimized FP8 write path is
     scalar-shaped today.
   - The NSA code has block scale machinery for its index K path, and FP4 KV
     has block scale buffers, but neither is a drop-in MHA K/V scale path for
     MiniCPM sparse FlashInfer.

So the realistic next implementation level, after exhausting POSTQ, is either:

- validate and package per-head K/V scale for MiniCPM MHA FP8 KV using the
  Q/output bridge around the existing FlashInfer paged wrappers; if that proves
  insufficient, then test whether FlashInfer accepts tensor `k_scale/v_scale` or
  can be extended to do so; or
- per-block/per-token scale with KV-pool scale buffers and a new attention
  wrapper/kernel path that reads those scales during dequant.

Both are code/runtime changes. They are not compatible with the current
script-only SOAR package constraint unless the modified SGLang source is also
included and the platform can build/use the new path within the five-hour
budget.

The bridge implementation is now packaged as a guarded experiment, but still
needs platform/GPU validation:

- checkpoint/report format for per-head `k_scale/v_scale`;
- production package format for emitting those per-head checkpoint tensors under
  the names SGLang loads;
- GPU smoke tests for MiniCPM FlashInfer pre-wrapper Q scaling and post-wrapper
  output scaling under prefill, decode, and CUDA graph paths;
- CUDA graph audit to ensure scale tensors are stable graph inputs or static
  layer attributes.

To make this measurable rather than speculative, the root
`scripts/kv_calibrate.py` now records optional per-head max-abs diagnostics in
its JSON report when the model exposes `num_kv_heads` and `head_dim`. The
script still writes scalar checkpoint scales for compatibility, but the report
adds, per layer:

- `k_per_head.max_abs` / `v_per_head.max_abs`;
- `k_per_head.scale` / `v_per_head.scale`;
- `scalar_over_median_head_scale`;
- `median_head_utilization_under_scalar`.

If median utilization is very low on the layers that hurt accuracy, that is
direct evidence that the scalar layer scale is wasting FP8 resolution and that
per-head/per-block runtime work is worth the build risk.

Use the report summarizer to rank that signal instead of eyeballing JSON:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/summarize_fp8kv_head_waste.py \
  /path/to/fp8kv_scales_report.json
```

Interpretation:

- `median_util ~= 1.0`: scalar layer scale is close to per-head scale.
- `median_util < 0.70`: there is visible per-head range imbalance.
- `median_util < 0.50`: the median head is using less than half of the
  available FP8 range under the scalar scale; runtime per-head/per-block scale
  is a plausible accuracy lever.

This tool is diagnostic only. It does not change prepared tarballs and does not
try to inject non-scalar scales into the current scalar-only runtime path.

## Why POSTQ is the strongest prepared precision test

The plain HP224 packages calibrate scale values from the BF16 base model. That
can mismatch the actual runtime writer after GPTQ:

- dense MiniCPM q/k/v projections are W4A16 in the final artifact;
- selected lightning attention layers are restored to BF16;
- qzeros are patched for Marlin after save.

`soar_fp8kv_POSTQ_HP224_OFFLOAD_FIXED_20260531_124940.tar.gz` fixes that source
mismatch by:

- running normal GPTQ save;
- patching qzeros;
- applying the selective BF16 overlay;
- registering `minicpm_sala -> MiniCPMHybridConfig` for the postq reload;
- reloading the final quantized artifact with GPTQModel;
- measuring dense K/V projection outputs from that final artifact;
- injecting `self_attn.attn.k_scale/.v_scale` as the final mutation.

This is the cleanest script-only test before changing runtime scale granularity.

## Why FP4 KV is not the next submission package

The local docs do mention `--kv-cache-dtype fp4_e2m1`, but the same docs mark
FP4 as experimental and tell us to verify backend compatibility. The attention
backend support matrix says MHA FlashInfer supports FP8 KV but not FP4 KV.
MiniCPM-SALA is using the MHA-style sparse FlashInfer path, not MLA.

The code matches that warning:

- `model_runner.py` can parse `fp4_e2m1` only if PyTorch exposes
  `torch.float4_e2m1fn_x2`.
- `model_runner_kv_cache_mixin.py` can allocate `MHATokenToKVPoolFP4`, whose
  KV pool stores packed FP4 plus block scale buffers.
- But `minicpm_attention_kernels.py` only special-cases FP8 to keep
  `q_data_type` at model dtype. For non-FP8 dtypes, it plans query dtype as
  the KV cache dtype. With FP4 this means FlashInfer planning sees FP4 queries.
- `minicpm_backend.py` also casts `q`, `q_rope`, and `k_rope` to the KV dtype
  for non-FP8 explicit KV cache dtypes.
- The CUDA graph FlashInfer planning path passes that planned dtype straight
  into `BatchDecodeWithPagedKVCacheWrapper.begin_forward(...)`.

So a script-only FP4 package would be structurally weaker than the current FP8
queue: it is likely to hit backend/dtype compatibility before it gives a clean
accuracy signal. It also targets maximum memory saving, not accuracy recovery;
the docs warn FP4 can drop more on harder/long-context workloads. Do not submit
FP4 for this queue unless we first change and validate the MiniCPM FlashInfer
backend contract.

## Next decision tree

1. The submitted old `095129` package is stale if it returns a failure or low
   score; do not advance the precision queue from that result.
2. `174343` completed low at 300 x 4K tail (`acc_ori=77.24`), so the queue
   advanced to the strong 8K diagnostic.
3. `225344` completed low at 300 x 8K multi-adaptive (`acc_ori=77.33`,
   `S1=650.26`), so the low quality is not mainly the 4K-tail shortcut. It also
   showed FP8KV has no visible speed win on this MiniCPM sparse path.
4. Current next submit:
   `soar_fp8kv_DENSEQKV_MULTI8K300_HP224_DIAG_OFFLOAD_FIXED_20260601_101730.tar.gz`
   (`md5=3ad3f92828d5595768d6977a4e839dab`). This keeps the same 300 x 8K
   multi-adaptive GPTQ coverage as `225344` and changes only the DENSEQKV
   scale-source hypothesis.
5. If `101730` OOMs, classify from its GPU diagnostics:
   - low free memory before quantize or unexpected external processes means
     platform/dirty-GPU state is the dominant cause;
   - clean GPU before quantize but current process grows to the card limit means
     the 300 x 8K DENSEQKV package itself is still too large.
   `tools/decide_fp8kv_next.py` implements this split as
   `oom_classification: platform_dirty_gpu` vs
   `oom_classification: package_peak_memory` when the log includes the new
   `GPU DIAG` blocks.
6. If `101730` completes low without explicit range symptoms, stop and build a
   POSTQ_MULTI8K HP224 package. Do not submit the prepared 4K POSTQ/PERHEAD
   tarballs, because that would mix scale-source and calibration-strength
   variables.
7. If POSTQ_MULTI8K is still low without explicit range symptoms, then build a
   comparable PERHEAD_MULTI8K bridge. The existing `221735` package validates
   the 4K per-head path shape, but it is not a clean follow-up to a low 8K
   DENSEQKV result.
8. If any E4M3 result gives clear range/outlier/saturation symptoms, route to an
   E5M2 range backup. Prefer a package with comparable successful GPTQ
   calibration strength to the last E4M3 result, rather than the old heavy
   `130420` tarball blindly.
9. If comparable HP224/DENSEQKV/POSTQ/PERHEAD/E5M2 routes all stay below gate,
   stop the prepared FP8KV queue and fall back to the best non-FP8 package.

Use `python3 tools/decide_fp8kv_next.py --last hp224_lowmem --score-json '<Score JSON>'`
for the 141006 result. It now routes to the `174343` mid-calibration diagnostic
package. Use `python3 tools/decide_fp8kv_next.py --current` to print the current
tarball/md5, `--list` to inspect the queue, and `--last denseqkv_multi8k_diag`
for the next platform result. The helper now routes `174343` low-without-range
to `225344`, `225344` low-without-range to `101730`, and `101730`
low-without-range to "build POSTQ_MULTI8K" rather than submitting an old 4K
POSTQ/PERHEAD tarball.

2026-05-31 0115 note: the original DENSEQKV offload package failed in
`prepare_model.sh` before inference because `offload_to_disk` was passed as a
`GPTQModel.load(...)` kwarg and GPTQModel 7.0.0 forwarded it into
`MiniCPMSALAForCausalLM.__init__`. That run is not a DENSEQKV accuracy signal.
All fixed queue packages keep disk offload on `QuantizeConfig` only and the
package audit now rejects `kwargs["offload_to_disk"]`.

2026-05-31 130420 note: the E5M2 range backup was repacked as
`soar_fp8kv_REALCALIB_TAIL_E5M2_OFFLOAD_FIXED_20260531_130420.tar.gz`
(`md5=ad603f6183956f73bd0a05f029c9ca3f`). It keeps the same E5M2 range
hypothesis but removes the known `GPTQModel.load(..., offload_to_disk=...)`
prepare failure path.

2026-05-31 095129 note: the old
`soar_fp8kv_POSTQ_PERHEAD_HP224_BRIDGE_20260531_095129.tar.gz` tarball is not a
valid per-head accuracy test. Audit fails because it still contains the same
`GPTQModel.load(..., offload_to_disk=...)` kwarg path that caused the 0115
prepare failure, and it predates the fixed per-head fallback dequant path.
Interpret a failed 095129 platform run as a stale-package failure, not as
evidence against the per-head bridge.
