# FP8KV Platform Postmortem — 2026-05-27

## Executive summary

The latest FP8KV package reached `INFERENCING` and completed the platform
benchmark, so the remaining failure is no longer package format, FlashInfer
cache plumbing, CUDA graph startup, or quantization OOM. The run failed the
score because quality regressed below the platform gate:

```text
[2026-05-27 21:43:00] PENDING / PREPARING / DOWNLOADING
[2026-05-27 21:52:01] INFERENCING SGLang 服务已就绪
[2026-05-27 23:56:28] SUCCESS

acc=95.39
acc_ori=76.31
final_score=0.0
S1=720.98  S8=1072.39  Smax=2355.80
```

The package did not time out because the volatile parts of the earlier FP8KV
path were constrained: offline wheel install, hard timeouts, no restored local
FlashInfer JIT cache, cache corruption guard, lower-memory quantization
defaults, and no whole-pool fp8->bf16 upcast during CUDA graph capture.

The package did not score because FP8 KV quality fell too far and the speed
profile did not improve over the known W4A16+bf16 baselines.

## Score comparison

| Variant | acc_ori | final_score | S1 | S8 | Smax | Delta acc vs v5j | Delta S1 vs v5j |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5j_dtype_bf16 | 82.18 | 23.18 | 717.76 | 1067.43 | 2343.38 | +0.00 | +0.00 |
| chunk32k_safe | 80.31 | 22.90 | 720.76 | 1081.21 | 2386.35 | -1.87 | +3.00 |
| full_w4a16 | 78.27 | 22.85 | 621.91 | 1002.06 | 2315.78 | -3.91 | -95.85 |
| fp8kv_platform | 76.31 | 0.00 | 720.98 | 1072.39 | 2355.80 | -5.87 | +3.22 |

Two conclusions are hard from this table:

1. FP8KV lost the correctness gate by 3.69pp (`76.31 < 80`).
2. FP8KV did not create a measurable speed win. S1 is slightly slower than the
   v5j baseline and essentially tied with chunk32k_safe.

## Why this run did not time out

Compared with the earlier 09:13-style package, the successful FP8KV path
removed or bounded the main sources of platform stalls.

### 1. Install steps became bounded

The hardened package wraps `uv pip install` with `timeout --kill-after=30s` and
uses short per-index retries. The earlier package used plain `timeout` in some
places but not the stronger kill-after wrapper, so a Python/CUDA child that
ignored SIGTERM could keep the platform in PREPARING/DOWNLOADING until the
outer five-hour limit.

The flash-attn dependency is also offline-by-default: a cp310 prebuilt wheel is
bundled in the tarball, and missing wheel is a fatal error unless an explicit
download flag is set. This avoids the historical GitHub wheel download stall.

### 2. FlashInfer cache stopped being copied across machines

The initial FlashInfer failure had a Ninja error like:

```text
missing ... /root/autodl-tmp/zyn/.../.cache/flashinfer/.../generated/...cu
```

That means the cache metadata referenced an AutoDL-local absolute path. The
later package does not restore a locally generated FlashInfer JIT cache. It
preserves a platform-local cache if it is valid, and only clears a cache target
when `build.ninja` references a missing CUDA source.

This is the correct rule: do not move FlashInfer JIT cache between machines.
The generated build files are not a portable artifact.

### 3. Quantization memory pressure was reduced

The failing OOM package used the older calibration defaults:

```text
NUM_CALIB=256
MAX_CALIB_LEN=8192
--no-offload-disk
```

The hardened package changed the platform default to:

```text
NUM_CALIB=150
MAX_CALIB_LEN=4096
disk offload enabled by default
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout --kill-after=<grace> 90m ...
```

This makes the prepare stage less accurate in principle, but it is much less
likely to OOM or silently hang. That trade was acceptable for an FP8KV plumbing
test, but it is not a free quality-preserving change.

### 4. Runtime stopped whole-pool upcasting fp8 KV

The previous runtime failure was:

```text
value_cache = value_cache.to(q.dtype)
CUDA out of memory. Tried to allocate 5.90 GiB
```

The current code keeps FlashInfer KV storage in fp8 instead of materializing
the whole KV pool as bf16:

- `minicpm_attention_kernels.py`: `q_data_type` is model dtype for fp8 KV, but
  `plan_kv_data_type` remains the actual KV cache dtype.
- `minicpm_backend.py`: `_prepare_kv_cache_for_attention` returns fp8 KV
  unchanged for the FlashInfer backend.

That is why CUDA graph capture can start now.

## Why acc is low

The strongest current hypothesis is uncalibrated or insufficiently calibrated
KV scales, exposed on long-output or exact-answer tasks.

### Local 60-sample evidence

The server-side 60-sample FP8KV run was not uniformly healthy:

```text
summary:         completed=60  ori_accuracy=81.83  tps=147.12
first 30:        ori_accuracy=85.67  avg_input=23752  avg_output=3459
second 30:       ori_accuracy=78.00  avg_input=32524  avg_output=4402
```

The second half had longer prompts and outputs, and dropped below the platform
80 gate even before the hidden 150-sample run.

Per-task local 60-sample breakdown:

| Task | n | score | avg input tokens | avg output tokens | fail rows |
|---|---:|---:|---:|---:|---:|
| fwe | 12 | 100.00 | 33695 | 769 | 0 |
| niah | 12 | 100.00 | 36572 | 395 | 0 |
| cwe | 12 | 92.50 | 36822 | 11550 | 8 partial |
| mcq | 12 | 66.67 | 152 | 6838 | 4 |
| qa | 12 | 50.00 | 33449 | 101 | 6 |

This is not a server-start or package issue. The model answers can be
systematically close on some tasks and wrong on others. `qa` and `mcq` are the
main quality failures; `cwe` shows long-output partial failures and occasional
very long generations.

### Scale evidence

SGLang's quantized-KV documentation says FP8 KV requires `k_scale` and
`v_scale`; otherwise the backend falls back to generic scale behavior. The
current SALA path has historically relied on scale `1.0` or minimal scale
plumbing. That is risky for SALA because the model has large long-context
activations and uses recurrent/lightning paths in addition to normal attention.

The current code forwards scales to FlashInfer only if they exist:

```python
if kv_data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
    if getattr(layer, "k_scale_float", None) is not None:
        scale_kwargs["k_scale"] = layer.k_scale_float
    if getattr(layer, "v_scale_float", None) is not None:
        scale_kwargs["v_scale"] = layer.v_scale_float
```

So an uncalibrated checkpoint can run, but it is not guaranteed to preserve
quality. This exactly matches the platform outcome: no crash, but acc dropped
from 80-82 to 76.

There is already a candidate fix track in
`submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer_kvscale/`:
run BF16 calibration, collect per-layer K/V max-abs, compute conservative
per-layer `k_scale/v_scale`, and inject those tensors into the quantized model.
That is the right next experiment if FP8KV is revisited.

## Why speed did not improve

FP8 KV saves memory. It only improves benchmark time if the workload is KV
bandwidth or KV capacity bound and the attention backend uses fp8 KV without
expensive dequant overhead.

This platform result suggests those assumptions did not hold here.

### 1. The benchmark is not visibly KV-capacity bound

FP8KV and chunk32k_safe use the same high-level serving shape:

```text
--attention-backend minicpm_flashinfer
--chunked-prefill-size 32768
--max-prefill-tokens 32768
--mem-fraction-static 0.70
--max-running-requests 32
--dense-as-sparse
--quantization gptq_marlin
--dtype bfloat16
```

The only major extra FP8KV knob is `--kv-cache-dtype fp8_e4m3`. If the platform
workload were bottlenecked mainly by KV memory capacity or KV bandwidth, S1/S8
should have moved materially. It did not.

### 2. `dense-as-sparse` changes the hot path

The package runs with `--dense-as-sparse`, which makes `dense_len=0` in the
MiniCPM sparse backend. That forces all sequences through the sparse path
rather than the regular dense prefix path. The hot path includes sparse page
metadata conversion, FlashInfer wrapper planning/replay, SALA-specific head
group reshaping, and non-attention work. FP8 KV does not optimize most of that.

### 3. Dequant/scale plumbing is not free

SGLang's own quantized-KV documentation warns that quantized KV can be slow if
dequantization is not fused with the attention kernel. In this SALA fork, the
FlashInfer path can pass fp8 KV and optional scales into `wrapper.forward`, but
the surrounding sparse metadata and wrapper handling remain. Any saved DRAM
traffic can be offset by wrapper planning, scale handling, or non-KV kernels.

### 4. Full-attention W4A16 showed where a real speed win is

The `full_w4a16` package cut S1 from `717.76` to `621.91`, a real 13% speed
improvement, because it quantized attention projections and reduced GEMM cost.
It still lost score because acc dropped to `78.27`.

That comparison is useful: attention/linear compute changes can move platform
timing; FP8KV did not, so FP8 KV is not attacking the dominant bottleneck for
this exact setup.

## Lessons

1. Treat "server starts" as plumbing only. It does not prove platform quality.
2. Do not carry JIT caches across machines. Cache artifacts with absolute paths
   are worse than no cache.
3. Every platform install/download path needs a bounded timeout and a bundled
   offline fallback when possible.
4. For GPTQ/FP8 experiments, a 30-sample local smoke is only a crash test. Use
   at least a 60- or 150-sample long-output canary before spending platform
   attempts on quality-sensitive changes.
5. FP8 KV should be revisited only with calibrated `k_scale/v_scale` and a
   measured speed win. Without both, it is dominated by v5j_dtype_bf16 or
   non-FP8 mixed-precision routes.
6. When speed and accuracy conflict, compare against `final_score`, not only
   raw S1/S8/Smax. `full_w4a16` was faster but did not beat the MLP-only
   baseline because quality loss erased the timing gain.

## Current recommendation

Do not resubmit the current FP8KV package. It is now a useful engineering proof
that the fp8 path can run to completion, but not a competitive package.

If FP8KV is revisited, the next package should be explicitly labeled as a
`kvscale` experiment and should be gated by:

- local/server 60+ sample `ori_accuracy >= 80`
- no severe `qa` or `mcq` collapse
- platform-projected S1/S8/Smax improvement over v5j or chunk32k_safe
- no restored local FlashInfer JIT cache
- hard timeout on install and quantization

Otherwise, the better near-term route is non-FP8 mixed precision, especially
the physically-rewritten lightning-skip package, because it targets accuracy
recovery rather than betting on unproven KV-memory speedups.

## 2026-05-30 update: ATTNSCALE and HP224

The 17:23 ATTNSCALE package changed the state of the FP8KV investigation:

```text
soar_fp8kv_ATTNSCALE_20260530_172346.tar.gz
acc=96.56
acc_ori=77.24
final_score=0.0
S1=649.48  S8=1016.54  Smax=2306.0
```

This run proves the e4m3 runtime path can be competitive on timing. The problem
is now accuracy only. It should not be attributed to the earlier 13:57 KeyError:
that package injected/loaded scale names inconsistently and failed before
inference. ATTNSCALE writes the real `self_attn.attn.k_scale/.v_scale` params and
finishes evaluation.

The next FP8KV package should be the REALCALIB HP224 follow-up: keep e4m3,
keep the direct scale-param path, but change the calibration from
`safe_max=60, samples=32` to `safe_max=224, samples=64`.
SGLang writes `cache_k.div_(k_scale)` and FlashInfer receives the same
`k_scale_float` for dequant, so increasing `safe_max` reduces the stored scale
and uses more of e4m3's representable range. PyTorch reports
`torch.float8_e4m3fn` max as 448, so 224 keeps 2x headroom while improving
resolution over the conservative ATTNSCALE setting.

Important correction discovered after the first HP224 packaging: SOAR
`perf_public_set.jsonl` stores prompts in the `question` field. Older
`kv_calibrate.py` copies did not read `question`, so they could load 64 JSON
rows but skip every row as `SKIP (no prompt field)`. In that failure mode the
hooks never fire, `k_max/v_max` stay zero, and the injected scales become all
1.0.

Second correction after deeper review: KV calibration also needs the same
tail-window logic as GPTQ calibration. SOAR long prompts put the actual
question/needle near the end, but the KV calibration tokenizer previously used
the tokenizer default truncation side. The TAIL packages set
`tokenizer.truncation_side = "left"` via `--truncation-side left`, so the 8192
tokens used for KV scale statistics match the evaluator-relevant tail region.
This supersedes both plain HP224 and qfield-only HP224 for submission order.

Prepared tail-window real-calib packages:

- `soar_fp8kv_REALCALIB_TAIL_HP224_20260530_234154.tar.gz`, md5
  `e5cf9f8d1df2d5cf5fbe33826b238036`: e4m3 with `safe_max=224`, first submit.
- `soar_fp8kv_REALCALIB_TAIL_MAX448_20260530_234219.tar.gz`, md5
  `72386164882f14dd7bd4549e6636ecab`: e4m3 with `safe_max=448`, a
  precision-extreme follow-up if TAIL HP224 still looks precision-limited.
- `soar_fp8kv_REALCALIB_TAIL_E5M2_20260530_234246.tar.gz`, md5
  `9c6386878975aeb92b8e79bc1e4470af`: e5m2 with `safe_max=224`, a range
  backup if e4m3 appears to saturate on hidden outliers.

All three TAIL tarballs passed full preflight and package-content audit:
`question` parser present, `truncation_side=left` present, zero-forward guard
present, zero-hook guard present, direct `self_attn.attn.k_scale` write present,
no legacy `self_attn.k_scale` tensor write, no `__pycache__`, and the expected
`--kv-cache-dtype`/`KV_CALIB_SAFE_MAX` settings in the packaged scripts.

The audit is now reproducible via:

```bash
python3 tools/audit_fp8kv_realcalib_package.py \
  soar_fp8kv_REALCALIB_TAIL_HP224_20260530_234154.tar.gz \
  soar_fp8kv_REALCALIB_TAIL_MAX448_20260530_234219.tar.gz \
  soar_fp8kv_REALCALIB_TAIL_E5M2_20260530_234246.tar.gz
```

This also verifies the first 64 calibration rows have non-empty `question`
fields and the expected task mix: `cwe=13`, `fwe=13`, `mcq=13`, `niah=13`,
`qa=12`.

When a platform result returns, choose the next FP8KV package with:

```bash
python3 tools/decide_fp8kv_next.py --last hp224 --score-json '<Score JSON>'
```

The helper uses the same queue documented above: startup/load failures stop the
scale A/B and require a plumbing fix; HP224 below gate without range symptoms
routes to MAX448; explicit range/outlier/saturation symptoms route to E5M2; a
correctness-clearing FP8KV result should be kept instead of submitting another
backup blindly.

The audit and decision helpers are covered by:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_fp8kv_realcalib_tools
```

Current result: 10 tests OK.
