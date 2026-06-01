# W4 KV calibration status 2026-06-01

## Current target

- Repo: `Spectrum-o/soar-openbmb-2026`, branch `quant/fp8-kv-cache`.
- Selected W4A16 artifact:
  `/autodl-fs/data/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized`.
- Reason to calibrate on this W4 artifact: KV scale factors depend on the
  hidden states entering K/V projections. Those hidden states change after W4
  MLP quantization, so collecting scales on the BF16 source model can miss the
  distribution served on platform.

## Implemented

- `MiniCPMSALAForCausalLM.load_kv_cache_scales()` now loads SGLang
  `--quantization-param-path` JSON for MiniCPM radix-attention layers and
  fails if a provided file loads zero radix layers.
- `run_sala.sh` accepts `QUANTIZATION_PARAM_PATH`, validates it, prints launch
  knobs and a GPU snapshot, then passes `--quantization-param-path`.
- `MiniCPMAttention` can collect K/V amax stats from the exact SGLang W4
  serving path when `SGLANG_MINICPM_KV_CALIB_DIR` is set. This is now the
  preferred collection path because it avoids HF GPTQ loader drift.
- `scripts/zyn_collect_kv_scales_sglang.sh` starts the selected W4 SGLang
  server in calibration mode, sends calibration prompts, merges collected
  stats, and writes the final FP8 scale JSON. It defaults collection to BF16
  KV cache so the measured K/V amax comes from the W4A16 path without
  uncalibrated FP8 KV perturbing downstream layers; serving/eval still uses
  FP8 KV with the collected scale JSON.
- `tools/zyn_merge_minicpm_kv_stats.py` merges per-rank SGLang stats into the
  SGLang `--quantization-param-path` schema. A fake-stats smoke test verified
  it emits all 32 layer keys, with real scales for radix layers and `1.0` for
  lightning layers.
- `tools/zyn_send_kv_calib_requests.py` sends the selected calibration prompts
  to a running OpenAI-compatible SGLang server with timing logs.
- `tools/zyn_collect_minicpm_kv_scales.py` selects representative calibration
  rows and can collect through a compatible HF GPTQ environment. It is kept as
  a secondary path; the SGLang path above is safer for this repo.
- A dry-run W4/KV calibration row set was written:
  `/autodl-fs/data/zyn/calib_sets/w4_kv_selected_20260601.jsonl`.
  It has 16 rows: 4 short, 4 medium, 4 long, 4 super; task mix covers
  `mcq`, `cwe`, `fwe`, `niah`, and `qa`.
- `scripts/zyn_partition_eval.py` now logs per-request elapsed time, input and
  output tokens, output TPS, rolling accuracy, total TPS, and GPU snapshots at
  eval startup and partition boundaries.

## Commands

Preferred: collect KV scales through the exact SGLang W4 serving path:

```bash
cd /autodl-fs/data/zyn/repo_quant_fp8_kv_cache_latest
MODEL_PATH=/autodl-fs/data/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized \
DATA_PATH=/autodl-fs/data/zyn/calib_sets/w4_kv_selected_20260601.jsonl \
bash scripts/zyn_collect_kv_scales_sglang.sh
```

Secondary HF-loader path, only if the Python env has a compatible GPTQ backend:

```bash
cd /autodl-fs/data/zyn/repo_quant_fp8_kv_cache_latest
python tools/zyn_collect_minicpm_kv_scales.py \
  --data-path /autodl-fs/data/zyn/calib_sets/w4_kv_selected_20260601.jsonl \
  --num-samples 16 \
  --max-window-len 8192 \
  --max-windows-per-sample 2 \
  --force
```

Serve with collected scales:

```bash
MODEL_PATH=/autodl-fs/data/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized \
QUANTIZATION_PARAM_PATH=/autodl-fs/data/zyn/kv_scales/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized_fp8_e4m3_kv_scales.json \
bash run_sala.sh
```

First-30 smoke eval with scales:

```bash
MODEL_PATH=/autodl-fs/data/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized \
QUANTIZATION_PARAM_PATH=/autodl-fs/data/zyn/kv_scales/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe-quantized_fp8_e4m3_kv_scales.json \
bash scripts/zyn_run_candidate_first30.sh
```

## Bottlenecks and risks

- Scale collection dry-run and tokenizer load were verified here. Actual W4
  forward did not run in this shell because the current system Python lacks a
  compatible GPTQ runtime: `optimum` enters the GPTQ path but has no usable
  `gptqmodel/auto-gptq` backend (`QuantizeConfig` is undefined). The collector
  now fails with an explicit message for this case.
- Initializing the runtime submodules in this checkout also hit an environment
  bottleneck: `/autodl-fs/data` has free bytes but is inode-constrained. A
  recursive submodule checkout failed while writing the nested cutlass tree
  with `No space left on device`; `df -i /autodl-fs/data` showed 100% inode
  use at the failure point. The partial submodule checkout was deinitialized
  to restore several thousand inodes.
- Do not solve that by blindly upgrading the shared system `transformers`.
  Run collection in the original quantization/runtime venv, or create a
  separate throwaway env with mutually compatible `transformers`, `optimum`,
  and `gptqmodel`.
- Preferred workaround is not to use HF GPTQ loading at all: use
  `scripts/zyn_collect_kv_scales_sglang.sh`, which loads W4 through the same
  SGLang `gptq_marlin` path used for serving.
- If actual collection OOMs at 8192-token windows, reduce
  `--max-window-len` to 4096 first, then increase `--num-samples` to keep
  coverage. Do not expand preprocessing time further until the scale path is
  proven end-to-end.
- FP8 KV can reduce KV memory and decode bandwidth, so it may improve long
  decode/concurrency. It will not necessarily speed prefill; A/B timing is
  still required with the same W4 model, same eval split, and the new logs.
- Platform submission must include both the modified SGLang source and the KV
  scale JSON, and the server launch must set `QUANTIZATION_PARAM_PATH`.
