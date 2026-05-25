# SOAR Full W4A16 Experiment

This variant is an isolated experiment for full GPTQModel W4A16 on MiniCPM-SALA.
It quantizes both attention/Lightning projections and MLP projections, while the
verified FP8KV path stays on `quant/w4a16`.

## Scope

Quantized modules:

- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`
- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.down_proj`

Kept in BF16 by `quantization_config.dynamic` skips:

- `self_attn.o_gate`
- `self_attn.z_proj`
- `self_attn.q_norm`
- `self_attn.k_norm`
- `self_attn.o_norm`

Those optional modules are not present on every SALA layer, so quantizing them
through a common GPTQModel module tree is unsafe.

## Version Contract

The package is written for the SOAR/AutoDL stack we verified locally:

| Component | Required / expected version | Why it matters |
|---|---:|---|
| Python | platform uses 3.10; local venv may be 3.12 | `flash_attn` wheel is CP310 for platform, not for local 3.12 |
| CUDA | 12.8 | bundled flash-attn wheel name is `cu128` |
| Torch | 2.9.1+cu128 | matches platform logs and wheel ABI |
| Transformers | `>=4.45`, verified with 4.57.1 | SALA custom config/model code needs the 4.x custom-model APIs |
| GPTQModel | exactly `7.0.0` | qzeros layout and Marlin metadata are version-sensitive |
| SGLang | bundled `sglang/python` from this repo | platform must not fall back to an unpatched preinstalled SGLang |
| Quantization loader | `--quantization gptq_marlin` | SGLang Marlin path expects GPTQ metadata and qzeros patching |

`prepare_env.sh` hard-pins `gptqmodel==7.0.0` and prints package versions after
install. Do not loosen that pin unless you also re-validate qzeros packing,
`quantize_config.json`, and SGLang load.

## Platform Interface

The platform is expected to call two scripts from the submission root:

```bash
bash prepare_env.sh
bash prepare_model.sh --input <original_model_dir> --output <processed_model_dir>
```

`prepare_model.sh` accepts only `--input` and `--output` directly; all tuning is
via environment variables:

| Env var | Default | Meaning |
|---|---:|---|
| `CALIB_JSONL` | bundled `perf_public_set.jsonl` if present | calibration source override |
| `NUM_CALIB` | `256` | number of calibration prompts; public rows are cycled if needed |
| `MAX_CALIB_LEN` | `8192` | tail tokens kept per calibration prompt |
| `CALIB_WINDOW_MODE` | `tail` | use `multi-adaptive` only for a slower quality experiment |
| `DISABLE_CHAT_TEMPLATE` | `0` | set `1` to calibrate raw prompts without chat template |
| `GROUP_SIZE` | `128` | set `64` for slower, finer-grained quantization |
| `QUANT_TIMEOUT_MIN` | `90` | hard wall-time cap for quantization |
| `GPTQMODEL_PIN` | `7.0.0` | keep exact unless deliberately testing a new GPTQModel |

The bundled `perf_public_set.jsonl` symlink is intentional. The repo ignores
`*.jsonl` globally, so use `git add -f` if this file is changed or recreated.

## Serving Interface

`prepare_env.sh` exports `SGLANG_SERVER_ARGS`. For this full-W4A16 experiment it
should stay conservative:

```bash
--disable-radix-cache --attention-backend minicpm_flashattn --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.70 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --dtype bfloat16
```

Important interface constraints:

- Do not add `--kv-cache-dtype fp8_*` in this variant. First isolate full W4A16
  correctness with BF16/FP16 KV cache. FP8KV is a separate verified path.
- Keep `--dtype bfloat16`; mixed FP16/BF16 boundaries were a repeated MiniCPM
  sparse-backend failure mode.
- Keep `--dense-as-sparse`; this avoids the compressed-K dtype path that is not
  the target of this experiment.
- `--attention-backend minicpm_flashattn` is inherited from the stable W4A16
  package. If testing Blackwell FA3 kernel issues, do that in a separate
  FlashInfer/FP8KV variant.
- Use OpenAI-compatible HTTP requests against `/v1/chat/completions` after
  launch. When testing locally in this environment, use `curl --noproxy '*'` or
  set `no_proxy=localhost,127.0.0.1`; otherwise the proxy can return `502`.

## Local Dry Run

This checks argument parsing, calibration loading, and the full module tree
without running GPU quantization:

```bash
python3 submission_gptqmodel_full_w4a16/quantize_gptqmodel_w4a16.py   --input /root/autodl-fs/zyn/models/submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_dequant-quantized   --output /tmp/full_w4a16_dry   --calib-jsonl submission_gptqmodel_full_w4a16/perf_public_set.jsonl   --num-calib 2   --max-calib-len 512   --dry-run
```

Expected dry-run markers:

- `DRY RUN - no GPU/quantization actually invoked`
- `--group-size 128`
- calibration rows load from `submission_gptqmodel_full_w4a16/perf_public_set.jsonl`

## Suggested Quantization Run

```bash
GROUP_SIZE=128 NUM_CALIB=256 MAX_CALIB_LEN=8192   bash submission_gptqmodel_full_w4a16/prepare_model.sh   --input /root/autodl-fs/models/OpenBMB/MiniCPM-SALA   --output /root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16-quantized
```

If it loads and serves but accuracy is poor, test one variable at a time:

```bash
GROUP_SIZE=64 NUM_CALIB=256 MAX_CALIB_LEN=8192 ...
NUM_CALIB=512 MAX_CALIB_LEN=8192 ...
CALIB_WINDOW_MODE=multi-adaptive NUM_CALIB=256 ...
```

## Known Risk

Historical notes say a previous full-attention GPTQ attempt served but scored
`acc=0`. This variant exists to re-test that result after the current fixes:

- GPTQModel pinned to `7.0.0`
- qzeros post-check hardening
- tokenizer overwrite fix
- bundled SOAR public calibration data
- dynamic skips aligned with omitted optional modules

Do not replace the stable MLP-only or FP8KV package with this until full local
or platform evaluation proves accuracy is acceptable.
