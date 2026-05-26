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
| CUDA | 12.8 | bundled flash-attn wheel name is `cu128`; current local wheel targets SM120 |
| Torch | 2.9.1+cu128 | matches platform logs and wheel ABI |
| Transformers | `>=4.45`, verified with 4.57.1 | SALA custom config/model code needs the 4.x custom-model APIs |
| GPTQModel | exactly `7.0.0` | qzeros layout and Marlin metadata are version-sensitive |
| SGLang | bundled `sglang/python` from this repo | platform must not fall back to an unpatched preinstalled SGLang |
| Quantization loader | `--quantization gptq_marlin` | SGLang Marlin path expects GPTQ metadata and qzeros patching |

`prepare_env.sh` hard-pins `gptqmodel==7.0.0` and prints package versions after
install. Do not loosen that pin unless you also re-validate qzeros packing,
`quantize_config.json`, and SGLang load.

The flash-attn wheel must be a real file or a symlink whose target exists.
`pack_submission.py` dereferences symlinks when staging; a broken wheel symlink
is now a hard preflight failure because the platform would otherwise fall back
to network/source install during `prepare_env.sh`.

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
| `NUM_CALIB` | `150` | cover every public calibration row once; multi-adaptive expands this to ~310-330 windows |
| `MAX_CALIB_LEN` | `8192` | tail tokens kept per calibration prompt |
| `CALIB_WINDOW_MODE` | `multi-adaptive` | use `tail` only to reproduce the g128 baseline |
| `MAX_CALIB_WINDOWS` | `4` | cap per-prompt windows for super-long rows; set `3` to reproduce the earlier draft |
| `DISABLE_CHAT_TEMPLATE` | `0` | set `1` to calibrate raw `question\nAnswer: ...` text without chat template |
| `GROUP_SIZE` | `64` | set `128` to reproduce the 2026-05-26 full-g128 platform baseline |
| `MIXED_SKIP_LAYERS` | empty | optional sensitive-layer BF16 skip list for mixed runs |
| `MIXED_SKIP_MODULES` | `all` | module aliases inside skipped layers: `down`, `up`, `gate`, `mlp`, `attn`, etc. |
| `QUANT_TIMEOUT_MIN` | `120` | hard wall-time cap for quantization |
| `FULL_QUANT_PROFILE` | `platform_acc` | log-only profile name for the full high-accuracy submission path |
| `GPTQMODEL_PIN` | `7.0.0` | keep exact unless deliberately testing a new GPTQModel |

The bundled `perf_public_set.jsonl` symlink is intentional. The repo ignores
`*.jsonl` globally, so use `git add -f` if this file is changed or recreated.

## Serving Interface

`prepare_env.sh` exports `SGLANG_SERVER_ARGS`. For this full-W4A16 experiment it
should stay conservative:

```bash
--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.70 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --dtype bfloat16
```

Important interface constraints:

- Do not add `--kv-cache-dtype fp8_*` in this variant. First isolate full W4A16
  correctness with BF16 KV cache. If full-g64 recovers accuracy, stack FP8KV as
  the next single-variable experiment using the existing Path Y runtime fix.
- Keep `--dtype bfloat16`; mixed FP16/BF16 boundaries were a repeated MiniCPM
  sparse-backend failure mode.
- Keep `--dense-as-sparse`; this avoids the compressed-K dtype path that is not
  the target of this experiment.
- `--attention-backend minicpm_flashinfer` matches the current chunk32k-safe
  BF16-KV runtime path. If testing Blackwell FA3 kernel issues, do that in a
  separate FP8KV variant.
- Do not add `--disable-cuda-graph` to this platform path. The 2026-05-26
  full-W4A16 platform success ran with CUDA graph enabled; disabling it was
  only a local Blackwell diagnostic and is too slow for the final runtime.
- Use OpenAI-compatible HTTP requests against `/v1/chat/completions` after
  launch. When testing locally in this environment, use `curl --noproxy '*'` or
  set `no_proxy=localhost,127.0.0.1`; otherwise the proxy can return `502`.

## Local Dry Run

This checks argument parsing, calibration loading, and the full module tree
without running GPU quantization:

```bash
python3 submission_gptqmodel_full_w4a16/quantize_gptqmodel_w4a16.py   --input /root/autodl-fs/models/OpenBMB/MiniCPM-SALA   --output /tmp/full_w4a16_dry   --calib-jsonl submission_gptqmodel_full_w4a16/perf_public_set.jsonl   --num-calib 2   --max-calib-len 512   --dry-run
```

Expected dry-run markers:

- `DRY RUN - no GPU/quantization actually invoked`
- `--group-size 64`
- calibration rows load from `submission_gptqmodel_full_w4a16/perf_public_set.jsonl`
- `answer continuations: 150/150` when using the bundled public calibration set

## Suggested Quantization Run

```bash
FULL_QUANT_PROFILE=platform_acc GROUP_SIZE=64 NUM_CALIB=150 CALIB_WINDOW_MODE=multi-adaptive MAX_CALIB_LEN=8192 MAX_CALIB_WINDOWS=4   bash submission_gptqmodel_full_w4a16/prepare_model.sh   --input /root/autodl-fs/models/OpenBMB/MiniCPM-SALA   --output /root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16_platform_acc-quantized
```

The previous full-g128 platform run is the baseline to beat:

```json
{
  "acc": 97.83,
  "acc_ori": 78.27,
  "final_score": 22.85,
  "benchmark_duration": {
    "S1": 621.91,
    "S8": 1002.06,
    "Smax": 2315.78
  }
}
```

Interpretation: full attention/MLP W4A16 is no longer broken. It realizes a
real S1 latency win, but g128 loses about 2 raw accuracy points versus the
MLP-only/chunk32k-safe line. The default `platform_acc` run is an
accuracy-recovery pass:

- still full quantization: q/k/v/o plus MLP stay W4A16 across all layers
- group-size 64 for finer scales
- multi-adaptive calibration over all 150 public rows, expanding long prompts
  into about 310-330 windows with up to 4 dispersed windows per super-long row
- answer-aware calibration: compact `gold` answers are preserved for all public
  rows; with chat template enabled they are rendered as an `assistant` turn, so
  cwe/fwe/niah/qa answer tokens are represented at the generation boundary

This should cost some Marlin throughput through extra scale loads, but may be
enough to move `acc_ori` above the known full-g128 platform baseline
(`78.27`, `final_score=22.85`) and ideally above 80 without giving up the
full-layer speed benefit.

If g64 loads and serves but accuracy is still poor, test one variable at a time:

```bash
MIXED_SKIP_LAYERS=30,31 MIXED_SKIP_MODULES=down GROUP_SIZE=64 ...
GROUP_SIZE=64 NUM_CALIB=256 CALIB_WINDOW_MODE=multi-adaptive MAX_CALIB_LEN=8192 MAX_CALIB_WINDOWS=3 ...
GROUP_SIZE=64 NUM_CALIB=150 CALIB_WINDOW_MODE=multi-adaptive MAX_CALIB_LEN=16384 MAX_CALIB_WINDOWS=3 ...
GPTQ_DESC_ACT=True GPTQ_STATIC_GROUPS=True GROUP_SIZE=64 NUM_CALIB=150 CALIB_WINDOW_MODE=multi-adaptive ...
```

The narrow mixed candidate above kept almost all attention/MLP projections in
W4A16 and only left the two highest-loss `mlp.down_proj` modules (`layers
30,31`) in BF16. It is a negative result: local shard 1 regressed from uniform
full-g64 `81.33` to `74.67`, and output tokens increased from `235,361` to
`297,743`. Do not submit or continue this exact candidate. The record is in
`experiments/FULL_W4A16_MIXED_SKIP30_31_DOWN_FAILURE.md`; the reproduction
command is kept only for audit:

```bash
bash scripts/run_full_w4a16_skip30_31_down_quant_local.sh
```

If full-g64 clears the accuracy target, the next experiment is full-g64 plus
FP8KV. Keep the exact same quantized artifact and change only the serving
arguments to add the verified FP8KV flashinfer/Path Y stack; do not combine that
with calibration or group-size changes in the same platform submission.

## GPU Queue Helper

If another experiment is using the GPU, do not kill it. When you want a command
that starts only if the GPU is idle right now, use:

```bash
bash scripts/run_full_w4a16_platform_acc_now.sh
```

This manual helper checks total visible GPU memory and the target `PORT`, then
exits `124` without waiting if either is busy. It is the safer choice while the
FP8KV service is being actively restarted or debugged.

If you explicitly want to queue the full platform-acc local run for later, use:

```bash
nohup bash scripts/run_full_w4a16_platform_acc_when_idle.sh \
  > /root/autodl-fs/zyn/logs/full_w4a16_platform_acc_when_idle.log 2>&1 &
```

The helper polls `nvidia-smi --query-gpu=memory.used` and checks that the
target `PORT` can bind; it never kills processes. When total visible GPU memory
use drops below `IDLE_MEM_MIB` (default 2000 MiB) and the port is free, it
launches:

```bash
bash scripts/local_eval_sharded.sh \
  --variant submission_gptqmodel_full_w4a16 \
  --quant-out /root/autodl-fs/zyn/models/submission_gptqmodel_full_w4a16_platform_acc-quantized \
  --max-samples 150 \
  --shard-size 30 \
  --concurrency 32 \
  --force-requant
```

The sharded eval appends one row to `scripts/eval_shards.csv` after every
30-sample shard, copies each shard's predictions to
`scripts/logs/sharded_predictions_*.jsonl`, prints shard and cumulative
accuracy immediately, and runs `scripts/checkpoint_full_w4a16.sh` after each
shard by default so partial CSV, predictions, and logs are committed and pushed
while the long run continues. Use `--no-checkpoint` only for local debugging.

Use the resulting 150-sample cumulative local eval before packing. If local `acc_ori` is
below the known full-g128 baseline (`78.27`), do not submit; move to the next
single-variable acc knob documented above. If it lands between `78.27` and
`80`, treat it as a slot-cost decision rather than an automatic submit. At
`>=80`, pack it as the preferred full-layer candidate.

After a local eval finishes, summarize the latest full result and get the
pack/next-experiment decision with:

```bash
python3 scripts/full_w4a16_decide_after_eval.py
```

This script is CPU-only. It reads `scripts/eval_results.csv` and
`scripts/eval_shards.csv`, locates the latest full predictions file from the
eval log or shard row, runs per-task analysis when possible, and prints either
a `full_preflight.sh --pack` command, a partial-progress decision, or the next
accuracy knob.

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
