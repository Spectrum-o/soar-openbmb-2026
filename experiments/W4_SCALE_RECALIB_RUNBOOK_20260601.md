# W4 Scale Recalibration Runbook - 2026-06-01

## Goal

Test whether GPTQ W4A16 `.scales` tensors are still an accuracy bottleneck,
without changing the proven serving recipe.

2026-06-02 current priority: this is the score-directed route. POSTQ calibrates
FP8 KV scales on the lower-EV FP8KV branch; it does not rewrite W4 `.scales`.
Use POSTQ results as engineering evidence, but do not treat a POSTQ pass/fail as
an answer to this W4 weight-scale question.

This experiment is intentionally narrow:

- keep `.qweight`, `.qzeros`, `.g_idx`, dynamic skip rules, BF16 overlay,
  quantization config, and server args unchanged;
- rewrite only GPTQ `.scales` tensors on a copied artifact;
- compare original vs recalibrated artifacts under identical SGLang launch
  args;
- use local public-set eval for accuracy direction and random benchmark rows
  for runtime bottleneck signals.

The scale rewrite itself is CPU/weight-only and does not need prompts or a GPU.
The GPU is needed for the A/B serving run that tells us whether the rewrite
actually changes accuracy or speed.

## Files

- `tools/recalibrate_gptq_w4_scales.py`: rewrites `.scales` by least-squares
  matching the fixed packed int4 values back to the BF16 base weights.
- `scripts/autodl_w4_scale_recalib_start.sh`: AutoDL one-key launcher. It
  auto-detects the best non-FP8 W4A16 artifact if `--artifact` is omitted,
  writes an environment snapshot, selects fast/full bench settings, and starts
  the A/B runner in foreground or `nohup` background mode.
- `scripts/w4_scale_recalib_ablation.sh`: GPU A/B runner. It copies the
  artifact, applies the scale rewrite, dequant-verifies both artifacts, serves
  both artifacts, runs a benchmark matrix, optionally runs public-set eval, and
  writes a summary.
- `tools/summarize_w4_scale_ablation.py`: reads the work dir and emits
  `summary.md` with scale MSE gain, bench deltas, eval delta, and a bottleneck
  read.
- `tests/test_recalibrate_gptq_w4_scales.py`: synthetic scale-rewrite safety
  coverage.
- `tests/test_summarize_w4_scale_ablation.py`: summary/bottleneck-read
  coverage.

## Pick the Artifact

Start from the best non-FP8 W4A16 artifact, not from an FP8KV artifact. The
current reference is the `last7attn_rmsopfusion_calib300_multi120` family,
because it is the platform all-time-high W4A16 point and avoids mixing FP8KV
loss into the scale test.

Use the exact serving args from that variant:
`submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120/prepare_env.sh`
exports `--attention-backend minicpm_flashinfer --chunked-prefill-size 32768
--max-prefill-tokens 32768 --mem-fraction-static 0.70 --dense-as-sparse
--quantization gptq_marlin --dtype bfloat16`. The A/B runner reads those args
when `--variant` is set, so original and recalibrated artifacts differ only by
the copied artifact's `.scales` tensors.

On AutoDL, locate the quantized artifact first:

```bash
find /root/autodl-fs /root/autodl-tmp -maxdepth 4 -type d \
  -name '*last7attn*rmsopfusion*calib300*quantized*' 2>/dev/null
```

If there are multiple candidates, prefer the artifact that matches the package
submitted as:

```text
soar_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120_20260529_1405.tar.gz
```

Do not run this first on `desc_act`, old `accmax`, or FP8KV packages. Those
would confound the result with known negative or unresolved changes.

## Full GPU A/B

Recommended AutoDL entry after pulling the latest branch:

```bash
cd /root/autodl-tmp/zyn/sglang

bash scripts/autodl_w4_scale_recalib_start.sh --full
tail -f /root/autodl-fs/zyn/w4_scale_recalib_*/driver.log
```

The wrapper starts in background by default and records the selected artifact,
git commit, branch, `nvidia-smi`, bench settings, and launch command in
`env_snapshot.txt` under the work dir. Use `--foreground` if you want the run to
occupy the current shell:

```bash
bash scripts/autodl_w4_scale_recalib_start.sh --full --foreground
```

If auto-detection picks the wrong artifact, pass it explicitly:

```bash
bash scripts/autodl_w4_scale_recalib_start.sh \
  --full \
  --artifact /root/autodl-fs/zyn/models/<best-w4a16-artifact>
```

The underlying command is still available when you need exact manual control:

```bash
cd /root/autodl-tmp/zyn/sglang

bash scripts/w4_scale_recalib_ablation.sh \
  --artifact /root/autodl-fs/zyn/models/<best-w4a16-artifact> \
  --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
  --variant submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120 \
  --eval
```

The script writes:

- `w4_scale_recalib_dryrun.json`: proposed scale changes;
- `w4_scale_recalib_apply.json`: actual scale rewrite report;
- `verify_original_fast.log` and `verify_recalibrated_fast.log`: dequant smoke
  for both artifacts;
- `logs/server_orig.log` and `logs/server_recal.log`: SGLang startup/runtime
  logs;
- `logs/bench_*`: random benchmark JSONL/stdout per profile;
- `logs/eval_orig.log` and `logs/eval_recal.log`: optional public-set eval;
- `bench_summary.csv`: compact bench rows;
- `summary.md`: bottleneck read.

## Faster First Pass

Use this if the full public-set eval would block other work:

```bash
bash scripts/autodl_w4_scale_recalib_start.sh --fast
tail -f /root/autodl-fs/zyn/w4_scale_recalib_*/driver.log
```

Equivalent manual command:

```bash
EVAL_SAMPLES=60 \
BENCH_MATRIX='decode,16,1024,256,1;concurrent,32,4096,128,8' \
bash scripts/w4_scale_recalib_ablation.sh \
  --artifact /root/autodl-fs/zyn/models/<best-w4a16-artifact> \
  --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
  --variant submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120 \
  --eval
```

This is a direction check only. If it shows an accuracy gain, rerun the full
A/B before packaging.

Recommended now:

1. Run `--fast` first. It is enough to catch "clearly bad" global scale rewrites.
2. Run `--full` only if `summary.md` reports `delta_acc_ori >= +0.5` and random
   benchmark throughput is within noise.
3. If global is flat/negative, try the narrow MLP-late regex before abandoning
   the idea; do not package a flat global rewrite.

## Narrow Module Tests

If global scale rewrite is flat or negative, isolate the sensitive regions
instead of packaging globally:

```bash
bash scripts/autodl_w4_scale_recalib_start.sh \
  --full \
  --module-regex 'model.layers.(2[4-9]|3[0-1]).*mlp'
```

Equivalent manual command:

```bash
MODULE_REGEX='model.layers.(2[4-9]|3[0-1]).*mlp' \
bash scripts/w4_scale_recalib_ablation.sh \
  --artifact /root/autodl-fs/zyn/models/<best-w4a16-artifact> \
  --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
  --variant submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120 \
  --eval
```

Useful regexes:

```text
model.layers.*mlp
model.layers.*(gate_proj|up_proj|down_proj)$
model.layers.(2[4-9]|3[0-1]).*
model.layers.*self_attn.(q_proj|k_proj|v_proj|o_proj)$
```

The self-attention regex may match few or no tensors on selective-BF16
artifacts, because selected attention layers can be restored to BF16. That is
expected.

## Bottleneck Interpretation

Read `summary.md` first.

Accuracy:

- `delta_acc_ori >= +0.5`: W4 scale reconstruction is a real candidate
  bottleneck. Repeat once, then package only if speed is neutral and dequant
  verification stays clean.
- `-0.5 < delta_acc_ori < +0.5` with positive weight MSE gain: static
  `.scales` reconstruction improved but model accuracy did not. The bottleneck
  is probably not global W4 scale MSE; look at activation-aware calibration,
  module selection, or runtime kernels.
- `delta_acc_ori <= -0.5`: global rewrite is unsafe. Try narrower module regex
  or abandon this lever.

Runtime:

- Scale-only rewrite should not materially change speed. A repeated >3%
  throughput delta usually means server args, CUDA graph/JIT warmup, cache
  state, or GPU pressure changed.
- `decode` isolates mostly decode/GEMM/sampling pressure.
- `concurrent` adds scheduling and request pressure.
- `longctx` stresses prefill, sparse metadata, page-table work, and memory
  pressure. Worse TTFT here is a prefill/runtime symptom, not a scale-tensor
  effect by itself.

Decision:

- Accuracy up, speed flat: build a W4-scale package candidate from the same
  proven non-FP8 recipe.
- Accuracy flat, speed flat: W4 scale is not the current bottleneck. Move to
  module-selection or strictly accuracy-neutral fusion.
- Accuracy down: do not package globally.
- Speed changes materially without accuracy movement: repeat before making any
  conclusion; `.scales` alone should not change the runtime graph.

## CPU Sanity Commands

Before using GPU time, check syntax and the scale tool locally:

```bash
bash -n scripts/autodl_w4_scale_recalib_start.sh
bash -n scripts/w4_scale_recalib_ablation.sh
python3 -m py_compile tools/recalibrate_gptq_w4_scales.py tools/summarize_w4_scale_ablation.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_recalibrate_gptq_w4_scales \
  tests.test_summarize_w4_scale_ablation
```

2026-06-02 local verification:

- `bash -n scripts/autodl_w4_scale_recalib_start.sh scripts/w4_scale_recalib_ablation.sh` OK
- `python3 -m py_compile tools/recalibrate_gptq_w4_scales.py tools/summarize_w4_scale_ablation.py` OK
- `python3 -m unittest tests.test_recalibrate_gptq_w4_scales tests.test_summarize_w4_scale_ablation` -> 4 OK

On a server that has the artifact and base model, a cheap dry run is:

```bash
python3 tools/recalibrate_gptq_w4_scales.py \
  --artifact /root/autodl-fs/zyn/models/<best-w4a16-artifact> \
  --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
  --dry-run \
  --limit 4 \
  --report /tmp/w4_scale_dryrun.json
```

This should not modify the artifact.
