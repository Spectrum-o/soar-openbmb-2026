# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **customized fork of [sgl-project/sglang](https://github.com/sgl-project/sglang)** specifically
for the **OpenBMB MiniCPM-SALA** 9B / 1M-context model. Upstream is `OpenBMB/sglang` branch
`minicpm_sala`. This fork is the team's working tree for the **SOAR 2026 inference
optimization competition** (W4A16 quantization push on RTX 6000D, evaluated on a hidden
long-context QA/NIAH/MCQ set). It is **not** a general-purpose SGLang clone — only use it
for MiniCPM-SALA.

The current working branch is `quant/w4a16`. Iteration centers on producing a W4A16
GPTQ artifact that scores non-zero on the SOAR platform.

## Read these BEFORE doing anything

These are living documents and **authoritative** over assumptions:

| File | When to read |
|---|---|
| `SUBMISSIONS.md` (repo root) | **Before building any submission tarball.** Chronological log of 20+ attempts, hard constraints verified by failure, scoreboard. |
| `experiments/STATE_OF_PLAY.md` | Single-source-of-truth on hypotheses (H1–H4, R1–R5), variant dirs, fix commits. |
| `experiments/MORNING_PLAYBOOK.md` | Step-by-step playbook for "platform result arrived, what now". |
| `experiments/V24_PLAN.md` | Decision tree by v23 platform outcome → which `submission_*_v24_*` variant to pack. |
| `docs/calibration_gotchas.md` | qzeros bug, truncation-side, repetition collapse, fp16 sed-patch. |
| `W4A16_README.md` | High-level W4A16 setup + pinned package versions. |

## Common commands

### Run the inference server (BF16 baseline)
```bash
bash run_sala.sh                              # uses ./models/MiniCPM-SALA
MODEL_PATH=/abs/path bash run_sala.sh         # override model path
```
The script encodes the tuned chunked-prefill config (`--chunked-prefill-size 65536
--max-prefill-tokens 65536 --mem-fraction-static 0.80 --dense-as-sparse`).

### Run the W4A16 server
```bash
bash run_sala_w4a16.sh                                              # default path
MODEL_PATH=/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16 bash run_sala_w4a16.sh
```

### Install / setup
```bash
bash install_minicpm_sala.sh                                        # full env install
bash install_minicpm_sala.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
source sglang_minicpm_sala_env/bin/activate
```

### Tests + sanity check
```bash
python3 -m unittest discover tests              # the local unit-test suite (100+ tests)
python3 -m unittest tests.test_fix_qzeros_post_check   # run a single test module
bash scripts/sanity_check.sh                    # tests + variant preflights + tool smoke (<5 sec, no GPU)
```

### Submission iteration loop (this is the daily core loop)
```bash
# Apply V24_PLAN.md decision tree to a platform log → pack the recommended variant
bash scripts/v24_auto.sh
bash scripts/v24_auto.sh --dry-run              # preview without packing
bash scripts/v24_auto.sh --log /path/to/platform_v23.log

# Manual pack (after editing a variant dir)
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16 \
    --check-only                                # preflight only (canary checks)
python3 tools/pack_submission.py \
    --variant submission_gptqmodel_calib_w4a16 \
    --output soar_..._submission_YYYYMMDD_vNN.tar.gz
```

### Local eval (always do this before platform submission — 5h per platform slot)
```bash
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --num-samples 200
```
Quantize → launch server → eval_model.py → prints `acc_ori`. ~10–20 min first run.

## Architecture: what is custom vs upstream

This fork's custom MiniCPM-SALA additions over upstream SGLang are concentrated in:

- `python/sglang/srt/layers/attention/minicpm_backend.py` — hybrid sparse/linear
  attention backend. Selected at server launch with `--attention-backend minicpm_flashinfer`.
  Uses InfLLM v2 + sparse-kernel CUDA implementations from the `3rdparty/` submodules.
- `python/sglang/srt/layers/attention/minicpm_attention_kernels.py`,
  `minicpm_sparse_kernels.py`, `minicpm_sparse_utils.py`, `minicpm_fuse_kernel.py` — kernel
  glue.
- `python/sglang/srt/models/minicpm*.py` — model definitions (the SALA variant relies on
  HF `trust_remote_code` modeling code from the model directory).
- `3rdparty/infllmv2_cuda_impl/` (branch `minicpm_sala`) and `3rdparty/sparse_kernel/` —
  required CUDA kernels, compiled at install time by `install_minicpm_sala.sh`.

Submission tarballs vendor a **trimmed copy of `python/`** (see `submission_*/sglang/`),
so weight loading goes through the platform-side install of THIS source tree, not the
public `pip install sglang`.

## Submission workflow architecture

Every "submission_*" directory is one variant for the SOAR platform. They share most files
via symlinks back to the base `submission_gptqmodel_calib_w4a16/`:

```
submission_gptqmodel_calib_w4a16/                  # v23 base — owns the source files
├── README_SUBMISSION.md
├── prepare_env.sh                                 # pip pins, sed-patches, exports SGLANG_SERVER_ARGS
├── prepare_model.sh                               # arg-parser wrapper → invokes the quantizer
├── quantize_gptqmodel_w4a16.py                    # GPTQModel + MiniCPM-SALA registration + qzeros fix
├── perf_public_set.jsonl                          # bundled calibration data
├── flash_attn-2.8.3+cu128torch2.9-cp310-...whl    # bundled wheel (platform has no GitHub network)
└── sglang/                                        # bundled SGLang source

submission_gptqmodel_calib_w4a16_v24_pin_transformers/  # forks one variable from v23
├── README_SUBMISSION.md
├── prepare_env.sh                                 # ← only file that differs
└── rest → symlinked into v23 base
```

The **platform contract**: SOAR's harness runs `prepare_env.sh` then `prepare_model.sh
--input <bf16> --output <quantized>` then launches `sglang.launch_server` with the
exported `SGLANG_SERVER_ARGS`. A submission must therefore (1) install all deps without
network access except PyPI, (2) write a quantized model the bundled SGLang can load,
(3) export server args.

`tools/pack_submission.py --check-only` runs an 11-canary preflight on a variant dir
(qzeros fix present, gptqmodel pin, H4 tokenizer-overwrite guard, SGLANG_SERVER_ARGS
exported, chunked-prefill not accidentally bumped, etc.). Always run preflight before
packing.

## Hard constraints (verified by failure — don't reinvent these)

From `SUBMISSIONS.md` "Hard constraints" — read that section in full before changing
quantization or inference flags. Highlights:

- `--quantization gptq_marlin` only supports `(bits=4, sym=True)` → uint4b8.
- For uint4b8: `scale = max(abs(w)) / 7`, NEVER `/ 8`. The `/8` bug cost a 5h submission.
- FP8 KV cache (`--kv-cache-dtype fp8_*`) is INCOMPATIBLE with MiniCPM sparse backend
  (FlashAttention rejects FP8).
- `--disable-cuda-graph` is for debug only; baseline runs WITH CUDA graph.
- SALA's HF modeling code hard-asserts `_attn_implementation == "flash_attention_2"`.
- `transformers >= 5.0` does an eager `import flash_attn` at `PreTrainedModel.__init__`.
- gptqmodel 7.0.0 + `sym=True` writes `qzeros=7` but Marlin expects `8` — every layer
  gets `+1*scale` bias → garbage output. Fix lives in
  `quantize_gptqmodel_w4a16.py::fix_qzeros_for_marlin` (patches `0x77777777` →
  `0x88888888` post-save). Idempotent. Removing this single line reproduces acc=0.
- GPTQModel.save() re-serializes tokenizer with a split chat_template (inline +
  `chat_template.jinja` sidecar). Transformers `<4.47` silently ignores the sidecar
  → empty chat template → garbage. Fix lives in
  `quantize_gptqmodel_w4a16.py::copy_runtime_assets` (overwrites tokenizer from
  base BF16 unconditionally).

## Where work actually happens (runs are on AutoDL, not locally)

This working tree at `/home/zyn/program/.../sglang/` is a **dev mirror**. Quant runs,
SGLang launches, and eval all happen on AutoDL (`/root/autodl-tmp/zyn/sglang`,
`/root/autodl-fs/zyn/...`). When writing commands, default to server-side paths.
Bootstrap a fresh AutoDL instance via the `bootstrap_gpu.sh` workflow documented in
the memory file `reference_autodl_bootstrap.md`.

## Code style / hygiene

- `.pre-commit-config.yaml` runs ruff (F401, F821), black, isort, codespell, yaml/toml
  checks. Activate via `pre-commit install` once per clone.
- The `no-commit-to-branch` hook is active — feature work goes on a branch, not `main`.
- `submission_*/sglang/` is a vendored copy. Don't edit it directly; edit `python/`
  and re-sync via the pack tool.
