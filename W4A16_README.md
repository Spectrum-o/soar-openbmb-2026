# W4A16 Quantization for MiniCPM-SALA

This branch implements the **GPTQ + Marlin W4A16 weight quantization** that the
SOAR Week 4 champion ("智算一队") used to crack the prize-tier leaderboard. Their
notes summarized the value:

> "推理 Kernel 使用 Marlin，在小 batch 下可以达到接近理论极限的 4 倍带宽节省，
> 非常适合 decode 阶段的 memory-bound 特性。应用 W4A16 后模型体积压缩至原先的
> 约四分之一，带宽瓶颈得到明显缓解，各档并发场景的端到端耗时均有显著下降。"

## Files

- `quantize_to_w4a16.py` — produces a W4A16 model from the BF16 base via
  llm-compressor + GPTQ.
- `run_sala_w4a16.sh` — launches sglang against the quantized model with
  `--quantization compressed-tensors` and the tuned chunked-prefill params.

## End-to-end workflow

### 1. Install the quantization toolkit (one-time)

The base venv built by `install_minicpm_sala.sh` does **not** include
llm-compressor. Add it:

```bash
source /root/autodl-tmp/zyn/sglang/sglang_minicpm_sala_env/bin/activate
uv pip install "llmcompressor>=0.4" datasets
```

The base venv also does **not** include `flash-attn` (it installs
`flash-linear-attention` for SALA's linear attention layers, but not the
mainline `flash-attn` package). HuggingFace transformers requires `flash-attn`
to load the SALA model with `attn_implementation="flash_attention_2"`, which
is a hard requirement (the model code asserts it in
`modeling_minicpm_sala.py:1328`).

```bash
uv pip install ninja
MAX_JOBS=20 uv pip install flash-attn --no-build-isolation
python -c "import flash_attn; print(flash_attn.__version__)"  # verify
```

If compile fails on Blackwell (sm_120), force the arch:
```bash
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=20 uv pip install flash-attn --no-build-isolation
```

`flash-attn` is only needed for the quantization step. Once the W4A16 model
is produced, sglang's own FA3 kernels handle serving and `flash-attn` is
optional.

### 2. Run quantization

```bash
# Default: wikitext calibration, 512 samples, max_seq_len=2048
# Wall time: ~1-3 hours on RTX PRO 6000 Blackwell
python quantize_to_w4a16.py \
    --model /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
    --output /root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16
```

Output is ~5 GB on disk (vs 18 GB bf16). The directory contains compressed-
tensors format weights, tokenizer, and a modified config.json.

### 3. Launch the W4A16 server

```bash
bash run_sala_w4a16.sh
```

### 4. Bench it against the bf16 baseline

```bash
# (server must already be up via run_sala_w4a16.sh)
cd /root/autodl-tmp/zyn/sglang
source sglang_minicpm_sala_env/bin/activate
python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 31111 \
    --dataset-name random-ids --num-prompts 64 \
    --random-input-len 4096 --random-output-len 512 \
    --output-file /root/autodl-fs/zyn/bench_w4a16_$(date +%H%M%S).json
```

Compare against the latest bf16 numbers in `/root/autodl-fs/zyn/bench_*.json`.

## Expected impact

Per champion's notes and the SALA workload profile:
- **Decode latency**: -30 to -50% (memory-bound regime is exactly where W4A16 shines)
- **Prefill latency**: smaller but non-zero gain (FFN GEMM is compute-bound, but
  W4A16 GEMM still has lower memory pressure)
- **Model size**: 18 GB -> ~5 GB
- **Accuracy**: typically <1% drop on standard benchmarks if calibration is
  representative

## Tuning the calibration set (do this after first successful run)

The default wikitext calibration is a "make it work" baseline. The champion
explicitly called out that calibration choice was high-impact and they
"iterated several rounds" matching the eval task distribution.

For SOAR specifically:
1. Inspect the eval task types in SOAR-Toolkit (RULER, needle-in-haystack,
   long-doc Q&A, etc.).
2. Build a calibration set that mirrors that distribution — long contexts,
   needle-style retrieval prompts, etc.
3. Re-quantize and re-benchmark accuracy via `eval_model.py`.

Override the default like so:

```bash
python quantize_to_w4a16.py \
    --model ... --output ... \
    --calib-dataset path/to/your/sft_jsonl_dataset \
    --calib-config "" \
    --calib-split train \
    --calib-text-column text \
    --num-samples 1024 \
    --max-seq-len 8192
```

## Composing with other optimizations

W4A16 is **orthogonal** to:
- Tuned chunked-prefill params (already baked into `run_sala_w4a16.sh`).
- Future operator fusion (RMSNorm+RoPE).
- KV cache quantization (currently bf16; FP8 KV cache path on SALA is
  non-trivial — see `quant/fp8-kv-cache` branch notes).

All wins should stack. Bench each in isolation first to attribute properly.

## Known risks

- SALA's hybrid sparse/linear attention has custom kernels in
  `python/sglang/srt/layers/attention/minicpm_backend.py`. These read from KV
  cache (unquantized here) and call quantized Linear projections; they should
  be transparent to W4A16 but **verify after first quantization that all 32
  layers load and run correctly**.
- If `oneshot` errors on a specific Linear module name during calibration, add
  that pattern to `--ignore`. Common candidates if issues arise: gate/up/down
  projections in Lightning Attention layers (`z_proj`, `o_gate`).
- Wall time grows linearly with `num_samples * max_seq_len * num_layers`.
  Start with the 512 / 2048 defaults; only scale up after a successful run.
