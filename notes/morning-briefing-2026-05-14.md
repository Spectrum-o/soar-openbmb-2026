# Morning Briefing — 2026-05-14

You went to sleep with one win (chunked-prefill tuning) and asked me to keep
working. Here's the complete picture so you can pick up fast.

## What you already had when you went to bed

- Baseline established: chunked_prefill=8K → 267.67 tok/s, 34s TTFT
- chunk=32K verified: 425.58 tok/s (+59%), 13.5s TTFT (-60%)
- chunk=65K verified: 488.84 tok/s (+83% total), 12.5s TTFT (-63%), ITL also dropped
- All on RTX PRO 6000 Blackwell, 64 prompts × 4096-in × 512-out random-ids

## What I built overnight

Six branches pushed to `origin/Spectrum-o/soar-openbmb-2026`. Listed in
recommended consumption order:

### 1. `config/chunked-prefill-tuned` — DEPLOY THIS, no thinking needed

`run_sala.sh` updated with the winning configuration (`chunked-prefill-size=65536`,
`max-prefill-tokens=65536`, `mem-fraction-static=0.80`). Header has tuning notes
and the two non-obvious gotchas (hidden 16384 default, mem_fraction_static
auto-calc breaks above ~50K chunks).

This is your new baseline-for-future-work. Everything else stacks on top.

### 2. `tools/bench-sweep` — USE THIS for every future param sweep

`tools/bench_sweep.sh` + `tools/compare_bench.py` automate the
"start server, wait healthy, run bench, kill, repeat" cycle you did manually
all night. Resumable — skips already-completed configs. Outputs a markdown
table after each run.

Quick example:
```bash
git checkout tools/bench-sweep
bash tools/bench_sweep.sh tools/configs/concurrency_sweep.txt
# Result table at: /root/autodl-fs/zyn/bench_sweeps/<ts>/summary.md
```

Three preset sweeps under `tools/configs/`:
- `chunk_size_sweep.txt` — re-runs the 8K→65K test for archival
- `concurrency_sweep.txt` — `--max-running-requests` 16/32/48/64 (next quick win to verify)
- `backend_ab.txt` — `minicpm_flashinfer` vs `minicpm_flashattn`

### 3. `quant/w4a16` — THE BIG ONE, champion's approach

Full W4A16 GPTQ pipeline:
- `quantize_to_w4a16.py` — produces W4A16 model from BF16 via llm-compressor
- `run_sala_w4a16.sh` — sglang launcher with `--quantization compressed-tensors`,
  inherits the tuned chunked-prefill params
- `W4A16_README.md` — full workflow

Quick start:
```bash
git checkout quant/w4a16
source /root/autodl-tmp/zyn/sglang/sglang_minicpm_sala_env/bin/activate
uv pip install "llmcompressor>=0.4" datasets

screen -S quantize   # take an hour or three
python quantize_to_w4a16.py \
    --model /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
    --output /root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16
# Ctrl+A D to detach

# When quantize done:
bash run_sala_w4a16.sh
# In another shell, bench it:
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31111 \
    --dataset-name random-ids --num-prompts 64 \
    --random-input-len 4096 --random-output-len 512 \
    --output-file /root/autodl-fs/zyn/bench_w4a16_$(date +%H%M%S).json
```

Per champion's claim: model 18 GB → ~5 GB, decode latency -30 to -50%.

### 4. `notes/op-fusion-analysis` — design doc, READ BEFORE TOUCHING FORWARD

Concrete analysis of the RMSNorm + scaled-residual fusion mentioned in the
champion's writeup. Found:
- sglang has `fused_add_rmsnorm` at `layernorm.py:130`
- MiniCPM doesn't use it because of the `scale_depth/sqrt(num_layers)` factor
- Three implementation paths ranked by risk

I did NOT implement this because forward-path changes need GPU verification.
Path A (pre-scale `o_proj.weight` + `down_proj.weight` at load time) is
documented step-by-step. Expected ROI: 2-5% prefill speedup.

### 5. `notes/fp8-kv-cache-investigation` — explains last night's FP8 failures

Root cause: `--kv-cache-dtype fp8_e4m3` alone doesn't attach a KV cache
quant method to the RadixAttention layers, so `layer.k_scale` stays None,
so FA3 gets fp8 Q without scales and rejects it.

Four fix paths documented (rank A → D). **Path A is a 5-minute test**: just
add `--quantization fp8` to the launcher. Path B is a 10-line code patch.
Recommended: try Path A AFTER W4A16 is shipped, since W4A16 already addresses
the same memory pressure with better ROI.

### 6. `notes/morning-briefing` — this document

## Recommended sequence for today

```
┌─ 1. Pull all the branches (5 min)
│
├─ 2. Verify the tuned config still works (15 min)
│     git checkout config/chunked-prefill-tuned
│     bash run_sala.sh    # smoke test
│     <verify it boots, hit /health>
│
├─ 3. Run W4A16 quantization (60-180 min, mostly hands-off)
│     git checkout quant/w4a16
│     uv pip install 'llmcompressor>=0.4' datasets
│     python quantize_to_w4a16.py --model ... --output ...
│     <coffee>
│
├─ 4. Bench W4A16 vs the tuned bf16 baseline (15 min)
│     bash run_sala_w4a16.sh &
│     <wait for healthy>
│     python -m sglang.bench_serving ... --output-file bench_w4a16_*.json
│     python tools/compare_bench.py \
│         /path/to/bench_chunk65k.json \
│         /path/to/bench_w4a16.json
│
├─ 5. (If time left) Try Path A from fp8-kv-cache-investigation.md (5 min)
│     Add --quantization fp8 to run_sala_w4a16.sh, bench again.
│
├─ 6. (If time left) Implement op fusion (Path A from op-fusion-analysis.md)
│     This is the only un-shipped major lever. Expected 2-5% prefill speedup.
│     Doc has step-by-step but the forward refactor + weight-load hook
│     is non-trivial. Maybe save for tomorrow.
│
└─ 7. Push best-bench-numbers JSONs and a final-decision doc to results/.
      Update run_sala.sh on the chosen optimal branch with the
      combination that won.
```

## Open questions for you

1. **Calibration dataset for W4A16**: I defaulted to wikitext-2-raw-v1 (small,
   safe). Champion called out matching SOAR eval distribution as high-impact.
   Worth doing a second quantization pass with SOAR-aligned data once first
   run succeeds.

2. **What's your accuracy budget?** SOAR requires `overall_accuracy` close to
   baseline. None of my changes touch correctness mathematically, but W4A16
   has accuracy cost. Need eval_model.py from SOAR-Toolkit to verify
   (couldn't set this up overnight without GPU + dataset).

3. **Workload shape**: all our benches use 64 × 4096in × 512out. Real SOAR
   benchmark may use different mixes (short prompts, long prompts, varying
   concurrency). The chunked-prefill win is highest at long prompts; might
   shrink for short ones.

## What I did NOT do

- Did not push to `minicpm_sala` directly. All work is on feature branches.
- Did not implement op fusion in code. Only the design doc.
- Did not commit Plan B or Plan C of fp8-kv-cache-investigation.md as code.
- Did not modify any shared files in `/root/autodl-fs/models/`.
- Did not change `~/.bashrc` or `/usr/local/bin/` on the GPU host.

## State summary

```
Branches (newest first, all pushed):
  origin/notes/morning-briefing             ← you are here
  origin/notes/fp8-kv-cache-investigation
  origin/tools/bench-sweep
  origin/notes/op-fusion-analysis
  origin/quant/w4a16
  origin/config/chunked-prefill-tuned
  origin/quant/fp8-kv-cache                 (your earlier work)
  origin/minicpm_sala                       (untouched main)
```

Local working tree is clean. Workspace is at `/mnt/c/Users/.../sglang`.
