# Bench Sweep Tooling

Automate "tweak a parameter, restart server, run bench, compare numbers" so you
don't have to babysit it. Designed to be resumable: if a config in the sweep
fails or you Ctrl+C mid-run, re-running picks up where it left off.

## Files

- `tools/bench_sweep.sh` — runs each config in a list, saves server log +
  bench JSON, rebuilds summary table after each result.
- `tools/compare_bench.py` — diffs N bench JSONs and prints/saves a markdown
  table with deltas vs the baseline (first file).
- `tools/configs/*.txt` — sweep config presets. Each line:
  `<tag>,<extra sglang launch args>`. Lines starting with `#` are comments.

## Quick start

```bash
cd /root/autodl-tmp/zyn/sglang

# Sweep chunked-prefill-size from 8K to 65K
bash tools/bench_sweep.sh tools/configs/chunk_size_sweep.txt

# When done, the summary is at /root/autodl-fs/zyn/bench_sweeps/<ts>/summary.md
```

The summary looks like:

```markdown
| metric          | chunk_08k_bf16 (base) | chunk_32k_bf16 | Δ vs base |
| --------------- | --------------------- | -------------- | --------- |
| out_tput (tok/s)| 267.67                | 425.58         | +59.0% 🚀 |
| TTFT_mean (ms)  | 34,049                | 13,545         | -60.2% 🚀 |
| ITL_mean (ms)   | 54.94                 | 50.85          | -7.4% 🚀  |
| ...             |                       |                |           |
```

## Available presets

- **`chunk_size_sweep.txt`** — chunked-prefill-size 8K → 65K, the parameter
  we already proved has 60%+ impact.
- **`concurrency_sweep.txt`** — `--max-running-requests` 16/32/48/64. With
  64 prompts in bench but capacity 32, half were queueing — raising this may
  help if KV cache room permits.
- **`backend_ab.txt`** — `minicpm_flashinfer` vs `minicpm_flashattn`. Pure A/B
  on backend choice with everything else fixed.

## Defining your own sweep

Create any `.txt` file with the same `<tag>,<args>` format. Notes:

- Don't include `--model`, `--port`, `--trust-remote-code`, `--disable-radix-cache`,
  `--skip-server-warmup`, `--dense-as-sparse` — `bench_sweep.sh` adds those.
- DO include any parameter you want varied (`--chunked-prefill-size`, etc).
- Use a unique tag per row; tag determines filename.

## Environment overrides

```bash
MODEL_PATH=/abs/path bash tools/bench_sweep.sh tools/configs/...
PORT=8000 bash tools/bench_sweep.sh ...
BENCH_NUM_PROMPTS=128 BENCH_INPUT_LEN=8192 bash tools/bench_sweep.sh ...
OUTPUT_DIR=/tmp/my_sweep bash tools/bench_sweep.sh ...
SERVER_BOOT_TIMEOUT=900 bash tools/bench_sweep.sh ...   # for slow Blackwell warmups
```

## Comparing existing bench files manually

If you already have a pile of `*.bench.json` from manual runs:

```bash
python tools/compare_bench.py \
    /root/autodl-fs/zyn/bench_baseline_*.json \
    /root/autodl-fs/zyn/bench_chunk32k_*.json \
    /root/autodl-fs/zyn/bench_chunk65k_*.json \
    --output /tmp/my_compare.md

cat /tmp/my_compare.md
```

## What's resumable

`bench_sweep.sh` skips any config whose `.bench.json` already exists in
`OUTPUT_DIR`. So if a sweep takes 30 minutes and your SSH drops at minute 20,
just re-run the same command and it picks up from where it left off.

To force re-run a single config: `rm ${OUTPUT_DIR}/<tag>.bench.json` then sweep.
