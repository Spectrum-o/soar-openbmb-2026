# SOAR platform test logic notes, 2026-05-27

Sources checked:

- https://github.com/OpenBMB/SOAR-Toolkit/blob/main/README.md
- https://github.com/OpenBMB/SOAR-Toolkit/blob/main/README_EN.md
- https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_model.py
- https://github.com/OpenBMB/SOAR-Toolkit/blob/main/bench_serving.sh

## Submission contract

The current official submission path is a `.tar.gz` package with:

- `prepare_env.sh` required.
- `prepare_model.sh` optional.
- Any other code/resources as needed, within the platform package limit.

The platform flow described by SOAR-Toolkit is:

1. Start the base environment.
2. Source `prepare_env.sh`. This is where package installs and exported
   runtime variables such as `SGLANG_SERVER_ARGS` take effect.
3. If present, call:

   ```bash
   bash prepare_model.sh --input <original model path> --output <processed model path>
   ```

4. Use the processed model path for the later inference service and benchmark
   phases.

Important detail: `prepare_env.sh` is sourced, not executed as an isolated
child process. Exported environment variables survive into the platform's main
script.

## Server launch interface

The SOAR base image documents these environment variables:

- `MODEL_PATH`, default `/models/MiniCPM-SALA`
- `HOST`, default `0.0.0.0`
- `PORT`, default `30000`
- `SGLANG_SERVER_ARGS`, default roughly:
  `--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse`

For our packages, the important practical surface is `SGLANG_SERVER_ARGS`.
That is where GPTQ/Marlin, bf16 dtype, chunk sizes, FP8 KV, CUDA graph batch
sizes, and memory fraction should be set for platform evaluation.

## Correctness eval

`eval_model.py` calls the already-started SGLang server through OpenAI
chat/completions.

Observed defaults in the official script:

- API base default: `http://127.0.0.1:30000`
- `max_seq_len`: `262144`
- CLI `--concurrency` default: `8`
- Generation max output: `max_out_len=65536`
- Sampling: `temperature=0.0`
- It sends stop strings collected from generation config eos IDs and tokenizer
  eos token.
- It scores `mcq` by extracting answer letters, and long-context tasks
  (`niah`, `cwe`, `fwe`, `qa`, `lcx`) by string containment/coverage.
- It writes `ori_accuracy` and normalized `overall_accuracy`, where normalized
  is capped at `avg_score / 80 * 100`.

Interpretation for our experiments:

- Correctness can have very long tails if the model fails to stop, because the
  cap is 65536 generated tokens.
- A long output is not automatically a wrong answer; scoring only checks the
  final text after `</think>` or the full prediction when no think tag exists.
- Long-tail correctness requests explain late `INFERENCING` wall time, but they
  do not explain a run that has not reached inference/service-ready.

## Speed eval

`bench_serving.sh` runs the official `python3 -m sglang.bench_serving` against
three optional speed datasets:

- `SPEED_DATA_S1`: S1, `--max-concurrency 1`
- `SPEED_DATA_S8`: S8, `--max-concurrency 8`
- `SPEED_DATA_SMAX`: Smax, no max-concurrency argument

For each provided dataset, it converts rows shaped like:

```json
{"question": "...", "model_response": "..."}
```

into a custom conversation with one user turn and one assistant turn. The
current SGLang `custom` dataset loader tokenizes the assistant turn and uses
that token count as `output_len`.

The benchmark command includes `--flush-cache`. For the SGLang OpenAI request
path, bench_serving uses `ignore_eos=true` by default unless
`--disable-ignore-eos` is passed.

Interpretation for our experiments:

- The speed benchmark is a fixed input/fixed output length throughput test.
- It does not depend on the model naturally emitting EOS.
- A correctness long-tail problem and a speed benchmark duration problem are
  related through decode throughput, but they are not the same failure mode.

## Meaning for the fp8kv platform hang

The user-reported platform status stayed around:

- `PREPARING`
- `DOWNLOADING` / "executing contestant submit preparation script"

The official docs place `prepare_env.sh` and optional `prepare_model.sh` before
service/eval. Therefore, if the UI label is literal, the primary suspect is
package setup or model preprocessing, not correctness long-tail output.

If the UI label also covers server startup until ready, then startup-time
suspects become relevant:

- FlashInfer SM120 JIT on a cold cache.
- Any submission script that deletes `~/.cache/flashinfer`, because deletion is
  quick but forces the next launch to rebuild kernels.
- CUDA graph capture only after the server starts loading. Local evidence for
  the current fp8_e4m3 + CUDA graph batch list showed capture itself took only
  about 2 seconds, so CUDA graph is not the first thing to disable.

Do not infer platform success from local tarball names. Only platform logs with
`SUCCESS` and a score should be treated as successful baselines.

## Practical checklist before next submit

1. Package must contain `prepare_env.sh`.
2. If using preprocessing, `prepare_model.sh` must accept exactly
   `--input <path> --output <path>`.
3. `prepare_env.sh` should export all server args through
   `SGLANG_SERVER_ARGS`.
4. Do not clear FlashInfer cache by default. Make cache rebuild opt-in via an
   explicit variable if needed.
5. Keep CUDA graph enabled unless a direct platform startup log shows graph
   capture is the actual blocker.
6. Treat `DOWNLOADING`/prepare-stage stalls separately from `INFERENCING`
   long-tail stalls.
