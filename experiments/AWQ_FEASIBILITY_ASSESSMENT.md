# AWQ feasibility assessment — Stream B-lite output

> 30-min CPU-only research: should we invest in building an AWQ
> (compressed-tensors) submission pipeline as a backup to GPTQ?

**Verdict**: 🟡 **CONDITIONAL GO**. Not NO-GO as initially feared, but
not free-fire either. Build only if v23 (GPTQ + H1+H4) platform
returns acc < 20, indicating systemic GPTQ-pipeline issues that AWQ
might sidestep. Detailed reasoning below.

## What we found

### Pro: SGLang has W4A16 compressed-tensors support

`python/sglang/srt/layers/quantization/compressed_tensors/schemes/`
contains `compressed_tensors_wNa16.py` — the scheme that handles
W4A16 / W8A16 / WNA16 generally. It's NOT just w8a8_fp8 (the
"well-tested" path per their README); wNa16 is registered and used.

### Pro: Champion team used this path

`quantize_to_w4a16.py` (repo root, line 5-6 docstring):

> "Champion's approach (SOAR Week 4 - 智算一队):
> - GPTQ algorithm (Hessian-based weight quantization)
> - W4A16 scheme (4-bit weights, 16-bit activations)
> - Marlin kernel at inference (~4x bandwidth savings)"

So an actual SOAR competition team used the
llmcompressor → compressed-tensors → SGLang path and won.

### Pro: May 14 artifact already on disk

`/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/` — produced via the
above script with default wikitext calibration. Never benchmarked,
but DOES NOT need to be re-quantized (no GPU time required to test).

### Pro: docs/awq_fallback_plan.md exists with step-by-step plan

Already-thought-through fallback. 3 stages: (1) quick load smoke
test, (2) eval against perf_public_set, (3) repack with SOAR-tuned
calibration if smoke test passes.

### Caveat: SGLang's own README says "only w8a8_fp8 well-tested"

```
For practical purposes, we have only applied the compressed_tensors
format of `w8a8_fp8`. If you have requirements for other formats,
you can submit an issue through this [link]...
```

This is the SGLang upstream's caveat. The wNa16 code IS there, but
production-grade testing has only been done on w8a8_fp8. The
champion team's success suggests wNa16 works in practice for at
least one OpenBMB-flavored model.

### Caveat: May 14 calibration used wikitext, not perf_public_set

The May 14 artifact was calibrated with generic wikitext — fine for
language modeling but NOT optimized for the SOAR eval distribution
(mcq / niah / qa / fwe / cwe). Local acc on perf_public_set could
be substantially lower than what a SOAR-tuned recalibration would
produce.

### Caveat: compressed-tensors path is structurally different from gptq_marlin

Our 7 preflight canaries (qzeros fix, copy_runtime_assets H4 fix,
gptqmodel pin, DIAGNOSTIC blocks, chunked-prefill check, etc.) are
ALL gptq_marlin-specific. They don't apply to compressed-tensors.
A v25-awq variant would need its own preflight canaries OR an
explicit "skip GPTQ canaries" flag.

### Caveat: AWQ-vs-GPTQ in quantize_to_w4a16.py is slightly misleading

Despite the doc string saying "AWQ" elsewhere in this repo,
`quantize_to_w4a16.py` actually uses **GPTQ algorithm via
llm-compressor** (not the original AWQ scaling-search algorithm).
The artifact is GPTQ-Hessian quantized but written in
compressed-tensors format. Distinct from our submission_*/ pipeline
which uses gptqmodel + Marlin format.

This is important: if our v23 fails because of GPTQ-pipeline issues,
the compressed-tensors path is the SAME algorithm written in a
different file format and read by a different SGLang loader. The
weights are quantized identically.

If our v23 fails because of FORMAT issues (qzeros encoding,
tokenizer drift, gptq_marlin loader quirks), then compressed-tensors
IS a different path. Likely sidesteps the issue.

If our v23 fails because GPTQ ALGORITHM itself is too lossy for
SALA + perf_public_set distribution, compressed-tensors won't help
— it's the same algorithm.

## Decision tree

```
v23 platform result:
  acc >= 30        → no need for AWQ. Push v24-perf + v25_calib_*.
  10 <= acc < 30   → marginal. Try v24_no_dtype_key / v24_pin_transformers
                     first (cheap, same algorithm). Hold AWQ as last resort.
  0 <= acc < 10    → systemic GPTQ-pipeline issue. AWQ likely helps if
                     it's a format/loader issue, doesn't help if it's
                     the algorithm. Either way, run docs/awq_fallback_plan.md
                     Step 1 (10-min smoke test) to find out.
  pipeline crashed → fix v23 first; AWQ is a longer detour.
```

## If we GO: minimum effort plan

Per `docs/awq_fallback_plan.md`:

### Step 1: smoke test (10 min, requires GPU)

Run sglang against `/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/`
directly. Does it load? Does it generate text?

```bash
python3 -m sglang.launch_server \
    --model-path /root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16 \
    --trust-remote-code \
    --quantization compressed-tensors \
    --port 31112 &
# wait for /health, send one prompt, check output
```

- If load fails / output is garbage → AWQ path on SGLang is broken
  for SALA. NO-GO.
- If output is coherent → continue to step 2.

### Step 2: eval against perf_public_set (40 min, requires GPU)

Use `scripts/local_eval.sh --variant <some-awq-variant>
--skip-quant --eval-data perf_public_set.jsonl --num-samples 150`.

- If `acc >= 49` (the GPTQ-Marlin v21 baseline) → AWQ is at least
  competitive. Build a submission pipeline.
- If `acc < 30` → wikitext calibration is too generic. Either
  re-quantize with SOAR-tuned calib (Step 3) or abandon.

### Step 3: re-quantize with SOAR-tuned calibration (3 hours)

Modify `quantize_to_w4a16.py` to use `perf_public_set.jsonl` as
calibration source. Re-quant. Re-eval.

If acc rises substantially over Step 2's wikitext-calibrated number,
build a submission pipeline (Step 4).

### Step 4: build submission pipeline (2-3 hours)

`submission_awq_compressed_tensors/` variant dir with:
- `prepare_env.sh`: install llmcompressor + compressed-tensors deps
- `prepare_model.sh`: invoke `quantize_to_w4a16.py` with SOAR-tuned
  calibration on the platform (re-quantize at platform-install time,
  similar to gptqmodel variant)
- SGLang launch args: `--quantization compressed-tensors` instead of
  `--quantization gptq_marlin`
- Custom preflight canaries (the gptq-marlin ones don't apply)

## Time / risk budget if we DO build it

- Smoke test + eval: 50 min total GPU time (acceptable nightly)
- Submission pipeline build: 5-6 hours (substantial)
- One platform slot to validate (5h wall, slot consumed regardless of result)

Total cost: ~10 hours work + 1 platform slot. Worth it ONLY if
v23-style GPTQ-Marlin path is structurally broken on the platform
(acc < 10 even with H1+H4 fixes).

## Conclusion

**Stream B-lite verdict**: AWQ is structurally feasible (SGLang
supports it, prior team used it, artifact already on disk), but
the cost-vs-benefit is poor unless v23 fails badly. Do NOT
preemptively build the pipeline.

**Action**: wait for v23 platform result. If `acc < 10`, kick off
docs/awq_fallback_plan.md Step 1 (10-min smoke test). Decide
further from there.

This document supersedes my earlier "AWQ likely incompatible"
worry — the wNa16 scheme is real, the champion-team precedent
exists, and the artifact is already on disk. The remaining
caveats are real but tractable.

## What we did NOT verify (would need GPU or platform)

- Whether the May 14 artifact's tokenizer is base BF16's (H4 fix
  applicable here too) or GPTQModel-resaved (different layout)
- Whether SGLang's compressed-tensors loader on the platform has
  any SALA-specific quirks (the SALA modeling code is loaded via
  trust_remote_code so the loader may or may not honor it)
- Whether the Marlin GEMM kernel is used under compressed-tensors
  (the docstring says yes; SGLang code might dispatch differently)

These are platform-runtime questions, not code-inspection questions.
The smoke test in Step 1 answers them.
