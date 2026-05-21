# Experiment runbook — 2026-05-21 (next-morning pickup)

> Read me first. Lists what's known, what's in flight, and the
> priority-ordered experiments to run today.

---

## What changed overnight

1. **v21 (MLP-only GPTQ, chat tpl ON + left-trunc + 8K + qzeros fix)
   local acc = 49.00%** on full 150 perf_public_set samples.
2. **v21 platform submission = `acc_ori=0.0, final_score=0`**.
3. **Diagnosis via `tools/analyze_predictions.py`**: failed samples ALL
   hit `output_tokens=65546` (the max_tokens cap) with the model in a
   **repetition loop** — e.g. cwe fails are `"stall", "stall",
   "stall"...` × N, niah fails are `"sky is blue. The sun is yellow..."`
   repeated. **The root cause of v21 is REPETITION COLLAPSE caused by
   quantization**, not by the chat-template / truncation choices we
   had targeted.
4. **fp16 sed-patch hypothesis ruled out** (via
   `tools/quant_config_validator.py`): v21 tarball's sed actually
   matches 7+3 lines in bundled sglang sources — the silent-noop story
   does not apply.
5. **Exp B (chat-template OFF + tail + 8K) was launched at 01:19** but
   stalled at 147/150 because the same repetition collapse keeps long-
   prompt samples generating for 30+ minutes per row. Watchdog snapshots
   every 5 min; check `outputs/<latest>/predictions.jsonl` for whether
   it completed.

## Per-task v21 numbers (the real story)

| task | n  | mean   | pass%  | failure mode                                                  |
|------|----|--------|--------|---------------------------------------------------------------|
| cwe  | 30 | 31.7%  | 46.7%  | partial credit; failures hit max_tokens with single-word loop |
| fwe  | 30 | 100.0% | 100%   | trivially easy (frequent-word extract); no diagnostic value   |
| mcq  | 30 | 60.0%  | 60%    | failures avg 21K out vs 9K passed → also looping in `<think>` |
| niah | 30 | 30.0%  | 30%    | failures hit max_tokens with haystack-phrase loop             |
| qa   | 30 | 23.3%  | 23.3%  | failures have short outputs (~108 tokens) — model gives up    |

**fwe is misleading**: 100% pass with avg 792 out_tokens is free credit
on an easy task. If we drop fwe, real perf is (31.7+60+30+23.3)/4 ≈ 36%.

The platform's hidden eval set probably does not have a fwe-equivalent
"easy" bucket, which would explain why local 49% ⇒ platform 0%.

---

## First thing to do when you wake up

```bash
cd /root/autodl-tmp/zyn/soar/sglang
git pull   # pull any watchdog snapshots

# 0) Did Exp B actually finish?
ls outputs/
# If outputs/20260521_011939/predictions.jsonl exists with 150 lines:
python3 tools/analyze_predictions.py \
    --input outputs/20260521_011939/predictions.jsonl \
    --output-md analysis_expB.md

# 1) Side-by-side compare with v21
python3 tools/diff_predictions.py \
    --baseline outputs/20260520_234226/predictions.jsonl \
    --candidate outputs/20260521_011939/predictions.jsonl \
    --output-md diff_v21_vs_expB.md
```

**Decision based on Exp B numbers**:
- If Exp B ≥ 60%: chat-template was the issue → pack v23 with chat
  template OFF and submit. (Note: today's quant variant already has
  `DISABLE_CHAT_TEMPLATE=1` baked in if you reuse the artifact.)
- If Exp B ≈ 49%: chat-template is NOT the issue. Move to Exp C/D below.
- If Exp B < 40%: chat-template was actively helping (unexpected); revert.

---

## Today's experiment ladder

Each box below is self-contained: hypothesis + command + expected +
decision. Run in priority order.

### Exp C — eval-only re-run with smaller max_tokens (TOP priority)

**Hypothesis**: Repetition collapse dominates. Lower the server's
generation ceiling so model can't burn 30+ minutes looping; total
wall time drops sharply, and failure patterns will show whether
collapse was hiding model-correct answers earlier in the output.

**Implementation note**: SGLang server-side `max_tokens` comes from the
eval_model.py sampling kwargs (currently `max_tokens=65536`). To override
without changing eval_model.py, set the per-request max_tokens via an
edit to `/root/autodl-fs/zyn/soar_toolkit/eval_model.py` (look for
`max_tokens` in `SGLang sampling kwargs`). Try 4096 or 8192.

```bash
screen -S expC

# Edit eval_model.py to set max_tokens=8192 (one-liner sed)
sed -i "s/'max_tokens': 65536/'max_tokens': 8192/" /root/autodl-fs/zyn/soar_toolkit/eval_model.py
# Verify the change
grep max_tokens /root/autodl-fs/zyn/soar_toolkit/eval_model.py

bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --skip-quant \
    --eval-data /root/autodl-tmp/zyn/soar/sglang/submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \
    --num-samples 150 \
    2>&1 | tee /root/autodl-fs/zyn/logs/expC_max8k_$(date +%Y%m%d_%H%M%S).log
```

**Expected**: total eval drops 39min → ~12min. acc roughly comparable
to v21 on the non-loop samples; loop samples either give up early
(score 0 like before) OR happen to land on the answer (some recovery).

**Decision**:
- Same 49% → quant model is genuinely collapsing; we need to fix quant
  quality. Move to Exp E.
- Significantly higher → quant outputs are OK on short windows; the loop
  is just an artifact of max_tokens being too generous.

### Exp D — eval-only with `repetition_penalty > 1.0`

**Hypothesis**: temperature=0 + greedy is maximally loop-prone. A small
repetition penalty breaks the cycle.

```bash
# Edit eval_model.py sampling kwargs to add repetition_penalty=1.1
# (look for the SGLang sampling kwargs dict)
# Then re-run eval-only as above.
```

**Expected**: cwe/niah/qa jump 10-30pp if loops are the problem. If
identical to v21, content quality is bad in the first place.

### Exp E — multi-adaptive calibration windowing

**Hypothesis**: tail-only calibration starves the Hessian of long-context
activation signal. multi-adaptive may both improve raw quant quality
AND reduce repetition collapse (by exposing the calibration to the
full prompt distribution).

```bash
screen -S expE
cd /root/autodl-tmp/zyn/soar/sglang

CALIB_WINDOW_MODE=multi-adaptive \
NUM_CALIB=300 \
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --force-requant \
    --eval-data /root/autodl-tmp/zyn/soar/sglang/submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \
    --num-samples 150 \
    2>&1 | tee /root/autodl-fs/zyn/logs/expE_multiadaptive_$(date +%Y%m%d_%H%M%S).log
```

Wall: quant ~20min (1.4× sample count) + eval ~40min ≈ 1h. Watchdog
catches everything.

### Exp F — alternative quant configs (last resort)

If Exp C-E all stuck at ~49%, the GPTQ recipe itself needs tuning. Try
in priority order; each requires a fresh quant (~14min):

| Tweak                                         | Why                                              |
|-----------------------------------------------|--------------------------------------------------|
| `--group-size 64` (default 128)               | Finer quant granularity, ~2× scales storage      |
| `--desc-act` (currently False)                | Activation-order column reordering; reduces error|
| `--bits 8`                                    | Less aggressive; trivially passes acc gate        |
| Full attention quant via v22 tarball          | Test if MLP-only is hurting                       |

Each Tweak: set the env vars / CLI args in `prepare_model.sh`, re-quant,
eval. Treat as A/B against v21.

### Exp G — try AWQ via the May 14 llmcompressor artifact

See `docs/awq_fallback_plan.md` for details.

---

## Mental model: what's the picture today?

```
                          v21 (acc=49% local, 0% platform)
                                       │
                                       │ repetition collapse on long prompts
                                       │ (analyze_predictions.py revealed this)
                                       ▼
              ┌────────────────────────┴─────────────────────────┐
              │                                                  │
       (quant quality issue)                           (sampling issue)
              │                                                  │
       Exp E / F: better calib                          Exp C: cap max_tokens
       Exp E / F: better config                         Exp D: repetition_penalty
              │                                                  │
              ▼                                                  ▼
     If still bad → Exp G (AWQ)                       If fixes → cheap win
```

## Resources

- `tools/analyze_predictions.py` — per-task pass/fail + failure mode
- `tools/diff_predictions.py` — A vs B transition matrix
- `tools/parse_experiment_logs.py` — bulk log → CSV/MD
- `tools/inspect_quant_artifact.py` — qzeros/scales sanity
- `tools/quant_config_validator.py` — pre-flight tarball check
- `tools/build_calib_set.py` — per-bucket calib sets
- `tools/calib_set_preview.py` — see what GPTQ sees
- `tools/pack_submission.py` — auto-pack v23/v24 etc.
- `scripts/list_all_quant_artifacts.sh` — server inventory

All tools have `--help`. Most have `--output-md`.

## Last-slot watchdog

If today's platform slot is unconsumed:
- `soar_bf16_baseline_match_submission_20260520_v4.tar.gz` is the
  zero-risk fallback (should clear correctness ≈ 19.13 baseline).
- If Exp B or any later run breaks 80% locally, pack that variant via
  `tools/pack_submission.py` and submit instead.
- **Don't burn a slot on anything < 80%** — it'll score 0 and consume
  the slot anyway.
