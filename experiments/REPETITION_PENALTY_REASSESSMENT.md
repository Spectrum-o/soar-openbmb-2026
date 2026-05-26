# Repetition-Penalty Recommendation — Reassessment (2026-05-26)

Reassesses `experiments/ACC_OPTIMIZATION_ROADMAP.md` Tier 1 lever
(`repetition_penalty=1.05` + `max_tokens=16384`) in light of the
current fp8kv+cudagraph run (`/root/autodl-tmp/zyn/eval_runs/
fp8kv_cg_e4m3_20260526_180812/`, 150 samples, `ori_accuracy=83.13`).

The roadmap's "+5–10pp" estimate is **not directly applicable to the
current state** and the 1.05 / 16384 numbers should not be adopted
without first verifying the failure-mode assumption on current
predictions.

## What the roadmap claims

ACC_OPTIMIZATION_ROADMAP.md:97-104 lists:

| Lever | Mechanism | Expected gain | Effort |
|---|---|---|---|
| `repetition_penalty=1.05` | Break repetition loops on cwe/niah | **+5–10pp** | 30 min |
| `max_tokens` cap (65536 → 16384) | Truncate before model digs grave | +3-5pp | 5 min |

Cites `project_v21_per_task_breakdown` ("cwe/niah failures as
quant-induced repetition loops").

## Why the numbers don't carry over to the current run

### 1. The evidence is from v21 era (local acc ~49)

`docs/calibration_gotchas.md:151-181` quotes `analyze_predictions.py`
on **v21** predictions (2026-05-21):

```
cwe failures (16 of 30):
  output_tokens = 65546 (max cap)
  prediction tail: "stall", "stall", "stall", "stall", "stall"...

niah failures (21 of 30):
  output_tokens = 65546 (max cap)
  prediction tail: "The sky is blue. The sun is yellow."*N
```

v21's local was ~49. The current fp8kv+cudagraph run is **83.13**.
Between v21 and now we landed: qzeros hardening (2026-05-21),
tokenizer overwrite fix (H4, 2026-05-22), BF16 native (drops the
fp16 sed-patch — see `project_fp16_sedpatch_killer`), chunk32k_safe,
and fp8kv Path Y (commit `203df1d59`) + cudagraph fix (commit
`9ec5708f9`). The repetition collapse may already be largely fixed.

**Premise the roadmap depends on — "cwe/niah are still hitting
65546-token repetition tails" — is unverified on the current run.**

### 2. The "+5-10pp" estimate was sized for a 49 → 55 jump

Roadmap quantified the lift relative to v21's failure rate. From a
base of 83.13, the same failure mode has a much smaller residual
pool to recover. The honest expected lift is **bounded by whatever
remains of the v21-style repetition pattern in current predictions
— which is the very thing we have not measured.**

### 3. `max_tokens=16384` is an S1 lever, not an acc lever

Truncating earlier saves wall-clock when the model loops, but:

- If the model is looping on garbage, capping at 16K vs 65K does
  not change the answer correctness — it just frees the GPU sooner.
- If the model needs > 16K tokens to emit a correct long CWE list,
  capping at 16K **kills a correct answer**. The roadmap's "+3-5pp"
  for the cap is not defensible as an acc gain.

The cap belongs in a latency-tuning phase, not an accuracy phase.

### 4. `1.05` is likely too weak to break tight loops

repetition_penalty (SGLang/vLLM impl): positive logits divided by
penalty. Under greedy decoding (`temperature=0`), if "stall"'s logit
is 10 and the next-best token is 8, then `10 / 1.05 = 9.52 > 8` —
the loop survives. Production fixes commonly use 1.10–1.30.

1.05 reads like "I'm worried about hurting other tasks so I'll be
gentle" — but at 1.05 it's likely too gentle to break the loop AND
too gentle to hurt anything else. A two-sided null.

### 5. Plumbing not verified

Roadmap line 287 itself asks: *"Does SOAR Toolkit's `eval_model.py`
accept `repetition_penalty` in request payload?"* — this is an
**open question**, not a resolved precondition. Prior research on
`SOAR-Toolkit/eval_model.py` shows the payload contains only
`{model, messages, temperature, max_tokens, stop}`. So
`repetition_penalty` would need to be injected via SGLang's
server-side default sampling params (e.g., `--default-sampling-params`
or equivalent), which has not been validated against this fork.

## What to do first

Before committing 30 min of platform slot time to a rep_penalty
sweep, **inspect the current predictions**. Tools already exist:

```bash
# v21 used these — re-run on the CURRENT fp8kv predictions
python3 tools/analyze_predictions.py \
    /root/autodl-tmp/zyn/eval_runs/fp8kv_cg_e4m3_20260526_180812/predictions.jsonl

python3 tools/repetition_analyzer.py \
    /root/autodl-tmp/zyn/eval_runs/fp8kv_cg_e4m3_20260526_180812/predictions.jsonl
```

Answer three questions before tuning anything:

1. **Are CWE failures still "in repetition", or now "missing
   keywords from a correct-shape output"?** Different fix paths.
2. **What is the `output_tokens` distribution on failed samples?**
   If most are far below the 65536 cap, no loop is occurring and
   `max_tokens=16384` is either inert or harmful.
3. **Are MCQ failures "wrong letter" (quant B↔D bias) or "no
   letter / overlong"?** B↔D bias is **not** addressable by
   sampling fixes; it requires more accurate quantization
   (group_size 64, multi-adaptive calib, AWQ).

## Decision tree from the inspection result

| If predictions show… | Action |
|---|---|
| CWE failures still emit 65K-token "stall" tails | rep_penalty IS the lever. Sweep 1.10 / 1.15 / 1.20, not 1.05. Also verify SGLang default-sampling-params injection works server-side. |
| CWE failures are short, content-wrong (no loop) | rep_penalty inert. Pivot to quant-precision levers (group_size 64, multi-adaptive calib, llm-compressor 5/14 artifact). |
| MCQ failures cluster on B↔D answer letter | Pure quant-precision problem. group_size 64 / longer calib / AWQ. Sampling fixes do nothing. |
| MCQ failures are overlong / no-letter outputs | Sampling-stability problem. rep_penalty + maybe stronger `stop` strings. |

The inspection is < 10 min of work and the cost of running it before
tuning is far lower than the cost of spending a 5h platform slot on
an inert change.

## Bottom line

The roadmap's Tier 1 recommendation is not wrong as a hypothesis —
it just operationalizes a 2026-05-21 failure mode that may have
been largely fixed in the intervening commits. With only one day of
competition time left, the order must be:

1. **Inspect current predictions** (5–10 min).
2. **Choose lever based on observed failure distribution** (not on
   inherited assumption).
3. **If sampling is the answer**, sweep 1.10 / 1.15 / 1.20 (not the
   pre-baked 1.05) and validate plumbing first.
4. **If quantization precision is the answer**, the `g64` / multi-
   adaptive-calib / llm-compressor-5/14 variants are pre-staged.

The decision belongs downstream of the inspection, not upstream.
