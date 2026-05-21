# Calibration gotchas (W4A16 on MiniCPM-SALA, May 2026 SOAR contest)

Hard-won lessons from the W4A16 quantization saga. Read this before
designing a new calibration recipe.

## TL;DR rules

1. **qzeros 0x88888888**, not 0x77777777. gptqmodel 7.0.0 + sym=True
   writes the wrong value; you MUST run `fix_qzeros_for_marlin()`.
2. **`truncation_side="left"`** for long-context calibration on SOAR.
   The default `"right"` keeps haystack filler and discards the question.
3. **Mind which task types your calibration set hits.** SOAR's tasks
   correlate strongly with length: short ≈ MCQ, long ≈ NIAH/QA/CWE.
   A "short prompts only" calib set is implicitly a "MCQ-only" calib set.
4. **`fwe` is a free credit on perf_public_set.** It's an easy task
   (find the most frequent word). Don't let 100% on fwe mask the fact
   that NIAH/QA are 23-30%.
5. **Repetition collapse is the dominant failure mode for current W4A16
   quants on SOAR.** Sampling with temperature=0 + max_tokens=65536
   gives the model 30+ minutes to dig its own grave. Mitigations:
   `repetition_penalty > 1.0`, or a tighter `max_tokens` ceiling.
6. **Local acc != platform acc.** The platform uses a hidden eval set
   that almost certainly has a harder task mix than perf_public_set.
   v21 local 49% → platform 0% is a 49-point gap, partially explained
   by hidden distribution differences.

---

## The qzeros encoding bug (resolved 2026-05-20)

gptqmodel 7.0.0 with `sym=True` targeting uint4b8 (Marlin's expected
format) writes every `.qzeros` tensor packed as `0x77777777`
(= 7 per 4-bit slot, signed int32 form `2004318071`). Marlin's dequant
formula `weight = (q_unsigned - qzero) * scale` expects qzero=8 (signed
int32 `-2004318072` for the packed form). With qzero=7, every weight
gets a per-channel `+1*scale` additive bias that accumulates through
every linear in every layer → garbage outputs → acc=0.

**Fix**: post-save, walk every safetensors shard and rewrite
`.qzeros` tensors whose unique value is `2004318071` to `-2004318072`.
Idempotent; safe to call on already-patched artifacts. Reference
implementation: `fix_qzeros_for_marlin()` in
`submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py`.

**Verification**: `python3 tools/inspect_quant_artifact.py --artifact
<quant_dir>` prints OK or BUG explicitly.

**Cost paid**: ~5 burned platform slots (v6-v17, each ~5 hours)
before identifying this.

---

## Truncation side matters for long-context calibration

`perf_public_set.jsonl` rows have:
- median length ~22K (mock) / ~30K (real) tokens
- p90 ~98K
- max ~128K

The **question, MCQ choices, NIAH answer pointer, etc. are all at the
END of the prompt**. The haystack filler is at the front.

HF tokenizer default is `truncation_side="right"`, which keeps the
FIRST `max_length` tokens and discards the tail. Applied to a 30K-token
row with max_length=4K, the model sees 4K of filler and Hessian is fit
to filler activations.

**Fix**: explicitly `tokenizer.truncation_side = "left"` before tokenizing
calibration prompts. This keeps the tail (where the task structure lives).

**Caveat** (newly discovered 2026-05-21): this fix alone did NOT solve
v21 acc=0 on platform. The repetition-collapse problem dominates.
But truncation_side="left" is still correct in principle and should
remain on by default.

---

## Chat template — uncertain effect

`apply_chat_template()` wraps a raw `question` string as `<用户>...<AI>...`
matching how SALA was instruction-tuned. We turned this on by default
and then later worried that platform eval feeds raw questions without
the wrapper, creating a distribution mismatch.

Exp B (chat template OFF) is in flight; its result will resolve this.
Current expectation: chat-template ON vs OFF will make ≤5 percentage
points difference because the actual problem (repetition collapse) is
not driven by token-distribution shift.

**Rule of thumb**: enable chat template if the eval pipeline preserves
the wrapper, disable if it strips. SOAR appears to strip but the cost
of mismatch is small relative to the quant quality issue.

---

## Task–length correlation on perf_public_set (verified
   2026-05-21 via tools/build_calib_set.py)

When you bucket the 150 perf_public_set rows by token length and look at
task distribution per bucket:

| bucket   | n  | task distribution                                       |
|----------|----|---------------------------------------------------------|
| short    | 27 | 97% mcq (everything else is rare NIAH single-fact)      |
| medium   | 52 | cwe / fwe / niah / qa balanced (~26% each)              |
| long     | 40 | cwe / fwe / niah / qa balanced (~25% each)              |
| super    | 11 | 64% niah, 36% qa (longest retrieval tasks)              |

**Consequence**: choosing a "short prompts only" calibration is
effectively choosing "calibrate for MCQ-pattern activations". Choosing
super-long calibration biases toward needle-retrieval. The desired
balance is in `medium` or `mixed`.

---

## perf_public_set is not a faithful proxy for platform eval

| signal                       | local (perf_public_set)              | platform                            |
|------------------------------|--------------------------------------|-------------------------------------|
| v18 + qzeros (30 sample)     | 63.33%                               | (not measured at this config)       |
| v21 (150 sample)             | 49.00%                               | acc_ori=0.0, final=0                |
| RTN-scalefix (older)         | 42                                   | 42                                  |

**Read**: when RTN got 42 both places, the eval sets were comparable.
When v21 jumped to 49 locally but 0 on platform, something about v21's
specific failure mode (repetition collapse on hard tasks) is fully
exposed on the platform's harder mix.

**Implication for tomorrow**: do not declare victory at 49%. Aim for
≥80% locally before another platform submission.

---

## Repetition collapse (the new discovery, 2026-05-21)

`tools/analyze_predictions.py` on v21 predictions reveals:

```
cwe failures (16 of 30):
  output_tokens = 65546 (max cap)
  prediction tail: "stall", "stall", "stall", "stall", "stall"...

niah failures (21 of 30):
  output_tokens = 65546 (max cap)
  prediction tail: "The sky is blue. The sun is yellow."*N

mcq failures (12 of 30):
  passed avg output_tokens = 9,392
  failed avg output_tokens = 21,020 (2x longer)
```

The model is not "answering wrong" — it's getting stuck in a token-level
loop and burning the entire 65K budget until generation hits max_tokens.
QA failures look different: short outputs (~110 tokens) suggest the
model gives up rather than loops.

**Hypotheses for tomorrow**:
1. The quant pushed the model just past the threshold where temperature=0
   sampling becomes unstable on long contexts. Either tighten max_tokens
   (cheap test) or add `repetition_penalty > 1.0`.
2. MLP-only quant might be the wrong split — the attention layers' BF16
   precision may be carrying the "stop tokens" signal that quantized MLPs
   then corrupt.
3. Calibration on tail-only prompts gave the MLPs an "answer-shaped" view
   of activations and they over-fit to long-output patterns.

Each hypothesis has a corresponding Exp in `experiments/PLAN.md`.

---

## fp16 sed-patch (background)

SALA's sparse attention backend (`minicpm_backend.py`,
`minicpm_sparse_utils.py`) hardcodes `torch.bfloat16` but Marlin GEMM
emits fp16. `prepare_env.sh` patches the backend at install time via:

```bash
sed -i 's/torch\.bfloat16/torch.float16/g' minicpm_backend.py
sed -i 's/"bfloat16"/"float16"/g' minicpm_backend.py
# (same on minicpm_sparse_utils.py)
```

**Failure mode (theoretical)**: if the bundled sglang sources don't
contain `torch.bfloat16` lines (e.g., upstream sglang was updated to a
fp16-tolerant version), sed silently no-ops and backend stays bf16 →
Marlin/sparse dtype mismatch → garbage.

**Verification**: `python3 tools/quant_config_validator.py --tarball X.tar.gz`
counts matched lines explicitly. As of v21, sed matches 7+3 lines.

---

## What hasn't worked

| Approach                                                | Outcome  | Lesson                                              |
|---------------------------------------------------------|----------|-----------------------------------------------------|
| RTN sym W4A16 (no Hessian)                              | acc=42   | RTN too lossy for SALA                              |
| GPTQ + qzeros bug                                       | acc=0    | qzeros must be 0x88888888                           |
| GPTQ + qzeros fix + chat-tpl ON + left-trunc + 8K (v21) | acc=49 / platform 0 | repetition collapse on long prompts |
| FP8 KV cache                                            | crash    | incompatible with MiniCPM sparse backend           |

## What might work (untested)

| Approach                                  | Cost           |
|-------------------------------------------|----------------|
| max_tokens=8192 ceiling (Exp C)           | ~12 min eval   |
| repetition_penalty=1.1 (Exp D)            | edit + ~40 min |
| multi-adaptive calib windowing (Exp E)    | ~1 hour        |
| group_size=64 / desc_act=True             | ~15 min quant  |
| full-attention quant (v22 tarball)        | already packed |
| AWQ via May 14 llmcompressor artifact     | unknown        |
