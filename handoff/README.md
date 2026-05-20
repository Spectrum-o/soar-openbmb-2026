# SOAR W4A16 acc=0 — Handoff (2026-05-20 evening snapshot)

**Read this if you are picking up the SOAR W4A16 quantization work.**
The current open question is: **why does every GPTQ-based submission score
acc=0, while a naive RTN script scores 42?** Below is everything needed
to continue without re-deriving the failure history.

---

## 1. TL;DR — the one question we are trying to answer

```
RTN (hand-written symmetric round, 9 Linears)        →  acc = 42
GPTQModel v17 (full-attn + MLP, 7 Linears)           →  acc = 0
GPTQModel v18/v19 (MLP-only, 3 Linears)              →  acc = 0
```

GPTQ should be **at least as good** as RTN at the same bit-width. The
fact that ALL GPTQModel variants collapse to 0 — even MLP-only —
means the regression is NOT in "what we quantize".

**Leading hypothesis** (see §5): the safetensors layout that GPTQModel
writes (specifically `qzeros`) does not match what SGLang's `gptq_marlin`
loader expects for symmetric uint4b8. RTN writes `qzero = 8` (the bias
that makes dequant `(q_unsigned - 8) * scale` symmetric); GPTQModel
may be writing `0`, `7`, or something else, silently corrupting every
weight after dequant.

**Next concrete action** (see §6): produce a GPTQModel-quantized model
locally on AutoDL, run `handoff/diagnose_qzeros.py` to read its
`qzeros` values, compare to the expected 8.

---

## 2. Server state (AutoDL, as of 2026-05-20 evening)

### 2.1 Filesystem inventory

```
/root/autodl-fs/
├── archive_before_kvfp8_20260519_185759/   # archived KV-cache fp8 attempt
├── models/                                 # SHARED, read-only by convention
│   ├── OpenBMB/MiniCPM-SALA/               # stock BF16 base model (~18 GB)
│   └── MiniCPM-SALA-GPTQ-W4A16*/           # OTHER labmates' work — DO NOT REUSE
├── wanghan_eval_backup/                    # labmate's backup, ignore
├── ziqian*/                                # labmate dirs, ignore
└── zyn/                                    # user's persistent space
    ├── bootstrap_gpu.sh
    ├── logs/
    ├── models/
    │   └── MiniCPM-SALA-W4A16/             # ONLY ONE quant artifact: 5/14 llmcompressor
    ├── requirements_known_good_20260514_1829.txt
    ├── soar_deploy_key                     # github deploy key
    └── soar_launcher.sh

/root/autodl-tmp/zyn/sglang/                # repo on fast disk
└── sglang_minicpm_sala_env/                # venv installed 2026-05-14
                                            # NOTE: venv's sglang is the 5/14
                                            # version → no kwargs.pop guard
                                            # in MiniCPMHybridConfig, no
                                            # torchao early-return.
```

### 2.2 Reproducibility — the one quant artifact that exists on disk

```
/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/    (~5.4 GB, dated 2026-05-14)
```

- **Producer**: llmcompressor (not GPTQModel, not RTN). `recipe.yaml`:
  ```yaml
  GPTQModifier:
    targets: [Linear]       # ALL Linears (incl. o_gate, z_proj)
    ignore: [lm_head]
    scheme: W4A16
    actorder: static        # = GPTQ desc_act=True (Marlin can't use this!)
    block_size: 128
  ```
- **Format**: `compressed-tensors` / `pack-quantized` (NOT gptq_marlin format).
- **Config preserved**: `auto_map`, `scale_emb=12`, `scale_depth=1.4`,
  `dim_model_base=256`, `mixer_types`, `sparse_config`, `attn_use_output_gate`,
  `qk_norm`. All SALA-specific fields survived `save_pretrained`.
- **Acc on platform**: UNKNOWN. SUBMISSIONS.md row for 2026-05-14 says
  "earliest, superseded", no `acc_ori` recorded.
- **To run it**: use `--quantization compressed-tensors --dtype bfloat16`
  (NOT gptq_marlin, NOT float16). Different code path from v17/v18/v19.

### 2.3 Git state on AutoDL

Pre-2026-05-20-evening, AutoDL was at the ~May-14 commit of the local
repo. After the user runs:
```bash
cd /root/autodl-tmp/zyn/sglang && git pull origin quant/w4a16
```
AutoDL has the recent commits including:
- `submission_gptq_v17_minconfig/` — the variant that reverts v17's
  config rewrites (see §5.3)
- `submission_gptqmodel_calib_w4a16/` updated to MLP-only (v18/v19's source)
- `scripts/local_eval.sh` updated to auto-detect variant entrypoints
- `python/sglang/srt/configs/minicpm.py` — `kwargs.pop` guard for derived
  @properties
- `python/sglang/srt/layers/torchao_utils.py` — early return guard

**Important**: AutoDL's venv-installed sglang is still the 5/14 version
and does NOT have the two source-level patches. Either re-install editable:
```bash
source sglang_minicpm_sala_env/bin/activate
pip install --no-deps -e python/
```
or any local-run experiment will hit the unpatched code path.

---

## 3. Source of truth files (read these next, in order)

| File | What it gives you |
|---|---|
| `SUBMISSIONS.md` (repo root) | Chronological log of all 17+ failed tarballs with status emoji / acc_ori / root cause. **THE primary log.** |
| `W4A16_README.md` (repo root) | Champion's notes summary + env setup walk-through |
| `submission_gptqmodel_calib_w4a16/README_SUBMISSION.md` | MLP-only v18/v19 rationale |
| `submission_gptq_v17_minconfig/README_SUBMISSION.md` | The variant that isolates v17's config-rewrite diff (built but never submitted yet) |
| `submission_rtn_sym_w4a16/quantize_gptq_rtn_sym.py` | **The acc=42 baseline.** Reference for what a "working" safetensors layout looks like. |
| `scripts/local_eval.sh` | The local quant → serve → eval pipeline (no platform slot needed) |

---

## 4. Hypotheses tested and **refuted**

These are dead. Do not spend time on them again.

### 4.1 H1 — removing `auto_map.AutoConfig` loses `scale_emb`

**REFUTED by source code reading.** SGLang's `MiniCPMHybridConfig.__init__`
accepts `**kwargs` (line 54) and calls `super().__init__(**kwargs)` which
goes to `PretrainedConfig.__init__`, which `setattr`s every unknown
field onto self. `scale_emb=12` survives.

See `python/sglang/srt/configs/minicpm.py:20-76`.

### 4.2 H2 — writing `has_sparse_attention=True` collides with `@property`

**REFUTED.** Lines 60-68 of the same file explicitly pop the derived
property names from kwargs BEFORE `super().__init__`:
```python
for key in ("mamba2_cache_params", "full_attention_layer_ids",
            "has_sparse_attention", "has_lightning_layers",
            "sparse_layer_ids", "lightning_layer_ids"):
    kwargs.pop(key, None)
```
So v17 writing the field statically was a no-op.

### 4.3 H3 — quantization tools clobber SALA-specific config fields

**REFUTED.** The May 14 llmcompressor product (see §2.2) preserved every
SALA field while NOT removing auto_map. So at least with auto_map kept,
the save/load roundtrip is clean.

---

## 5. Active hypothesis: GPTQModel safetensors layout mismatch

### 5.1 The decisive evidence

```
RTN (hand-written packing, all 9 Linears) →  acc = 42
GPTQModel MLP-only (3 Linears)            →  acc = 0
```

If the cause were "what is quantized", MLP-only should be the **safest**
GPTQ variant (only feed-forward, no attention). It still scores 0. So
the regression is not "which weights" but "how they are encoded /
loaded".

### 5.2 The specific suspect: `qzeros` encoding

RTN's `quantize_gptq_rtn_sym.py` writes:
- `qzeros` = `int32`, every 4-bit slot = **8** (the symmetric bias)
- `g_idx` = `[i // group_size for i in range(in_features)]`

SGLang's `gptq_marlin` dequant for `sym=True, bits=4` is:
```
weight = (q_unsigned - qzero) * scale
       = (q_unsigned - 8)     * scale     when qzero == 8
       ∈ [-8, +7] * scale                 — symmetric
```

If GPTQModel writes `qzero = 0`, dequant gives `q_unsigned * scale ∈ [0, 15] * scale` —
all weights become non-negative, the model output collapses. The model
still serves, requests still return tokens, but every response is garbage.
**This is exactly the v17/v18/v19 failure shape.**

Other suspects in the same layout family:
- `g_idx` might be all zeros (some GPTQModel paths emit that when `desc_act=False`)
- `scales` might be saved as bfloat16 instead of float16
- `qweight` packing order (column-major vs row-major) — Marlin expects a
  specific pack order

### 5.3 What the v17-minconfig variant tests

`submission_gptq_v17_minconfig/` reverts v17's three config-rewrite
changes back to RTN's minimal pattern (matches RTN-scalefix), keeps
everything else identical to v17 including the q/k/v/o + MLP module set.
**If acc is still 0 after this variant runs**, that confirms config
rewrites were never the issue and the layout mismatch is the answer.

This variant has NOT been submitted to the platform yet. **Do not submit
it before running the qzeros diagnostic in §6** — that is cheaper and
gives the same answer without burning a slot.

---

## 6. Next action — qzeros diagnostic

### 6.1 Goal

Produce a GPTQModel-quantized model locally on AutoDL, inspect its
`qzeros` values, compare with the expected `8` (or its packed form
`0x88888888 = 2290649224`).

### 6.2 Steps

```bash
# 0. On AutoDL, pull latest code
cd /root/autodl-tmp/zyn/sglang
git pull origin quant/w4a16
source sglang_minicpm_sala_env/bin/activate

# Make sure venv's sglang has the recent source-level patches.
pip install --no-deps -e python/

# 1. Verify fp16 patch state
bash scripts/fp16_patch.sh status
# If not patched: bash scripts/fp16_patch.sh apply

# 2. Run GPTQModel quantization (MLP-only is fastest, ~30-60 min).
#    --no-eval skips the bench_serving step we don't need for diagnosis.
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --no-eval \
    --num-samples 1

# Output lands at /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized/

# 3. Run the diagnostic
python3 handoff/diagnose_qzeros.py \
    /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized
```

### 6.3 What to look for

In the output for `model.layers.0.mlp.gate_proj`:

```
qzeros  int32  (out, in//8)  unique[:6]=[2290649224]   ← GOOD (every 4-bit slot = 8)
qzeros  int32  (out, in//8)  unique[:6]=[0]            ← BAD (every slot = 0 → bias bug)
qzeros  int32  (out, in//8)  unique[:6]=[2004318071]   ← BAD (every slot = 7 — off-by-one)
qzeros  int32  (out, in//8)  unique[:6]=[mixed values] ← UNEXPECTED, investigate
```

### 6.4 How to interpret

| Result | Conclusion | Next step |
|---|---|---|
| qzeros packed = `2290649224` (=8 per slot) | Layout is correct; bug is elsewhere (calibration data, desc_act, etc.) | Run `local_eval.sh` with full eval (`--num-samples 50`) to get an acc number for MLP-only variant. If still 0, dig deeper into calibration. |
| qzeros = 0 or 7 | **Root cause found.** Patch the GPTQModel save (or post-process safetensors) to write 8. | Add a post-save patch step in `quantize_gptqmodel_w4a16.py` that rewrites qzeros to `0x88888888`. Submit v20. |
| qzeros mixed | GPTQModel is using asymmetric encoding despite sym=True | Force sym=True at all layers; check GPTQModel version interaction |

---

## 7. Slot economy (read before submitting anything)

- **Daily submission cap**: 3 slots / day
- **Slot consumed**: 5h pipeline timeout (⏰), or full pipeline with acc=0 (🔴)
- **Slot NOT consumed**: prepare_env or prepare_model exits < 60s (⛔)
- **Today's status**: v17 + v18 + v19 already burned all 3 slots for 2026-05-20.
  Next submissions are 2026-05-21 onward.

So there is **no rush** to submit anything tonight. The diagnostic in §6
takes ~1h, gives a precise answer, and burns zero slots.

---

## 8. Files in this handoff folder

| File | What |
|---|---|
| `README.md` (you are here) | Single-file entry point |
| `diagnose_qzeros.py` | Standalone safetensors inspector for §6. No GPU needed. |

---

## 9. User context (for non-human readers)

The user (zyn) is competing solo in SOAR 2026 — OpenBMB's inference
optimization competition on MiniCPM-SALA (9B params, 1M context, RTX
PRO 6000 96GB target). Shares the AutoDL instance with labmates but
their quant artifacts (`/root/autodl-fs/models/MiniCPM-SALA-GPTQ-W4A16*`)
should NOT be reused.

The user has read the SGLang paper but is newer to the codebase, and
prefers concise / direct technical answers. They've been working this
problem ~6 days; the v17 acc=0 result is the current blocker.

Champion route (per "SOAR 周冠军笔记 04") is W4A16 GPTQ + Marlin with
calibration matched to eval task distribution, NOT MLP-only. The
intention is to get there; MLP-only was a fallback safety variant.
