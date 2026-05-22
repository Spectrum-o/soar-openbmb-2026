# Quantization Strategy Research Summary (post-1849 breakthrough)

> Written 2026-05-22 evening. Synthesizes findings from 4 parallel research agents
> exploring next-step optimization paths after 1849 platform result
> (acc=58.61, acc_ori=46.89, final_score=0).
>
> **For server-side Claude**: this is your "what to try next" reference. It
> ranks techniques by ROI and resolves a conflict between agents about whether
> MLP-only is correct for hybrid attention models.

---

## TL;DR — 5 most impactful changes (in order)

| # | Change | Expected gain | Effort | Source |
|---|---|---|---|---|
| 1 | **Switch sym=True → sym=False + actorder="weight"** | +1-3pp (production standard) | LOW (config change) | Agent 4 (RedHatAI recipes) |
| 2 | **Bump calibration: 256 → 1024 samples + dampening_frac 0.01 → 0.1** | +1-3pp | LOW (config change) | Agent 4 (RedHatAI Llama-3.1 recipe) |
| 3 | **Test 5/14 llm-compressor artifact locally** | unknown, but free | ZERO (artifact exists, no re-quant) | already on parallel branch |
| 4 | **Skip first-2 + last-2 transformer blocks from W4A16** | +0.5-2pp | LOW (4 lines dynamic config) | Agent 1 (APTQ + Jamba evidence) |
| 5 | **Switch tool from GPTQModel to llm-compressor (AWQ recipe)** | +3-8pp from method swap | MEDIUM (1-1.5 days integration) | Agents 2 + 4 (OpenBMB ships AWQ for MiniCPM-V) |

**Stacked (with ~60% capture efficiency): +10-20pp realistic. From 46.89 → ~57-67 raw acc.** Probably NOT enough to clear 80% gate alone, but unlocks the next phase.

---

## Key research findings

### Agent 1: Mixed-precision for hybrid attention (Jamba/Mamba lineage)

- **MLP-only is the safe recipe for hybrid models.** Jamba-1.5 (ExpertsInt8) and Quamba2 converge on "quantize FFN/MLP, leave the sequence mixer alone" — exactly what we do.
- **Last recurrent layers are activation hotspots.** Jamba team flagged the output of the last Mamba layers as FP16-overflow zones. Strongly suggests keeping the **last** transformer block(s) in BF16.
- **First/last-N skip is supported in llm-compressor.** Default MoE recipe ignores `model.layers.0.*`, `model.layers.1.*`, `model.layers.2.*`. APTQ confirms early/late layers carry higher sensitivity.
- **GPTQModel has first-class per-layer `dynamic` config** via regex + `-:` prefix for skips and per-module bits/group_size overrides.
- **No public W4A16 recipe for Gated DeltaRule / Lightning Attention.** We're on the frontier here.

**Verdict on mixed precision**: MLP-only stays. Add **first-2 + last-2 layer skip** (low cost, +0.5-2pp). Sensitivity-tiered bits (W8 on lightning-adjacent MLP) is highest-leverage but **needs Marlin mixed-bit verification** before sinking a calib run.

### Agent 2: AWQ vs GPTQ for instruction-tuned long-context

- **AWQ typically wins +1-3pp** on instruction-tuned models, with larger margins on ARC-c and GSM8K. Comprehensive study: arxiv 2409.11055.
- **Our model is exactly AWQ's target regime**: scale_emb=12 + scale_depth=1.4 → guaranteed outlier channels in early layers. AWQ's per-channel scaling protects those.
- **OpenBMB ships AWQ (not GPTQ) for MiniCPM-V 4.5** (Qwen3-8B backbone, 8-9B size). Official recipe: zero_point=True, q_group_size=128, w_bit=4, GEMM, bfloat16.
- **W4A16 long-context degrades sharply**: 99.5% recovery at ≤32K context, drops to 85% at 128K (8B model) on RULER. Our eval is retrieval-heavy (NIAH, count-word-entries) — worst case for 4-bit.
- **llm-compressor AWQ default**: `W4A16_ASYM`, 512 samples × 2048 tokens.

**Verdict on AWQ**: **Switch to AWQ.** Three converging reasons: (a) MiniCPM-style architecture creates AWQ's target outliers; (b) OpenBMB itself uses AWQ for MiniCPM-V; (c) we're stuck at 46.89 against an 80% gate — need a structural change, not another hyperparam pass. Expected delta: **+3-8pp from method swap alone**.

### Agent 3: Rotation methods (QuaRot / SpinQuant / QuIP)

- **W4A16 gain from rotation is modest (+2-5pp).** Rotation methods primarily target W4A4 where activation outliers kill accuracy; at W4A16 the gain is small.
- **MambaQuant warning**: QuaRot drops 21% acc on Vim-T (Mamba family) at W8A8. Mamba's parallel scan amplifies outliers and Hadamard's variance assumptions break. **SALA's Lightning Attention is in this risky class.**
- **R1+R2 offline rotations are baked into weights at quant time**, zero runtime cost, compatible with GPTQ-Marlin. R3/R4 online rotations need a Hadamard kernel that **SGLang does not currently have**.
- **llm-compressor v0.7/0.8 has `SpinQuantModifier`** as a drop-in (~5 lines added to oneshot script).
- **SGLang's compressed-tensors adapter is mostly w8a8_fp8 only** (issue #2871). W4A16 + transform metadata is not a tested path.

**Verdict on rotation**: **Defer, do not skip.** 1-day spike with R1+R2 only is worth doing AFTER cheap wins are exhausted. Don't sink a week into custom R-matrix mapping for SALA's hybrid attention.

### Agent 4: Production W4A16 recipes (RedHatAI / openbmb)

- **Production canonical recipe** (Llama-3.1-8B-Instruct, Qwen3-8B/32B):
  ```python
  GPTQModifier(
      ignore=["lm_head"],
      sequential_targets=["<YourDecoderLayer>"],
      targets="Linear",
      scheme="W4A16",                # group_size=128
      dampening_frac=0.01-0.1,
      # actorder="weight" default since v0.8
  )
  oneshot(model, dataset, recipe, max_seq_length=8192, num_calibration_samples=1024)
  ```
- **Group size**: g128 sym for ≥4B; drop to g64 asym for smaller or struggling models.
- **Mixed precision in production**: NO. RedHatAI quantizes every Linear in decoder layers, only `lm_head` is skipped. **No first/last layer BF16 in any shipped recipe.**
- **Accuracy retention on 8B instruction models**: Llama-3.1-8B MMLU recovery 97.9%, GSM8K 100.2%, HumanEval 99.7%. **Production W4A16 achieves ~100% recovery on OpenLLM v1.**
- **Our 46.89 raw is far below production W4A16 norms** — this is a recipe bug, NOT a quant ceiling.
- **Recommendations**: switch to llm-compressor; sym=False + actorder=weight; bump to 1024 samples; dampening_frac=0.1.

**Verdict on tooling**: **Switch from GPTQModel to llm-compressor.** Eliminates the qzeros bug class, tokenizer re-serialize, transformers pin chain.

---

## Conflict: MLP-only vs full Linear quantization

| Agent 1 (hybrid attention) | Agent 4 (production recipes) |
|---|---|
| "MLP-only is the safe recipe" (Jamba, Quamba2) | "Stop MLP-only, quantize every Linear in decoder" (RedHat Qwen3-8B) |

**Resolution**: Both right, depends on architecture.

- **Standard transformers** (Llama, Qwen, Mistral): full Linear quant is fine because dense attention quantizes cleanly. Agent 4's recommendation applies.
- **Hybrid attention** (Jamba, Mamba, SALA): Linear attention's recurrent state h_t accumulates quant errors across time steps. MLP-only is the safe path. Agent 1's recommendation applies.

**For SALA**: stay MLP-only on the attention side, but adopt **all of Agent 4's other recommendations** (sym=False, actorder=weight, more calibration, higher damp, switch to llm-compressor). These are orthogonal to module set.

**Open caveat**: production W4A16 recipes don't include SALA-like architectures. Whether attention quantization works on dense MiniCPM4 layers (the non-lightning blocks) is untested. **Conservative move: keep all attention BF16 for now, revisit per-layer quant after baseline.**

---

## Prioritized action plan

### Phase 0 (already in ACC_OPTIMIZATION_ROADMAP.md): Establish baseline

- [ ] J0: re-quant locally with current recipe, anchor to `acc_ori ≈ 46-49`
- [ ] per-task breakdown via `analyze_predictions.py`

### Phase 1: Cheap recipe-config wins (no new tools)

Adjust `quantize_gptqmodel_w4a16.py`'s QuantizeConfig:
- [ ] **sym=False** (asymmetric) — production standard, +1-2pp
- [ ] **actorder="weight"** if GPTQModel 7.0.0 supports this flag (verify via grep)
- [ ] **dampening_frac=0.1** (currently 0.01) — production value for 8B-class
- [ ] **num_calibration_samples=1024** (currently 256)
- [ ] **First-2 + last-2 layer skip** via dynamic config

**Re-quant + eval, expected**: 46.89 → 50-55. Cost: ~80 min × 5 single-knob ablations OR ~80 min for all-at-once + see direction.

### Phase 2: Tool switch — llm-compressor (orthogonal to Phase 1)

Even if Phase 1 stays low, switching tool is high-EV:
- [ ] Test 5/14 llm-compressor GPTQ artifact already on disk (zero cost — script ready at `scripts/test_llmcompressor_artifact.sh`)
- [ ] If 5/14 result is competitive, build a fresh llm-compressor-based W4A16 with current recipe knobs

**Why switch**: eliminates qzeros bug (gptqmodel 7.0 sym=True), tokenizer re-serialize (H4), transformers pin requirement (H5). Cleaner deployment.

### Phase 3: AWQ via llm-compressor

After Phase 2 lands (we have llm-compressor pipeline working):
- [ ] Replace `GPTQModifier` with `AWQModifier` + `QuantizationModifier`
- [ ] `W4A16_ASYM`, group_size=128
- [ ] Same calibration as Phase 1
- [ ] Eval — expected **+3-8pp from method swap**

**Strongest a-priori bet** given OpenBMB ships AWQ for MiniCPM-V.

### Phase 4: Sensitivity-tiered mixed precision (only if Phase 1-3 stuck < 65)

- [ ] Audit which layer indices are Lightning Attention vs dense MiniCPM4 in SALA's `mixer_types`
- [ ] Apply W8 to MLP feeding into Lightning layers, W4 elsewhere
- [ ] **Verify Marlin supports mixed-bit at inference** before sinking a re-quant
- [ ] Re-quant + eval

### Phase 5: Rotation (R1+R2 offline only)

If Phase 1-4 reach 60-70 but not 80:
- [ ] Apply `SpinQuantModifier(rotations=["R1","R2"], transform_type="hadamard")` in llm-compressor recipe
- [ ] R1+R2 fuse into weights at quant time, load as plain W4A16 via existing SGLang path
- [ ] 1-day spike, abort if SALA attention layer mapping fails

### Phase 6 (deferred): Training-assisted

- [ ] LoRA-on-quant or QAT — only if all post-training quant ceilings exhausted

---

## Critical URLs (server-side Claude reading order)

**Start here (production recipes):**
1. https://huggingface.co/RedHatAI/Qwen3-8B-quantized.w4a16 — closest size match, full recipe in card
2. https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a16/ — canonical template
3. https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 — alternate (damp=0.1, desc_act=True)

**AWQ specifically:**
4. https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/awq/ — official AWQ recipe
5. https://huggingface.co/openbmb/MiniCPM-V-4_5 — OpenBMB's AWQ for MiniCPM-V (closest analog)

**Hybrid attention quantization:**
6. https://arxiv.org/html/2408.12570v1 — Jamba-1.5 ExpertsInt8 (MLP-only justification)
7. https://arxiv.org/html/2501.13484v1 — MambaQuant (warns rotation breaks on Mamba)
8. https://arxiv.org/html/2410.13229v1 — Quamba2 (W4A16 for SSMs)

**Long-context degradation:**
9. https://arxiv.org/abs/2505.20276 — Quantization on long-context (RULER/NoCHA)
10. https://developers.redhat.com/articles/2024/02/03/how-well-do-quantized-models-handle-long-context-tasks

**Rotation methods (for Phase 5):**
11. https://github.com/vllm-project/llm-compressor/blob/main/examples/transform/README.md — llm-compressor R1+R2 example
12. https://arxiv.org/pdf/2405.16406 — SpinQuant paper (W4A8/A16 table)

**Comparative benchmarks:**
13. https://arxiv.org/html/2409.11055v1 — Quantized instruction-tuned LLMs comprehensive eval
14. https://arxiv.org/html/2411.02355v3 — "Give Me BF16 or Give Me Death" (counter-evidence)

---

## Risk register

| Risk | Mitigation |
|---|---|
| Production recipes don't include SALA-like architectures | Test on isolated Phase 1 changes before stacking |
| AWQ on 1M-context hybrid attention is uncharted | Keep GPTQ baseline as fallback; A/B locally before platform submission |
| Switching tool eliminates accumulated debugging knowledge | The 5/14 artifact already exists — start there before re-implementing |
| Long-context cliff (-15% at 128K) might cap us regardless of quant method | Repetition_penalty + max_tokens sampling fixes are still in roadmap |
| Marlin mixed-bit support uncertain | Verify in code before Phase 4 |
| SGLang compressed-tensors W4A16 path "not fully tested" (issue #2871) | The 5/14 artifact is in compressed-tensors format — testing it validates this path |

---

## Key strategic shifts vs ACC_OPTIMIZATION_ROADMAP.md

The roadmap (committed earlier today) is still mostly right but **research moves these into sharper focus**:

1. **Phase 2 (Tool switch to llm-compressor) is now Phase 2, not Phase 4.** Production evidence + bug elimination + AWQ access makes this very high ROI.
2. **AWQ is no longer "high cost research grade".** It's a 5-line recipe change in llm-compressor and ships in production for MiniCPM-V.
3. **First/last layer skip is a low-cost win** that should go alongside Phase 1 config changes, not be its own phase.
4. **MLP-only stays** despite Agent 4's general advice — SALA's Lightning Attention is the qualifier.
5. **Rotation defers further** — confirmed MambaQuant risk + SGLang gaps make it not first-priority.

---

## What server-side Claude should do first

After pulling this doc:

1. **Read `experiments/ACC_OPTIMIZATION_ROADMAP.md`** Phase 0 (J0 baseline + per-task)
2. **Read this doc** for the prioritized phase list
3. **Skim production recipe URLs** above (especially Qwen3-8B card)
4. **Execute Phase 1** with all 4 config changes batched (sym/actorder/damp/calib) — single re-quant gives the directional signal
5. **In parallel: test the 5/14 llm-compressor artifact** (Phase 2 step 1) — zero quant cost
6. **Record everything** in the eval_results table per the roadmap

If Phase 1 + 5/14-test give a clear winner, Phase 3 (full AWQ retrain) is the natural next.

---

**End of research summary. Status: 4 agents completed, findings cross-validated, action plan ranked.**
