# Acc Optimization Roadmap (post-1849 breakthrough)

> Written 2026-05-22 evening, right after 1849 platform submission returned
> `acc=58.61, acc_ori=46.89, final_score=0.0`.
>
> Project transitioned from **"why acc=0"** phase to **"how to push 58.61 → 80+"** phase.
> This document captures the state of the question, all known optimization levers,
> ROI estimates, and a phased plan.

---

## Snapshot — current state

### Platform result (1849 tarball, 2026-05-22)

```
acc                = 58.61    ← main metric, first nonzero W4A16 result
acc_ori            = 46.89    ← probably aggregate WITHOUT fwe free-credit
final_score        = 0.0      ← correctness gate not cleared (gate ≈ 80%)
benchmark_duration {
  S1   = 713.99    (vs v17 ~625, +14% slower)
  S8   = 1054.01   (vs v17 ~997, +6% slower)
  Smax = 2326.46   (vs v17 ~2290, +1.6%)
}
```

### Recipe that produced 58.61

| Knob | Value | Source of truth |
|---|---|---|
| Algorithm | GPTQ (Hessian-based) | `quantize_gptqmodel_w4a16.py` |
| Tool | gptqmodel==7.0.0 (exact pin) | prepare_env.sh step 1 |
| Bit width | W4A16 | `--bits 4` |
| Group size | 128 | gptqmodel default |
| sym | True (uint4b8 for Marlin) | hard constraint |
| desc_act | False | Marlin requirement |
| Module set | **MLP-only** (gate/up/down across all 32 layers) | layer_modules in `register_minicpm_sala_with_gptqmodel` |
| Attention | NOT quantized (BF16) | dynamic skip `-:.*self_attn.*$: True` |
| Calibration | perf_public_set, 256 samples, max 8192 tokens, tail-truncate, chat template | prepare_model.sh |
| transformers (platform) | **4.57.1** (forced via fix2 cascade) | the H5 breakthrough |
| huggingface-hub (platform) | <1.0 (cascaded from transformers reinstall) | fix2 |
| Inference loader | gptq_marlin | SGLANG_SERVER_ARGS |
| dtype | float16 (sed-patched from bf16) | prepare_env.sh runtime patch |
| Sampling params | (none) | NOT set |
| repetition_penalty | (not set, default 1.0) | NOT set |
| max_tokens | default 65536 | NOT set |

### What worked vs what didn't (history)

| Submission | Module set | qzeros fix | tokenizer fix | transformers pin | Platform acc |
|---|---|---|---|---|---|
| v17 | full-attn GPTQ | ❌ | ❌ | ❌ | **0** |
| v21 | MLP-only GPTQ | old (broken) | ❌ | ❌ | **0** (local 49) |
| v22 | full-attn GPTQ | old (broken) | ❌ | ❌ | **0** |
| v23 | MLP-only GPTQ | hardened (broken POST-CHECK) | ✅ | ❌ | killed by my POST-CHECK bug |
| v23b | MLP-only GPTQ | hardened | ✅ | ❌ | shape mismatch crash |
| **1849** (v24c) | **MLP-only GPTQ** | **hardened** | **✅** | **✅ (4.57.1)** | **58.61** ✅ |
| RTN-scalefix | numpy direct (no GPTQModel) | — | unconditional copy | n/a (bypasses transformers) | 42 |

**Conclusion** (server-side analysis 2026-05-22 evening):
- **H1 (hardened qzeros)** + **H4 (tokenizer overwrite)** are the **acc drivers**. They lift acc from 0 to ~47.
- **H5 (transformers pin to 4.57.1)** is a **load-time shape fix**, NOT an acc fix. H4 now overwrites `modeling_minicpm_sala.py`; on platform's transformers 5.9.0 this caused v23b's shape mismatch. H5 makes platform agree with our overlay.
- **All three must stack** for a working pipeline, but H5's contribution is "unblock loading", not "raise acc".
- **acc_ori=46.89 ≈ local v21 acc 47-49 within noise** → local↔platform delta on the raw metric is only ~2pp, NOT the 49pp we feared before fix2 landed.
- **`acc=58.61` is platform-weighted** (probably fwe free-credit or task-bonus weighted). Don't anchor experiments to 58.61 — **anchor to acc_ori=46.89** when comparing with local results.
- RTN's 42 is a separate baseline that doesn't go through any of these.

---

## User's actual goal (calibration check)

- **Mentor's directive**: implement most of what champion described, target ~rank 20
- **Leaderboard target**: rank ~20 ≈ final_score 30+
- **Current**: final_score 0 (gate not cleared)

**Two independent paths to rank 20**:

1. **BF16 + chunked-prefill safetynet** (predicted final_score 25-30) → rank 20 alone, NO quant required
   - `soar_bf16_chunk32k_safetynet_submission_20260520_v3.tar.gz` already built (2KB, not bundled SGLang)
   - Should be submitted as the **floor**, regardless of W4A16 progress
2. **W4A16 acc → 80+** (clears gate, final_score could be 30-50) → stretch
   - Requires bridging the **21pp gap** (58.61 → 80)
   - Probability of success uncertain; below

**The W4A16 ceiling is NOT a hard requirement for the rank-20 goal.** Treat it as a high-variance bonus on top of BF16 floor.

---

## All optimization levers (the universe)

Categorized by **timing** (when in the pipeline you act) and **ROI** (gain ÷ effort).

### Tier 1 — Inference-time changes (cheapest, no re-quant)

| Lever | Mechanism | Expected gain | Effort |
|---|---|---|---|
| `repetition_penalty=1.05` | Penalize already-emitted tokens at sampling → break repetition loops on cwe/niah | **+5-10pp** | 30 min |
| `max_tokens` cap (65536 → 16384) | Truncate before model digs grave on stuck repetition | +3-5pp | 5 min |
| `top_p < 1.0` or `temperature > 0` | Add stochasticity to break loops | +2-5pp | 5 min |
| `chunked-prefill 65K` (currently 8K) | Latency optimization (acc unchanged) | 0 acc, -20-30% S1/S8 latency → final_score wins after gate cleared | 5 min |

**Failure mode addressed**: `project_v21_per_task_breakdown` documented cwe/niah failures as **quant-induced repetition loops**. Sampling fixes are the direct intervention.

**Caveat**: `repetition_penalty` may need to be passed in the chat completion request payload, not as a SGLang server arg. Need to check SOAR Toolkit's `eval_model.py` to see if it forwards arbitrary sampling params.

### Tier 2 — Quant config tuning (re-quant required)

| Lever | Mechanism | Expected gain | Effort | Variant ready? |
|---|---|---|---|---|
| `group_size=64` (down from 128) | Finer scales = less quant error per group | +2-5pp uniformly | 50 min re-quant | ✅ `submission_gptqmodel_calib_w4a16_v25_g64/` |
| W8A16 (`--bits 8`) | Less aggressive quant, acc much higher but latency win smaller | +10-20pp acc | 60 min | ✅ `submission_gptqmodel_calib_w4a16_v24_bits8/` |
| `desc_act=True` (activation reordering) | Quantize most-important columns first, reduce error | +3-7pp **but Marlin-incompatible** | requires switching to compressed-tensors loader | ❌ |
| First/last N layer skip (BF16) | First/last layers most sensitive (industry heuristic) | +3-8pp | requires layer_modules surgery (see mixed-precision section) | ❌ |
| Calibration: multi-adaptive | 1-3 windows per long prompt instead of single tail | +3-5pp on long-context tasks (niah/cwe) | 60 min re-quant | ✅ `submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive/` |
| Calibration: more samples (512+) | More Hessian statistics | +1-3pp | longer quant (~90 min) | parameterizable |
| Calibration: longer max_len (16K) | Capture more of the long prompts | +2-4pp niah/cwe | longer quant + more VRAM | parameterizable |

### Tier 3 — Mixed precision / architectural

| Lever | Mechanism | Expected gain | Effort |
|---|---|---|---|
| **Lightning layer MLP not quantized** | Lightning attention is recurrent; quant errors accumulate across recurrence. Keep MLP BF16 in lightning layers. | **+5-10pp** | Medium — requires both quant-time skip AND SGLang dynamic skip rules to be aligned. See "Mixed-precision implementation pitfall" below. |
| Per-layer sensitivity profiling + skip | Run BF16 forward, measure quant-induced acc drop layer-by-layer, skip worst N | +5-10pp | High — 1 day profiling work |
| Outlier-aware split (SparseGPT-style) | Top 0.1% outlier weights kept FP16, rest W4 | +5-10pp | High — requires custom loader |
| Lightning gate FP32 promotion | Force `g_t = sigmoid(...)` in FP32 to stabilize recurrence | +3-8pp | Medium — code change in `MiniCPMLightningMixer` |

**Mixed-precision implementation pitfall** (discovered 2026-05-22 evening):

> "Just patch quantize_config.json to add dynamic skip rules" does NOT work. If GPTQModel
> wrote `.qweight/.scales/.qzeros` for a module but SGLang dynamic skip says "treat as
> BF16", SGLang's UnquantizedLinearMethod expects `.weight` (which is missing) →
> KeyError on load. **Quant-time skip and load-time skip must be aligned.** Same class
> of bug as v14's `KeyError: model.layers.0.self_attn.o_gate.weight` (SUBMISSIONS.md
> row 99).
>
> Implementation requires either (a) gptqmodel's `QuantizeConfig.dynamic` to also
> skip during quant — needs code audit to verify, (b) custom per-layer-index filter
> in `layer_modules`, or (c) post-quant safetensors surgery (high risk).

### Tier 4 — Algorithm switch

| Lever | Mechanism | Expected gain on SALA | Effort |
|---|---|---|---|
| **llm-compressor with GPTQ** (the 5/14 artifact) | Same algorithm, different implementation. 5/14 artifact at `/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/` exists but **never benchmarked**. | ±5pp (could go either way) | **0 re-quant** — just load and eval (30 min) |
| **AWQ** (activation-aware) | Per-channel scaling protects outlier channels. SALA has lightning gate + scale_depth/scale_emb outliers → theoretically AWQ-friendly. | **+5-10pp**, possibly higher if outlier-driven failures are real | 1-1.5 days (different tool + different SGLang loader path) |
| SmoothQuant pre-processing | Migrate activation outliers into weights before GPTQ | +5-10pp | Medium |
| SpinQuant / Hadamard rotation | Rotate weight+activation to make distribution more quant-friendly | +10-15pp (paper data) | High — research-grade |

### Tier 5 — Training assist (research-grade)

| Lever | Mechanism | Expected gain | Effort |
|---|---|---|---|
| LoRA-on-quant | Train rank-8 LoRA after quantization to compensate quant errors | +10-20pp | several hours training + tuning |
| QAT (quantization-aware fine-tune) | Short SFT after quant, lets weights re-adapt | +15-25pp | 1+ day |
| Distillation from BF16 teacher | BF16 model teaches W4A16 student | +10-20pp | 1-2 days |

**These are NOT in the champion playbook** (笔记 04/05) — champion assumed no training. **If "implement champion techniques" is the only metric, skip Tier 5. If you have flexibility, Tier 5 is the most direct path to clear the 80% gate.**

### Tier 6 — Orthogonal (latency-only)

These don't help acc but improve final_score AFTER gate is cleared:

| Lever | Predicted speedup | Effort |
|---|---|---|
| chunked-prefill 8K → 65K | -20-30% latency | 5 min server args |
| Op-fusion (笔记 04, `experiments/op_fusion_verify.patch`) | -5-10% latency | medium (GPU validation required) |
| NVFP4 KV cache (笔记 05) | -10-20% latency on Smax | hard (untested on SALA) |

---

## Decision tree — what to do next

```
Read 1849's per-task breakdown (after Exp J0 baseline + analyze_predictions.py)
│
├─ niah/cwe < 30% (repetition collapse confirmed)
│    │
│    ├─ Tier 1: repetition_penalty sweep ............ HIGHEST ROI, 30 min
│    └─ Tier 3: lightning-skip mixed precision ...... +5-10pp targeted, 2-3 hours
│
├─ All tasks moved up roughly evenly (~10pp from v21 levels)
│    │
│    ├─ Tier 2: group_size 64 ........................ +2-5pp uniform, 50 min
│    └─ Tier 2: calibration multi-adaptive ........... +3-5pp, 60 min
│
├─ qa stuck at ~30% but cwe/niah recovered
│    │
│    └─ Calibration tuning: longer max_len + more samples
│
└─ All tiers exhausted, acc plateaus < 70
     │
     ├─ Tier 4: llm-compressor 5/14 artifact (free test, 30 min)
     ├─ Tier 4: AWQ (1-1.5 days)
     └─ Tier 5: LoRA-on-quant (several hours training)

Parallel to ALL above:
└─ Submit BF16 v3 safetynet → locks final_score 25-30 ≈ rank 20 floor
```

---

## Phased plan (concrete next steps)

### Phase 0 — Baseline establishment (DO FIRST when GPU available)

**Total: ~80 min, all on AutoDL**

1. **Verify env matches fixenv2** (5 min)
   - python_pkg_exact_version transformers 4.57.1
   - huggingface-hub < 1.0
   - gptqmodel == 7.0.0
   - tokenizers in [0.22, 0.23)
2. **Exp J0** — `bash scripts/local_eval.sh --variant submission_gptqmodel_calib_w4a16 --num-samples 150 --force-requant` (~50 min)
3. **per-task analysis** — `python3 tools/analyze_predictions.py <predictions.jsonl>` (~5 min)
4. **Record results** in eval_results table:
   ```
   variant | calib | n_calib | max_len | g_size | rep_pen | acc | niah | cwe | fwe | qa | mcq | t_quant | t_eval
   J0      | tail  | 256     | 8192    | 128    | 1.0     | ?   | ?    | ?   | ?   | ?  | ?   | ~20m    | 30m
   ```

**Pass criterion**: J0 local **acc_ori within 5pp of 46.89** (the platform's raw metric — NOT 58.61, which is platform-weighted). The expected match is ~46-49 (also v21 local's range). If local acc_ori is **<40 or >55**, local≠platform — investigate before further experiments. The `acc` field (~58 range) is platform-side weighting that local may not reproduce, so don't anchor to it.

### Phase 1 — Cheap wins (no re-quant)

**Total: ~1 hour**

5. **Test 5/14 llm-compressor artifact** — `bash scripts/test_llmcompressor_artifact.sh --num-samples 150` (on parallel/non-gptqmodel-paths branch). 30 min.
6. **repetition_penalty sweep** (1.0 / 1.05 / 1.10) — 30 min. Need to confirm SGLang server-arg or modify eval_model.py request payload.

**Decision points**:
- If llm-compressor's acc_ori > 47 → beats GPTQModel local, strong evidence to switch tool entirely. (Compare on acc_ori, not the platform-only acc.)
- If repetition_penalty raises niah/cwe by >5pp → keep it as baseline going forward

### Phase 2 — Quant config tuning

**Total: ~3 hours**

7. **Exp N**: g64 — re-quant with `--group-size 64` (50 min) + eval. Variant ready.
8. **Exp M**: multi-adaptive calib — variant ready. ~60 min.
9. **Exp MN**: combine g64 + multi-adaptive. ~60 min.

**Decision points**:
- If Exp N alone gives < +2pp → group size isn't the bottleneck. Skip Exp MN.
- If Exp M gives big niah/cwe gain → calibration coverage was real issue. Push further (longer max_len, more samples).

### Phase 3 — Mixed precision (if Phase 1+2 stuck < 65)

**Total: ~3 hours setup + 60 min experiment**

10. **Audit gptqmodel's `dynamic` field** — does it skip during quant or only at load? Determines implementation path.
11. **Implement lightning-skip mixed precision** — either via `dynamic` (if supported) or via custom `layer_modules` filter.
12. **Re-quant + eval**. ~60 min.

### Phase 4 — Algorithm switch (if Phase 3 stuck < 70)

13. **AWQ via llm-compressor**. 1-1.5 days.

### Phase 5 — Training assist (if Phase 4 stuck < 75)

14. **LoRA-on-quant or QAT**. 1+ day.

---

## Realistic outcome estimates

Assuming all phases executed and each lever delivers its expected gain (no 100% capture, but most stack at ~60% combined efficiency):

| Phase exit | Acc | Probability |
|---|---|---|
| Phase 0 only (baseline) | 58 | 100% |
| Phase 1 (sampling) | 60-65 | 80% |
| Phase 2 (quant config) | 63-70 | 70% |
| Phase 3 (mixed precision) | 67-75 | 50% |
| Phase 4 (AWQ) | 70-80 | 30% |
| Phase 5 (LoRA/QAT) | 78-85 | 20% |

**Calibrated honest summary**: clearing the 80% gate is **possible but not guaranteed** even after all phases. Probability of `final_score > 0` from W4A16 path alone: ~20-30%. **This is why BF16 v3 safetynet must be submitted as the floor.**

---

## Open questions / unknowns

1. **What is the actual correctness gate threshold?** Common practice is 80% but SOAR may be different. RTN-42 didn't clear, 1849-58.61 didn't clear → gate ≥ 60. Could be 60, 70, 80, or higher.
2. **What is the platform's hidden eval set distribution?** Per `calibration_gotchas.md` line 7, "almost certainly harder than perf_public_set". After 1849: acc_ori=46.89 vs local v21 47-49 → **delta is ~2pp on the raw metric** (within noise). The +12pp from acc_ori to platform `acc` is a platform-side weighting (fwe free-credit / task-bonus), NOT a difficulty difference. **Anchor future comparisons to acc_ori, not acc.**
3. **Does gptqmodel's `QuantizeConfig.dynamic` field skip during quant?** Critical for clean mixed-precision implementation. Requires source code audit.
4. **Does the 5/14 llm-compressor artifact still exist on disk?** `/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/` may have been pruned. Check first before relying on it.
5. **Does SOAR Toolkit's `eval_model.py` accept `repetition_penalty` in request payload?** Determines whether repetition_penalty sweep is a one-line change or a script modification.

---

## Existing infrastructure

What's already in the repo, ready to use:

| Tool / variant | Path | Purpose |
|---|---|---|
| 1849 source | `submission_gptqmodel_calib_w4a16_v24_pin_transformers/` | Reproduce baseline |
| g64 variant | `submission_gptqmodel_calib_w4a16_v25_g64/` | Exp N |
| Multi-adaptive variant | `submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive/` | Exp M |
| W8A16 variant | `submission_gptqmodel_calib_w4a16_v24_bits8/` | If W4A16 path dies |
| BF16 v3 safetynet | `soar_bf16_chunk32k_safetynet_submission_20260520_v3.tar.gz` | Floor for rank 20 |
| Per-task analyzer | `tools/analyze_predictions.py` | Phase 0 step 3 |
| Repetition analyzer | `tools/repetition_analyzer.py` | Diagnose cwe/niah loops |
| local_eval.sh | `scripts/local_eval.sh` | End-to-end quant + eval |
| Op-fusion patch | `experiments/op_fusion_verify.patch` | Tier 6 (latency only) |
| BF16+op_fusion variant | `submission_bf16_op_fusion/` (on parallel branch) | Latency stretch after gate |
| llm-compressor test script | `scripts/test_llmcompressor_artifact.sh` (on parallel branch) | Phase 1 step 5 |

What's NOT yet built:

- repetition_penalty sweep wrapper script
- lightning-skip mixed-precision quant variant
- AWQ-based llm-compressor recipe
- Per-layer sensitivity profiler

---

## Memory updates needed when work continues

When phases progress, update these memory files:

- `feedback_h5_transformers_runtime_drift.md` — already updated to "VERIFIED platform=58.61"
- `project_v21_per_task_breakdown.md` — update with 1849's per-task numbers once J0 is run
- Add new memory: `project_acc_optimization_phase_<N>_results.md` for each phase outcome

---

**End of roadmap.** Pick a phase, execute, record results, update this doc.
