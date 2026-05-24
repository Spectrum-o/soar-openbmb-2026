# EAGLE3 for MiniCPM-SALA — Design Document

> Branch `spec/eagle3-sala`. Created 2026-05-25 from research over 5 parallel
> Explore agents + spot-verification. This doc is the authoritative scope and
> sequencing source. Update in-place rather than fork.

## TL;DR (read first)

The task description ("Lightning Attention 兼容性是核心算法创新点") **overestimates
the open-research portion**. The required algorithm is already implemented in
this codebase — just for a different kernel family. Concretely:

- `GDNAttnBackend` (hybrid_linear_attn_backend.py:877) implements
  *snapshot-and-replay* tree verify for the gated-delta-rule recurrence.
  Tree state reload uses `intermediate_states_buffer` +
  `retrieve_parent_token` plumbed through the Triton kernel
  `fused_recurrent_gated_delta_rule_update_fwd_kernel`
  (fla/fused_recurrent.py:344, guarded by `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK`).
- `SimpleGLAAttnBackend` (hybrid_linear_attn_backend.py:1440) — the backend
  SALA's `MiniCPMLightningMixer` actually uses (minicpm.py:351 isinstance
  assert) — has **zero** `is_target_verify` branching. Its `forward()` calls
  `fused_recurrent_simple_gla` / `chunk_simple_gla` from upstream `fla`, both
  of which lack tree-mask support.

The "innovation" is therefore: port the snapshot-and-replay tree-verify
pattern from the GDR kernel to a new `fused_recurrent_simple_gla_update`
kernel, and wire `SimpleGLAAttnBackend` to call it under
`is_target_verify()`. This is a bounded engineering port, ~150 LoC of
Triton + ~100 LoC of Python plumbing, with an existing GDR template to
plagiarize.

Everything else is "standard" EAGLE3 work: a draft-head model file modeled
on `llama_eagle3.py`, a training pipeline (which this repo does not have),
and submission packaging.

---

## 1. Architectural background (verified)

### 1.1 SGLang's EAGLE3 pipeline

Two coexisting workers:

| | File | When |
|---|---|---|
| **V1** | `multi_layer_eagle_worker.py` | When overlap is OFF |
| **V2** | `multi_layer_eagle_worker_v2.py` | When `enable_overlap` is ON. Splits draft + target across `plan_stream`. Preferred in production. |

Both V1 and V2 share the same draft/verify mechanics. The contract:

1. Draft worker produces `EagleVerifyInput`:
   `draft_token[bs*N]`, `custom_mask` (tree mask), `positions[bs*N]`,
   `retrive_index / retrive_next_token / retrive_next_sibling[bs, N]`,
   where `N = topk^spec_steps + 1`.
2. Tree mask built by `sgl_kernel.build_tree_kernel_efficient`
   (eagle_utils.py:47, calling `sgl-kernel/csrc/speculative/eagle_utils.cu`).
3. Target model forward in `ForwardMode.TARGET_VERIFY`. Attention backends
   must respect the tree mask.
4. Verify decision via `sgl_kernel.verify_tree_greedy` (greedy) or
   `tree_speculative_sampling_target_only` (sampling).
5. Hook for attention backends to pre-allocate buffers:
   `get_verify_buffers_to_fill_after_draft()` (base_attn_backend.py).

### 1.2 SALA's model wiring in this fork

**Important:** Contrary to the task description, the SALA model **does NOT**
load via `trust_remote_code` in this fork — it has a native SGLang
implementation:

- `python/sglang/srt/models/minicpm.py`
  - `MiniCPMSALAForCausalLM` (line 564) — top-level CausalLM
  - `MiniCPMDecoderLayer` (line 381) — dispatches by `mixer_types[i]`:
    - `"minicpm4"` → `MiniCPMAttention` (line 90, dense, RadixAttention)
    - `"lightning"` / `"lightning_attn"` / `"lightning-attn"` →
      `MiniCPMLightningMixer` (line 201, calls SimpleGLAAttnBackend)
  - `MiniCPMMLP` (line 52) — SiluAndMul + gate-up + down
- `python/sglang/srt/configs/minicpm.py`
  - `MiniCPMHybridConfig` (line 9), `mixer_types`, `sparse_config`,
    `lightning_nkv`, `lightning_head_dim`, `lightning_scale`

This means we **can directly reuse** `MiniCPMAttention`, `MiniCPMMLP`,
`RMSNorm` from `minicpm.py` for the draft head's transformer block.
A new `minicpm_sala_eagle3.py` will be ~200 LoC, almost all template.

### 1.3 The Lightning Attention verify gap

The GDR kernel works as follows (paraphrased from fused_recurrent.py:344):

```
for token_idx in range(seq_len):
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK and tree_step >= 1:
        parent_idx = retrieve_parent_token[token_idx]
        h = intermediate_states_buffer[parent_idx]    # snapshot reload
    else:
        h = h  # continue recurrence linearly
    h = α_t * h + β_t * (k_t v_t^T) - α_t β_t * k_t (k_t^T h)  # GDR update
    intermediate_states_buffer[token_idx] = h        # store for descendants
    o[token_idx] = q_t @ h
```

Simple GLA differs only in the update step:

```
h = α_t * h + k_t * v_t^T          # no β, no delta-rule subtraction
```

So the tree-snapshot scaffolding is **structurally identical**; only the
update line changes. This is a copy-paste-and-edit Triton kernel.

### 1.4 SOAR submission interaction (verified)

- W4A16 + EAGLE3 compatibility: gptq_marlin loader doesn't know about
  draft model; both load via standard SGLang weight-loading paths.
- Hard constraint (from SUBMISSIONS.md): all dtype must be bfloat16 (W4A16
  cast to bf16 at Marlin output), or we hit the InfLLMv2 dtype-mismatch
  crash documented in row 128. Draft head must therefore also be bf16.
- 2 GB cap: NOT verified in any SOAR doc in this repo. The user told us
  this constraint; we'll respect it conservatively.
- Tarball overhead: ~2 GB draft model + existing ~5 GB W4A16 main + ~250 MB
  flash-attn wheel + ~50 MB bundled SGLang = ~7.5 GB total, vs the platform
  has 84 GB free; fits with margin.

### 1.5 SOAR scoring & expected upside

From SUBMISSIONS.md row 129: best current W4A16+bf16 → `final_score=23.18`,
S1=717s (decode-bound), acc_ori=82.18 (clears 80 gate by 2.18 pp). The
scoring formula rewards throughput; EAGLE3 attacks S1 most directly via
decode-step amortization. Realistic upside (per Explore Stream 5):

| Config | Expected `final_score` | Acceptance length target |
|---|---|---|
| W4A16+bf16 (current best) | 23.18 | n/a |
| + EAGLE3 (bf16 head, no spec on lightning) | 26–28 | ~2.0 |
| + EAGLE3 with full lightning verify fix | 28–32 | ~2.5–3.0 |
| + chunked-prefill 32K stack | 30–34 | (perf orthogonal) |

These are upper-bound estimates; risk of regression if acceptance length
< 1.5 (overhead exceeds savings).

---

## 2. Risks / Gotchas pre-implementation

1. **Accuracy regression if lightning layers compute incorrectly under tree
   verify**. Without the kernel fix, lightning layers' `o` will not respect
   ancestor-only attention → acc_ori may drop below 80 gate. Mitigation:
   land the kernel fix before any platform submission. Test path: synthetic
   2-branch tree, compare GDN output (known correct) vs SimpleGLA output.

2. **Draft head dtype boundary**. Same trap as v5j: if draft is fp16 and
   we run server with `--dtype bfloat16`, downstream cast paths may mismatch
   at sparse backend boundaries. Decision: train and ship the draft head in
   bf16. Cost: ~1.6 GB BF16 vs ~0.8 GB FP16, still inside 2 GB cap.

3. **Tokenizer drift**. H4 memory: GPTQModel re-serializes tokenizer with
   split chat_template that older transformers ignores. Lesson applies
   here: draft head MUST share base SALA's tokenizer; do NOT re-save via
   any HF API. Symlink `draft_model/tokenizer.json` → base.

4. **Hidden-state aggregation for EAGLE3**. llama_eagle3.py uses
   `capture_aux_hidden_states` (line 226) to expose multiple intermediate
   layers, not just last hidden. SALA has 32 hybrid layers — which ones to
   capture for aux is an open design choice. EAGLE3 paper uses 3 layers;
   for hybrid models, lightning-layer hidden states may be lower-quality
   for draft prediction (recurrent state ≠ rich representation). **Pick
   dense layers only for aux capture.** Verify in §5 ablation.

5. **No training infra in this repo**. We must write it from scratch.
   ~600 LoC: dataset loader + hidden-state collector + supervised
   cross-entropy training loop. References: EAGLE upstream repo training
   scripts (must read; not yet inventoried).

6. **Platform's SGLang version may differ**. Mitigation: bundle current
   `spec/eagle3-sala` source via existing `submission_*/sglang/` pattern.
   Same approach as v5j.

---

## 3. Proposed phasing

### Phase 0 — preflight (this doc + a tiny smoke)
- ✅ Branch created, design doc written, agents validated.
- Pending: 1 small commit with `experiments/EAGLE3_SALA_DESIGN.md`
  + this branch's tracking task entries.

### Phase 1 — SimpleGLA tree-verify kernel (CORE ALGORITHM WORK)
**No GPU required to write; required to test.**

1.1. Add `fused_recurrent_simple_gla_update_fwd_kernel` to
     `python/sglang/srt/layers/attention/fla/fused_recurrent.py`. Copy
     `fused_recurrent_gated_delta_rule_update_fwd_kernel` and strip the
     `β_t k_t (k_t^T h)` delta term. Keep `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK`,
     `intermediate_states_buffer`, `retrieve_parent_token` plumbing
     unchanged.
1.2. Add Python wrapper `fused_recurrent_simple_gla_update(...)` modeled
     on `fused_recurrent_gated_delta_rule_update_fwd` (line 522). Strict
     stride parity with GDR variant.
1.3. Modify `SimpleGLAAttnBackend` (hybrid_linear_attn_backend.py:1440):
     - Add `is_target_verify` branch in `forward()`. When set, call the
       new kernel path with `retrieve_parent_token` and
       `intermediate_states_buffer` from `forward_metadata`. Pattern from
       `GDNAttnBackend.forward_extend` lines 993–1105.
     - In `_forward_metadata` / `_capture_metadata` / `_replay_metadata`
       (inherited from `MambaAttnBackendBase`): plumb the parent token
       buffer. Look for the GDN equivalents in
       `init_forward_metadata*` (lines 220–292) and mirror them.
     - Allocate `intermediate_state_cache` per-layer.
1.4. Unit test in `test/srt/test_simple_gla_tree_verify.py` (new):
     two-branch synthetic tree of depth 3, compare:
     - GDN backend output with delta rule disabled (β=0)
     - SimpleGLA backend output with new kernel
     - They must match to atol 1e-3 in bf16. (CPU equivalence pre-check
       can be done with torch eager fallback.)
1.5. CI gate: extend `scripts/full_preflight.sh` to run the new test.

**Output of Phase 1**: `git diff` ~300 LoC, single PR-able commit.
SOC verified that `MiniCPMHybridConfig` lightning routing reaches the
new kernel — no platform run yet.

### Phase 2 — EAGLE3 draft-head model file
**No GPU required.**

2.1. Create `python/sglang/srt/models/minicpm_sala_eagle3.py`, modeled on
     `llama_eagle3.py`:
     - 1-layer transformer block using `MiniCPMAttention` (dense, **not**
       lightning) + `MiniCPMMLP` + `RMSNorm` from `minicpm.py`.
     - `fc` projection: input is concat of `[embed, target_hidden]`,
       optionally `[embed, hid_aux1, hid_aux2, hid_aux3]` for true EAGLE3.
       Match upstream's `target_hidden_size * 3 → hidden_size`.
     - Tied embeddings with target (saves ~1 GB).
     - Custom `load_weights()` honoring `qkv_proj` / `gate_up_proj` stacking,
       same as llama_eagle3.py:229.
     - Register `EntryClass = [MiniCPMSALAEagle3ForCausalLM]`.
2.2. Add a `MiniCPMSALAEagle3Config` in `configs/minicpm.py` if
     non-trivial; otherwise reuse `MiniCPMHybridConfig` with
     `num_hidden_layers=1`.
2.3. Register in `models/__init__.py` / model_loader dispatch.
2.4. Unit test: load random-init weights, forward through tiny input
     dimensions, verify output shape `[bs, hidden]` + aux list.

**Output of Phase 2**: ~200 LoC new file, ~30 LoC dispatch wiring.

### Phase 3 — Hidden-state collection + draft-head training
**GPU required (AutoDL).**

3.1. Write `scripts/collect_eagle3_hidden_states.py`:
     - Spin up SALA target via SGLang infrastructure (offline mode).
     - For each prompt in calibration corpus (start with
       `submission_*/perf_public_set.jsonl` — already long-context, same
       distribution we're optimizing for).
     - Capture `output_hidden_states[layers_to_aux]` per token.
     - Serialize to npz / safetensors files; ~ each prompt = a few MB.
3.2. Write `scripts/train_eagle3_head.py`:
     - Dataset: streaming over hidden-state files.
     - Loss: cross-entropy on next-token logits + auxiliary feature loss
       (cosine with target hidden, per EAGLE3 paper).
     - Optimizer: AdamW; 2–3 epochs on ~100K samples (≈ realistic
       overnight on RTX 6000D).
     - Save BF16 safetensors to
       `/root/autodl-fs/zyn/models/sala-eagle3-draft/`.
3.3. Local sanity: load draft head, do a 10-prompt greedy decode comparison
     of (no spec) vs (EAGLE3 with `--speculative-num-draft-tokens 4`).
     Measure acceptance length and latency.

**Output of Phase 3**: training tarball, draft head safetensors (~1.5 GB),
sanity report.

### Phase 4 — Submission packaging + smoke + platform run
**AutoDL + 1 platform slot.**

4.1. New variant `submission_w4a16_eagle3/`:
     - Symlinks from current best (`submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe`).
     - Override `prepare_env.sh` to export
       `SGLANG_SERVER_ARGS="... --speculative-algorithm EAGLE3
       --speculative-draft-model-path ./draft_model
       --speculative-num-draft-tokens 4 --speculative-eagle-topk 1
       --speculative-num-steps 3"`.
     - Bundle draft head under `submission_w4a16_eagle3/draft_model/`.
     - Bundled SGLang source includes Phase-1 + Phase-2 changes.
4.2. Extend `scripts/full_preflight.sh`:
     - Check draft model dir exists, dtype = bf16.
     - Check `SGLANG_SERVER_ARGS` carries `--speculative-algorithm EAGLE3`.
     - Check bundled SGLang has the new kernel files.
4.3. GPU smoke (30 samples) on AutoDL:
     - Acc must stay ≥ 80 gate.
     - Acceptance length ≥ 1.5 (else spec overhead exceeds gain).
     - S1 latency drop ≥ 15% vs chunk32k_safe baseline.
4.4. Platform submission only after smoke clears all three.

**Output of Phase 4**: 1 packed tarball, 1 platform result (5h slot).

---

## 4. File-level work breakdown

| File | New / Modified | Phase | LoC est. |
|---|---|---|---|
| `python/sglang/srt/layers/attention/fla/fused_recurrent.py` | M | 1 | +180 |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | M | 1 | +120 |
| `test/srt/test_simple_gla_tree_verify.py` | N | 1 | +200 |
| `python/sglang/srt/models/minicpm_sala_eagle3.py` | N | 2 | +250 |
| `python/sglang/srt/models/__init__.py` (or registry) | M | 2 | +5 |
| `python/sglang/srt/configs/minicpm.py` | M | 2 | +30 |
| `test/srt/test_minicpm_sala_eagle3.py` | N | 2 | +120 |
| `scripts/collect_eagle3_hidden_states.py` | N | 3 | +300 |
| `scripts/train_eagle3_head.py` | N | 3 | +400 |
| `submission_w4a16_eagle3/README_SUBMISSION.md` | N | 4 | +60 |
| `submission_w4a16_eagle3/prepare_env.sh` | N | 4 | +40 |
| `submission_w4a16_eagle3/prepare_model.sh` | N | 4 | +20 |
| `scripts/full_preflight.sh` | M | 4 | +30 |
| `experiments/EAGLE3_SALA_DESIGN.md` | N | 0 | (this doc) |
| `tools/pack_submission.py` (canary additions) | M | 4 | +40 |

Total estimate: ~1800 LoC code, ~400 LoC tests, ~120 LoC docs.

---

## 5. Open questions to resolve before Phase 3

(Phase 1 + 2 can proceed without these answered.)

1. **Aux-hidden layer choice for SALA**: which subset of the 32 hybrid
   layers gives the best draft-prediction signal? Dense-only? Last
   N dense? Sample ablation on small training run.
2. **`draft_vocab_size`**: SALA has 150K vocab. Reduce to 50K "hot
   tokens" for size + speed? EAGLE paper does this; saves ~0.4 GB BF16.
3. **`speculative-num-draft-tokens`**: 4 vs 8 vs 16. Larger trees =
   more verify cost; long-context decode regime favors larger trees up
   to a point (overhead from KV-cache mem). Pick after Phase 4 smoke.
4. **Training corpus**: just `perf_public_set.jsonl` (150 samples) is
   too small. Where do we get 100K SALA-distribution samples? Mix of
   public long-context QA / NIAH / synthetic? Open.

---

## 6. References (within this repo)

| Doc | What it gives us |
|---|---|
| `SUBMISSIONS.md` | Hard constraints + scoring + W4A16 + bf16 path |
| `experiments/STATE_OF_PLAY.md` | What's currently best (W4A16+bf16+chunk32k) |
| `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` | Reference draft-worker main loop |
| `python/sglang/srt/speculative/eagle_info.py` | Verify-input dataclass shapes |
| `python/sglang/srt/speculative/eagle_utils.py` | Tree-mask builder wrapper |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` (GDNAttnBackend) | Snapshot-replay template to port |
| `python/sglang/srt/layers/attention/fla/fused_recurrent.py` (gated_delta_rule_update kernel) | Triton template to port |
| `python/sglang/srt/models/llama_eagle3.py` | Draft-head template |
| `python/sglang/srt/models/minicpm.py` (`MiniCPMAttention`, `MiniCPMMLP`) | Reusable building blocks |
| `tools/pack_submission.py`, `scripts/full_preflight.sh` | Submission canaries |

## 7. Out of scope for this design

- Multi-layer EAGLE3 (stacking more than 1 draft block). Upstream
  llama_eagle3.py is single-layer; we'll match. Multi-layer can be a
  follow-up after Phase 4 lands a positive platform result.
- AWQ + EAGLE3 stacking. AWQ path is independently progressing in
  `submission_awq_*` variants; we can revisit interaction once both
  paths individually clear the 80 gate.
- Op-fusion overlay (`perf/op-fusion`) interaction. Orthogonal; can
  stack after EAGLE3 ships.

## 8. Status

| Phase | Status | Owner | Date |
|---|---|---|---|
| 0 (preflight + design) | **DONE** | claude | 2026-05-25 |
| 1 (SimpleGLA tree-verify) | pending | TBD | — |
| 2 (draft-head model) | pending | TBD | — |
| 3 (training) | pending | TBD (needs GPU) | — |
| 4 (submission) | pending | TBD (needs platform slot) | — |
