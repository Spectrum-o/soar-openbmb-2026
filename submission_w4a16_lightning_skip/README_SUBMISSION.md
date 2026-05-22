# SOAR W4A16 Lightning-Skip Mixed Precision Variant

> Built 2026-05-23 (night, autonomous). Mixed-precision quant on SALA:
> dense (minicpm4) layers' MLPs are W4A16, lightning attention layers'
> MLPs stay BF16. Targets the suspected long-context (cwe / niah)
> repetition-collapse failure mode of the all-MLP-quantized 1849 baseline.

## Hypothesis

SALA's hybrid attention has ~24 lightning + ~8 dense layers (per config's
`mixer_types`). Lightning attention is recurrent:
```
h_t = g_t * h_{t-1} + k_t @ v_t.T
out_t = q_t @ h_t
```

Quantization errors in the MLPs that FEED into this recurrence
accumulate across time steps. memory `project_v21_per_task_breakdown` notes
that cwe/niah failures in v21 (acc=49 local) are "quant-induced repetition
loops" on long inputs — exactly what recurrence-state error compounding
would produce.

This variant tests whether keeping lightning-adjacent MLPs in BF16 (while
still quantizing dense-layer MLPs) breaks the repetition collapse.

## Implementation: post-quant surgery (no API risk)

Standard approaches don't compose cleanly here:
- `layer_modules` in GPTQModel registration is per-layer-TYPE, not per-INDEX
- gptqmodel's `QuantizeConfig.dynamic` quant-time semantics aren't reliably documented
- Tool dispatch hooks would require deep gptqmodel internals

Instead this variant uses **post-quant surgery**:

1. Standard 1849-style MLP-only GPTQ (writes .qweight/.qzeros/.scales for ALL MLPs)
2. `apply_lightning_skip_overlay.py` runs after:
   - Reads `mixer_types` from config.json → finds lightning layer indices
   - Reads original BF16 .weight tensors for those lightning MLPs from input dir
   - Writes them into a new `model-lightning-skip-overlay.safetensors` shard
   - Removes the corresponding `.qweight/.qzeros/.scales` entries from `model.safetensors.index.json`
   - Adds `.weight` entries pointing at the overlay shard
   - Updates `quantize_config.json` dynamic field with `"-:model.layers.{i}.mlp.gate_proj$": True` etc

After this, SGLang loads the model:
- Dense MLPs: still W4A16 via gptq_marlin path (UnquantizedLinearMethod not triggered for them)
- Lightning MLPs: dynamic skip → UnquantizedLinearMethod → reads .weight from overlay shard → BF16

Quant-time + load-time alignment is preserved because both reference the same
list of layer indices computed from `mixer_types`.

## Expected outcome

| Metric | Direction | Why |
|---|---|---|
| Aggregate acc | ↑ (target +3-8pp) | cwe/niah unblocked, dominant failure mode addressed |
| `niah` per-task | ↑↑↑ | long-context retrieval is the hardest hit by recurrence errors |
| `cwe` per-task | ↑↑↑ | count-word-entries is even longer context |
| `qa` per-task | ↑ (modest) | benefits less, qa is less recurrence-sensitive |
| `mcq` per-task | ≈ | short context, was already best in v21 |
| `fwe` per-task | ≈ | already ~100% free credit |
| Latency | -10% (slower) | lightning MLPs in BF16 = bigger memory footprint |
| Artifact size | +30% | extra BF16 weights for ~24 lightning layers |

## Tarball structure

```
soar_w4a16_lightning_skip_<timestamp>.tar.gz
├── README_SUBMISSION.md
├── prepare_env.sh                 ← symlink → 1849's
├── prepare_model.sh               ← unique: runs quant + overlay
├── quantize_gptqmodel_w4a16.py    ← symlink → 1849's
├── apply_lightning_skip_overlay.py ← bundled (post-quant surgery)
├── perf_public_set.jsonl          ← symlink → 1849's
├── flash_attn-...whl              ← symlink → 1849's
└── sglang/                        ← symlink → 1849's
```

## Hard Constraints status

`python3 tools/hard_constraints_lint.py --variant submission_w4a16_lightning_skip`

Should report **15 PASS / 0 FAIL** (all 1849 constraints preserved + the
overlay tool doesn't violate any constraints by adding BF16 weights).

## Tunable env vars

Same as 1849 (`NUM_CALIB`, `MAX_CALIB_LEN`, `CALIB_WINDOW_MODE`,
`DISABLE_CHAT_TEMPLATE`). The overlay step is unconditional (always applied
when there are lightning layers in `mixer_types`).

## Validation

### CPU-only (no GPU needed)

Dry-run the overlay tool against any existing 1849 output to see what
it WOULD do:

```bash
python3 tools/apply_lightning_skip_overlay.py \\
    --quantized-dir /path/to/existing/quantized \\
    --bf16-dir /path/to/MiniCPM-SALA \\
    --dry-run
```

Prints lightning layer indices + estimated overlay shard size.

### GPU-side full test

```bash
bash scripts/local_eval.sh \\
    --variant submission_w4a16_lightning_skip \\
    --num-samples 150 \\
    --force-requant
```

Wall time ~30 min (20 min quant + 1 min overlay + 10 min eval @ 150 samples).
Compare per-task acc against 1849 baseline (acc_ori 46.89 platform, 49 local):
- If cwe/niah jump ≥ 10pp → strong evidence for the hypothesis, prep platform submission
- If cwe/niah barely move → revert to 1849, lightning recurrence isn't the bottleneck

## Risks

| Risk | Probability | Mitigation |
|---|---|---|
| Overlay shard format incompatible with SGLang | Low | safetensors is standard; SGLang loads multiple shards routinely |
| `mixer_types` parsing wrong (lightning markers I haven't seen) | Low | tool warns + the dry-run mode lets you preview before committing |
| Removed quantized tensors leave broken shards | Medium-low | tool only edits the JSON index, not the safetensors files themselves — original quant tensors stay readable but unreferenced |
| Dense-only quant + lightning BF16 still has the repetition collapse | Medium | hypothesis falsified; switch to repetition_penalty sampling fix or AWQ |
| Latency drop too large to clear final_score even with acc gain | Medium | accept; this variant trades latency for acc |

## Submission gate

DO NOT submit this variant until:
1. Server-side GPU validation has run (`bash scripts/local_eval.sh --variant submission_w4a16_lightning_skip --num-samples 30 --force-requant`) without crash
2. Per-task results show meaningful niah/cwe movement vs 1849
3. Aggregate acc is at least as high as 1849 (no regression)

If validation passes, this is the **strongest candidate for clearing the
correctness gate** because it directly addresses the architectural
hypothesis behind the repetition collapse.
