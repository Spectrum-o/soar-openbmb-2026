# SOAR W4A16 Submission Log & Lessons Learned

> Living document. Every platform submission, what failed, the root cause, and
> the fix that was made. **Read this BEFORE building a new submission** so we
> don't re-hit a known constraint.

---

## Scoreboard

| Metric | Value | Notes |
|---|---|---|
| **Baseline (BF16, no extra args)** | **19.13** | Number to beat |
| **Best score so far (with quant)** | — | None of our quant attempts has cleared correctness gate |
| **W4A16 attempts** | 19+ | See chronological log below |
| **Slots consumed today (2026-05-20)** | 1 confirmed (v17 ran end-to-end with acc=0) | Early prepare/startup failures appear not to consume the full slot, but check platform dashboard for the authoritative count |
| **Slots remaining today** | ? | Check platform dashboard |

---

## Hard constraints (verified by failure)

These are platform/library facts you cannot work around without major surgery.
Each is linked to the submission that proved it.

### Quantization config

| Constraint | Source of truth | Verified by |
|---|---|---|
| Marlin (`--quantization gptq_marlin`) only supports `(4, True)` → `uint4b8` | `gptq.py:223` `TYPE_MAP` | 2026-05-19 RTN with `sym=False` rejected at startup |
| For uint4b8: `scale = max(abs(w)) / (2^(bits-1) - 1)` = `/ 7`, NOT `/ 8` | dequant = `(q_unsigned - 8) * scale`, so max = `7 * scale` | 2026-05-19 RTN with `/ half` gave acc_ori = 42.51 → 0 final_score |
| RTN at 4 bits is fundamentally too lossy for SALA — need Hessian (GPTQ) | local roundtrip test (`scripts/test_quant_roundtrip.py`) | RTN `/7` fix only raised acc to 42.18 (~no improvement) |

### Model / inference path

| Constraint | Why | Verified by |
|---|---|---|
| FP8 KV cache (`--kv-cache-dtype fp8_*`) is INCOMPATIBLE with MiniCPM sparse backend | `minicpm_flashinfer` uses FlashAttention internally; FA only supports fp16/bf16 | 2026-05-15 gptqmodel_marlin_fp8kv failures |
| `--disable-cuda-graph` is for debugging only; baseline runs WITH CUDA graph | Baseline 19.13 has CUDA graph enabled | 2026-05-15 plain GPTQ + `--disable-cuda-graph` hit 5h timeout |
| SALA's HF modeling code hard-asserts `_attn_implementation == "flash_attention_2"` | `modeling_minicpm_sala.py:1328` MiniCPMInfLLMv2Attention.__init__ | 2026-05-19 21:16 gptqmodel v3 AssertionError |
| `transformers >= 5.0` does a HARD import check of `flash_attn` package at PreTrainedModel.__init__ | `modeling_utils.py:1714` `_flash_attn_import_error` | 2026-05-19 21:16 gptqmodel v3 ImportError |
| GitHub release downloads can hang on the platform's network — bundle wheels in the tarball | SOAR cloud network can't reach github.com reliably | 2026-05-19 21:48 gptqmodel v4 PREPARING stuck on download |
| `apply_torchao_config_to_model` does eager import of torchao APIs that were removed in 0.16.0 | `torchao_utils.py:46` imports `float8_dynamic_activation_float8_weight` etc. before checking if config is empty | 2026-05-20 12:13 v15 ImportError |
| Removing `auto_map.AutoConfig` is required for SGLang's `MiniCPMHybridConfig` to win the `isinstance` check | `model_runner.py:1494` `minicpm_hybrid_config` property + `hybrid_linear_attn_backend.py:1456` assertion | 2026-05-20 inferred from source trace |
| Do **not** serialize derived config properties such as `has_sparse_attention` into `config.json` | SGLang's `MiniCPMHybridConfig` exposes them as read-only `@property` values derived from `sparse_config`/`mixer_types`; `AutoConfig.from_pretrained` crashes if config.json tries to assign them | 2026-05-20 15:25 v18 `AttributeError: can't set attribute 'has_sparse_attention'` |

### GPTQModel-specific

| Constraint | Why | Verified by |
|---|---|---|
| `MODEL_MAP["minicpm_sala"] = ...` is NOT enough to register | `gptqmodel/models/auto.py:325` `SUPPORTED_MODELS = list(MODEL_MAP.keys())` is built ONCE at import time | 2026-05-20 v10/v11/v12 `auto_detect_module_tree` ate o_gate from layer 0 |
| Must ALSO mutate `gptqmodel.models.auto.SUPPORTED_MODELS` after registration | (same as above) | v13+ |
| SALA's `MiniCPMSALADecoderLayer` is the correct `layer_type` value | All 32 layers (both minicpm4 and lightning-attn types) use this class | Verified via HF modeling source fetch |
| GPTQModel `BaseQModel` is the base class name (NOT `BaseGPTQModel`) | `gptqmodel/models/base.py` | 2026-05-20 11:42 v13 ImportError typo |
| GPTQModel 7.0 passes unrecognized **kwargs to model `__init__` via auto_factory.from_config | `auto.py:505` → `loader.py:623` → `from_config(**kwargs)` → model `__init__(**kwargs)` | 2026-05-19 20:30 v1 `max_memory` TypeError |
| Therefore NEVER pass non-model kwargs (`max_memory`, `attn_implementation`) directly to GPTQModel.load — monkey-patch upstream instead | (same as above) | v2-v17 |

### SGLang gptq_marlin loader

| Constraint | Why | Verified by |
|---|---|---|
| SGLang and GPTQModel must agree on which modules are quantized vs unquantized | If GPTQModel omits a module but SGLang initializes it with a different quantization method, state-dict names diverge and weight loading fails | 2026-05-20 12:13 v14 `KeyError: model.layers.0.self_attn.o_gate.weight` |
| Use `dynamic: {"-:.*<mod>$": True}` in quantize_config.json to skip modules at SGLang load time | `utils.py:248` `get_dynamic_override` returns False → `UnquantizedLinearMethod` | v15+ |

### Calibration (the open question)

| Observation | Implication |
|---|---|
| `perf_public_set.jsonl` has 150 rows, many prompts are long | First-4K calibration may be weak for some tasks, but this is still a hypothesis, not the proven cause of v17's acc=0 |
| RTN got `acc_ori≈42`, while full-attention GPTQ got `acc_ori=0` | The evidence points at structural sensitivity/loader compatibility of full attention quantization at least as strongly as calibration quality |

---

## Chronological submission log

Status legend:
- 🟢 passed correctness ≥ 80 + produced final_score > 0
- 🟡 passed correctness but low score
- 🔴 ran 5h pipeline, score=0 (slot **consumed**)
- ⛔ early crash before pipeline ran (slot **NOT** consumed)
- ⏰ 5h timeout (slot **consumed**)
- 📦 built but not yet submitted

| Date | Time | Tarball | Status | Wall | acc_ori | Notes / root cause |
|---|---|---|---|---|---|---|
| 2026-05-14 | — | `soar_w4a16_submission_20260514.tar.gz` | 🔴 | — | — | Earliest W4A16 attempt; superseded |
| 2026-05-15 | — | `soar_stable_gptq_rtn_submission_20260515.tar.gz` | ⏰ | 5h | — | Plain GPTQ + `--disable-cuda-graph` + RTN `sym=False`; ran into 5h timeout |
| 2026-05-15 | — | `soar_gptqmodel_marlin_fp8kv_submission_20260515.tar.gz` | 🔴 | — | — | GPTQModel + FP8 KV — FlashAttention rejects FP8 dtype |
| 2026-05-15 | — | `soar_gptqmodel_marlin_fp8kv_submission_20260515_v2.tar.gz` | 🔴 | — | — | Retry of above — same FP8 KV incompatibility |
| 2026-05-19 | — | `soar_official_rtn_w4a16_g128_marlin_cfg_submission_20260519.tar.gz` | ⛔ | <1m | — | RTN script wrote `sym=False`, Marlin rejected at config parse |
| 2026-05-19 | 08:54-10:51 | `soar_rtn_sym_w4a16_g128_marlin_submission_20260519.tar.gz` (rtn_sym v1) | 🔴 | 2h | 42.51 | sym=True fix worked but scale formula was `/ half` (=8) instead of `/ (half-1)` (=7); clamped +ve outliers ~12.5% |
| 2026-05-19 | 12:50-14:48 | `..._20260519_scalefix.tar.gz` (rtn_sym v1.1) | 🔴 | 2h | 42.18 | `/ (half-1)` scale-formula fix did NOT help. Confirms RTN itself is too lossy for SALA |
| 2026-05-19 | 20:30 | `soar_gptqmodel_calib_w4a16_submission_20260519.tar.gz` (gptq v1) | ⛔ | 14s | — | `TypeError: ...__init__() got an unexpected keyword argument 'max_memory'`. GPTQModel passed default `cpu_max_mem` through to SALA constructor |
| 2026-05-19 | 20:46 | `..._v2.tar.gz` (gptq v2) | ⛔ | 17s | — | `AssertionError: Only flash_attention_2 is supported for sparse attention` (modeling_minicpm_sala.py:1328) |
| 2026-05-19 | 21:16 | `..._v3.tar.gz` (gptq v3) | ⛔ | 17s | — | monkey-patched `_attn_implementation`, but transformers 5.x does eager `import flash_attn` at PreTrainedModel.__init__. flash_attn package missing |
| 2026-05-19 | 21:48 | `..._v4.tar.gz` (gptq v4) | ⛔ stuck | ~10m+ | — | prepare_env tried to `uv pip install` flash_attn wheel from GitHub; network too slow / unreachable from platform; eventually timed out the PREPARING phase |
| 2026-05-20 | 00:30 | `..._v5.tar.gz` | (not submitted as v5 directly) | — | — | Bundled the 242 MB flash-attn 2.8.3 cu128torch2.9 cp310 wheel into the tarball |
| 2026-05-20 | (various) | gptq v6-v12 | ⛔ | <5m | — | `ValueError: layer module item self_attn.o_gate not found in model`. GPTQModel fell back to `BaseQModel + auto_detect_module_tree` because `SUPPORTED_MODELS` (snapshotted at import time) didn't include "minicpm_sala". Auto-detector picked up o_gate from layer 0 ("minicpm4") and crashed on layer 1 ("lightning-attn") which doesn't have it |
| 2026-05-20 | 11:42 | `..._v13.tar.gz` | ⛔ | ~5m | — | `SUPPORTED_MODELS` mutation added; got past register but `ImportError: cannot import name 'BaseGPTQModel'` — typo; should be `BaseQModel` |
| 2026-05-20 | 11:53 | `..._v14.tar.gz` | ⛔ | ~19m | — | **Quantization FINISHED for the first time** (19 min). SGLang reached weight loading, then crashed: `KeyError: model.layers.0.self_attn.o_gate.weight`. GPTQModel and SGLang disagreed on the quantized/unquantized module set; fixed by writing matching `dynamic` skip rules |
| 2026-05-20 | 12:35 | `..._v15.tar.gz` | ⛔ | ~16m | — | Added `dynamic: {"-:.*o_gate$": True, ...}` to quantize_config.json. **Weight load succeeded** (6.53 GB, matches W4A16 expected). Crashed at `apply_torchao_config_to_model` because torchao ≥ 0.16 removed `float8_dynamic_activation_float8_weight` (and SGLang's torchao_utils.py imports it eagerly before the `if torchao_config == "": return` early-return) |
| 2026-05-20 | ~12:50 | `..._v16.tar.gz` | (intermediate, superseded by v17) | — | — | torchao early-return patch applied |
| 2026-05-20 | 13:01-14:25 | `..._v17.tar.gz` | 🔴 acc | 5h | 0.0 | **FULL PIPELINE COMPLETED** end-to-end for first time. Added: input-config preservation + explicit `has_sparse_attention=True` + removed `auto_map.AutoConfig`. SGLang launched, served 256 prompts, all three bench tiers ran. **But model output was completely broken (acc=0.0).** Most likely causes: full q/k/v/o attention quantization sensitivity or SGLang loader/layout incompatibility; calibration truncation remains an open hypothesis. Bench timings: S1=625s, S8=997s, Smax=2290s. Slot consumed |
| 2026-05-20 | 15:12 | `soar_bf16_chunk32k_safetynet_submission_20260520_v2.tar.gz` | ⛔ | 13s | — | BF16 safetynet bundled SGLang source; crashed early with transformers model_type list dump (truncated error). Hypothesis: our bundled SGLang version has some incompatibility with base env's libraries that baseline (which uses base env's SGLang) doesn't have |
| 2026-05-20 | TBD | `soar_bf16_chunk32k_safetynet_submission_20260520_v3.tar.gz` | 📦 | — | — | **v3 = 2 KB safetynet without bundled SGLang.** Uses base env's SGLang. Only diff from baseline: `--chunked-prefill-size 32768 --max-prefill-tokens 32768 --enable-mixed-chunk` flags. Expected result is unverified; do not assume score 22-26 without a platform/local run |
| 2026-05-20 | 15:25-15:37 | `soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v18.tar.gz` | ⛔ | ~12m | — | MLP-only GPTQ got through preparation, then SGLang startup failed: `AttributeError: can't set attribute 'has_sparse_attention'`. Root cause: output `config.json` serialized a read-only derived property from `MiniCPMHybridConfig` |
| 2026-05-20 | 15:42 | `soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v19.tar.gz` | 📦 | — | — | Fixes v18 by removing derived read-only config keys from quantized config and adding defensive `kwargs.pop(...)` in bundled `MiniCPMHybridConfig`; still MLP-only GPTQ |
| 2026-05-20 | 15:50 | `soar_bf16_baseline_match_submission_20260520_v4.tar.gz` | 📦 | — | — | Baseline-match safetynet tarball built. Purpose: verify that identity BF16 + official default args still reproduces baseline before testing extra flags |
| 2026-05-20 | 16:30 | `soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v20.tar.gz` | ⚠️ STALE | — | — | Built BEFORE qzeros root-cause fix (22:08) and calibration fix (21:50). **Do NOT submit.** Superseded by v21 |
| 2026-05-20 | 22:08 | (no tarball — root-cause commit) | — | — | — | **ROOT CAUSE FOUND** for v6-v17 acc=0: gptqmodel 7.0.0 + sym=True writes `qzeros=7` per slot but Marlin dequant expects `qzeros=8`. Every weight gets `+1*scale` bias → garbage output. Fix in `fix_qzeros_for_marlin()` (commit `324ea90d2`) post-processes safetensors `0x77777777` → `0x88888888`. Idempotent. |
| 2026-05-20 | 22:59 | (local_eval, no tarball) | — | ~13m | 63.33 (30/150) | v18 quant + qzeros in-place patch. First non-zero GPTQ result. 30-of-150 sample subset on `perf_public_set.jsonl`. Confirms qzeros fix was the right answer; remaining gap to 80-gate must come from calibration / module-set choices |
| 2026-05-20 | 23:36 | `soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v21.tar.gz` | 🔴 acc | 5h | 0.0 | MLP-only GPTQ + qzeros fix + calib fix baked into the quantizer. **Platform acc=0** despite local acc=49 on full 150 perf_public_set. Bench timings S1=626 S8=998 Smax=2290 — identical to v17 → SGLang served + generated tokens normally, content was systematically broken. Triggered the platform-debug pre-processing turn 2026-05-21. |
| 2026-05-20 | 23:50 | `soar_gptqmodel_v17_minconfig_full_attn_submission_20260520_v22.tar.gz` | 🔴 acc | 5h | 0.0 | Full-attn GPTQ + same qzeros+calib fixes as v21. **Platform acc=0**, same bench timings as v17/v21 (S1=626 S8=998 Smax=2290). v22 ALSO = 0 kills "MLP-only is the bug" hypothesis dead — issue is in the GPTQModel pipeline itself, not the module set. See `experiments/PLATFORM_DEBUG_HANDOFF.md` for the hardened response. |
| 2026-05-22 | 02:39 | `soar_gptqmodel_calib_w4a16_mlp_only_submission_20260522_v23.tar.gz` | ⛔ self-bug | 39m | — | **v23 = MLP-only + hardened qzeros + H4 tokenizer overwrite**. Platform FAILED at prepare_model step, NOT during eval. Root cause: my own `POST-CHECK` assertion in `fix_qzeros_for_marlin()` checked `shards[0]` blindly — but for MLP-only artifacts shard 0 contains embedding+lm_head with NO .qzeros tensors at all → the for-loop completed without `verified=True` → spurious FATAL. The qzeros patch itself succeeded (96/96 tensors correctly 0x77→0x88 in shards 2-3). **The slot tested my assertion code, not H1/H4 hypotheses.** Fixed in commit `551826514` (POST-CHECK now iterates `per_shard_patched` list, not blind shards[0]). v23b = repack from same source with the fix. Status: ⛔ (slot lost to my bug; H1/H4 still unverified on platform). |

---

## Open questions / next experiments

### 🚨 2026-05-22 update: H4 (tokenizer drift) joins H1 (qzeros) as co-equal prime hypotheses

Two independent failure modes uncovered in the 2026-05-21..22 debug session, EITHER OF WHICH ALONE could explain v17/v21/v22 platform=0:

**H1 — `fix_qzeros_for_marlin()` silently no-op'd on the platform**. Documented in the 2026-05-21 update below. Fix: hardened in commit `d8aaaf1e4` (glob `*.safetensors`, accept uint32, per-element replacement, HARD EXIT on suspicious states, POST-CHECK reload sanity).

**H4 — `GPTQModel.save()` re-serializes tokenizer; older transformers reads only inline chat_template.** Discovered 2026-05-22 by server-side analysis; mechanism verified against transformers v4.46.0 vs v4.47.0 source:
- GPTQModel's `model.save()` calls HF `tokenizer.save_pretrained()`, which under newer tokenizers library splits the chat_template into BOTH inline `tokenizer_config.json` AND a separate `chat_template.jinja` sidecar.
- The sidecar reader was added in **transformers v4.47.0** (PR #33957, "Separate chat templates into a single file", 2024-12-05). See `src/transformers/tokenization_utils_base.py:108` `CHAT_TEMPLATE_FILE = "chat_template.jinja"`.
- On `>= 4.47`, the sidecar **wins over** the inline field.
- On `< 4.47`, the sidecar is **silently ignored** (no error, just not read).
- AutoDL has transformers 4.57.1 → reads sidecar; platform (suspected `< 4.47`) → reads inline. The local-49 vs platform-0 gap implies GPTQModel wrote a *correct* sidecar but a *broken-or-different* inline.
- `tokenizer.json` also grows from 3.6M (base) to 6.7M (GPTQModel-rewritten); `added_tokens.json` appears where it wasn't in base → vocab encoding may also have drifted.

This explains RTN-passes-at-42 vs every-GPTQModel-fails-at-0 perfectly: RTN's `quantize_gptq_rtn_sym.py:107` (`copy_metadata`) unconditionally `shutil.copy2()` overwrites tokenizer files from the input dir AFTER its numpy quant step → RTN's artifact has the base model's tokenizer byte-for-byte. v17/v21/v22's `quantize_gptqmodel_w4a16.py:976` (`copy_runtime_assets`) had an `if not target.exists()` guard that silently kept GPTQModel's bloated re-serialization. Fix: removed the guard in commit `57f9cef06`.

### 2026-05-22 experiments

| Exp | Hypothesis tested | Result | Verdict |
|---|---|---|---|
| **Exp H** | chunked-prefill-size 65536 vs 8192 affects acc | v21 artifact + 65K → 49.00; same artifact + 8K → 48.78 | -0.22pp inside noise → **chunked-prefill is NOT the platform=0 cause**. v23 ships with 8192 for variable isolation; the +83% throughput config in `run_sala.sh` is kept for local bench but withheld from submission tarballs until v23 confirms the quant pipeline fixes |
| **Exp I** | tokenizer overwrite from base BF16 changes local acc | running 2026-05-22 (servervside) | Expected local ≈ 49 (AutoDL transformers 4.57.1 reads .jinja sidecar, so local behavior is similar regardless of which tokenizer is on disk). Real H4 test is the platform |

### 2026-05-22 fixes landed on `quant/w4a16`

| Commit | Fix | Hypothesis | Test sentinel |
|---|---|---|---|
| `d8aaaf1e4` | Hardened `fix_qzeros_for_marlin()` + `gptqmodel==7.0.0` pin + `prepare_model.sh` DIAGNOSTIC blocks | H1 + H2 | `tools/parse_quant_diagnostic.py` (22 tests) parses the SUMMARY block; `tools/pack_submission.py --check-only` rejects variants missing the POST-CHECK canary |
| `57f9cef06` | `copy_runtime_assets` removes `if not target.exists()` guard; canary docstring `CRITICAL 2026-05-22` added | H4 | `tools/pack_submission.py --check-only` rejects variants missing the canary; `tools/check_tokenizer_compat.py` diffs base vs artifact; `scripts/overwrite_tokenizer_with_base.sh` applies the fix to an existing artifact without re-quant |
| `a4cbcfdc6` | `scripts/local_eval.sh` runtime sed-patch + revert; `submission_*/prepare_env.sh` chunked-prefill reverted from 65536 to 8192 (variable isolation); PLATFORM_DEBUG_HANDOFF.md "Hypothesis discipline" warning | n/a (infra) | `tests/test_pack_submission_validate.py:test_chunked_prefill_65k_flagged` |
| `ad3d08b5e` | `tools/pack_submission.py` preflight check from 2 → 7 canaries; 16 new tests | n/a (infra) | 65 total unit tests in `tests/` all pass |

### Why did v17 produce acc=0?

The pipeline ran successfully. Output was garbage. Hypotheses ranked:

1. **🔴 HIGHEST: full attention/Lightning quantization is structurally unsafe**
   - v17 quantized q/k/v/o + MLP and served successfully, but `acc_ori=0`
   - RTN W4A16 still got ~42, so "any 4-bit quantization always gives zero" is false
   - **Fix to try**: MLP-only GPTQ (v18/v19) before adding attention back per layer type

2. **🟡 MEDIUM: calibration truncation hurt GPTQ**
   - perf_public_set prompts can be very long, and v17 used first-token truncation
   - This may produce weak Hessian statistics, but it is not proven as the main cause until MLP-only is tested
   - **Fix to try if MLP-only still fails correctness**: compare first-4K vs last-4K vs mixed span calibration

3. **🟡 MEDIUM: removing `auto_map.AutoConfig` changed config semantics**
   - Required for SGLang's `MiniCPMHybridConfig` path and SimpleGLA checks, so this is not an easy revert
   - Could still affect defaults subtly; verify by comparing parsed config fields rather than toggling blindly

4. **🟢 LOW: fp16 vs bf16 patch accumulating error**
   - SALA trained in bf16, we run in fp16 + sed-patched sparse backend
   - RTN was also fp16 and got acc=42, not 0 → unlikely main cause

### Current direction (2026-05-20 ~15:00 onward)

User updated `submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py` to do **MLP-only quantization** (no attention quantization). Reasoning per docstring update:
> "full q/k/v/o W4A16 successfully served but produced acc=0 on the platform, so this route is intentionally MLP-only"

This sidesteps the gating-mismatch hypothesis and tests whether attention quantization specifically is the problem. Expected outcomes:
- ✅ acc clears 80 → attention quantization was the issue; can iteratively add it back per-layer-type
- ❌ acc still 0 → calibration or config/loader semantics remain suspect; next experiment should compare calibration windows and inspect sample outputs before another blind platform submission

---

## How to use this document

1. **Before building a new tarball**: scan "Hard constraints" + "Open questions"
2. **When a submission fails**: add a new row to the chronological log with status, wall time, acc_ori, root cause
3. **When you learn a new constraint**: add it to "Hard constraints" with a link to the failure that proved it

The Codex hand-off lives in `/home/zyn/.claude/projects/.../memory/` for me; this document is the human-readable source of truth.

---

## Inventory of tarballs at repo root

Quick `ls *.tar.gz` cheat sheet (May 2026):

```
soar_w4a16_submission_20260514.tar.gz                           # 5/14 earliest
soar_stable_gptq_rtn_submission_20260515.tar.gz                 # 5/15 plain GPTQ, timeout
soar_gptqmodel_marlin_fp8kv_submission_20260515.tar.gz          # 5/15 FP8 KV broken
soar_gptqmodel_marlin_fp8kv_submission_20260515_v2.tar.gz       # 5/15 same
soar_rtn_sym_w4a16_g128_marlin_submission_20260519.tar.gz       # 5/19 RTN v1, acc=42.51
soar_rtn_sym_w4a16_*_v{2,3,4}*_20260519.tar.gz                  # 5/19 RTN variants, same bug
soar_rtn_sym_w4a16_*_20260519_scalefix.tar.gz                   # 5/19 RTN scalefix, acc=42.18
soar_gptqmodel_calib_w4a16_submission_20260519.tar.gz           # 5/19 evening gptq v1, max_memory bug
soar_gptqmodel_calib_w4a16_submission_20260519_v{2..5}.tar.gz   # 5/19-5/20 various crashes
soar_gptqmodel_calib_w4a16_submission_20260520_v{6..17}.tar.gz  # 5/20 the o_gate / torchao saga
soar_bf16_chunk32k_safetynet_submission_20260519.tar.gz         # 5/19 safetynet original
soar_bf16_chunk32k_safetynet_submission_20260520_v2.tar.gz      # 5/20 safetynet w/ sglang (crashed)
soar_bf16_chunk32k_safetynet_submission_20260520_v3.tar.gz      # 5/20 safetynet WITHOUT sglang (extra flags, unverified)
soar_bf16_baseline_match_submission_20260520_v4.tar.gz          # 5/20 identity BF16 + official default args
soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v18.tar.gz # 5/20 MLP-only GPTQ, startup failed on read-only config property
soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v19.tar.gz # 5/20 MLP-only GPTQ, fixes v18 config serialization bug
soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v20.tar.gz # 5/20 16:30 STALE — built before qzeros fix; do NOT submit
soar_gptqmodel_calib_w4a16_mlp_only_submission_20260520_v21.tar.gz # 5/20 23:36 MLP-only GPTQ + qzeros fix + calib fix; acc TBD
soar_gptqmodel_v17_minconfig_full_attn_submission_20260520_v22.tar.gz # 5/20 23:50 FULL-ATTN GPTQ + qzeros fix + calib fix; use if v21 hits 70-80
```
