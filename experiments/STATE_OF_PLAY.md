# SOAR 2026 W4A16 — STATE OF PLAY

> **Single-source-of-truth doc.** Any future Claude session opens this
> first. Last updated: 2026-05-22 ~03:10 (Stream A-E + extras complete).

## 30-second TL;DR

We're trying to make W4A16 quantization (via GPTQModel) produce a
non-zero accuracy on the SOAR platform. The MiniCPM-SALA 9B baseline
on the platform scores 19.13 (BF16). RTN W4A16 scores 42. Our
GPTQModel-based W4A16 submissions (v17/v21/v22) all scored **0**.

We identified two prime hypotheses (H1 qzeros, H4 tokenizer), fixed
both, and submitted v23 ~02:35 local on 2026-05-22. v23 platform
result expected ~07:35 local.

While waiting, we pre-prepped 6 contingency variants (`v24_*`, `v25_*`)
and an auto-iterate script so the morning iteration is one command.

**Morning playbook**: `experiments/MORNING_PLAYBOOK.md` → 5 steps,
center is `bash scripts/v24_auto.sh`.

## Submission history (canonical: see SUBMISSIONS.md)

| Tarball | Module set | Platform acc | Notes |
|---|---|---|---|
| RTN-scalefix | Full (numpy, no GPTQModel) | **42** | Baseline-to-beat. Tokenizer copied unconditionally from base. |
| v17 | Full-attn GPTQ | 0 | First end-to-end run. No qzeros fix, no tokenizer overwrite. |
| v21 | MLP-only GPTQ + qzeros fix v1 | 0 (local 49) | Old fix_qzeros only globbed model-*.safetensors. |
| v22 | Full-attn GPTQ + qzeros fix v1 | 0 | Same broken qzeros fix as v21. |
| **v23** | MLP-only GPTQ + hardened qzeros + tokenizer overwrite | **submitted 2026-05-22 ~02:35**, awaiting | First submission with H1+H4 fixes baked in. |

## Hypotheses (current state)

| H# | Description | Status | Fix commit | Test sentinel |
|---|---|---|---|---|
| H1 | `fix_qzeros_for_marlin()` silently no-op'd on platform — old function only globbed `model-*.safetensors`, required exact `int32` dtype, exact `unique == [0x77777777]` | **Fixed and locally verified** | `d8aaaf1e4` | `dequant_one_tensor.py` PASS on layer 0/15/31; Exp I local 47.24 (~ noise from v21's 49) |
| H2 | gptqmodel `>=7.0,<8.0` install gate skipped install if any 7.x present; platform may have had different .x | **Fixed (pinned 7.0.0)** | `d8aaaf1e4` | `pack_submission --check-only` enforces `GPTQMODEL_PIN` canary |
| H3 | Python ABI mismatch (AutoDL 3.12.13 vs platform 3.10) — gptqmodel/safetensors install different wheels | **Indirectly addressed via H2 pin** | n/a | parse_quant_diagnostic will surface platform's python version |
| H4 | GPTQModel re-serialized tokenizer with split chat_template (inline + .jinja sidecar); platform's transformers <4.47 silently ignores sidecar → empty chat_template → garbage | **Fixed (copy_runtime_assets overwrites)** | `57f9cef06` | `check_tokenizer_compat.py`; `scripts/overwrite_tokenizer_with_base.sh` |
| R1 | Platform-specific transformers/python ABI | **Awaiting v23 log for [versions] dump** | n/a | platform_v23.log [versions] block |
| R2 | Bundled SGLang vs platform's pre-installed SGLang import order | Untested, only platform log can tell | n/a | platform_v23.log SGLang init lines |
| R3 | qweight packing math wrong | **Refuted** by `dequant_one_tensor.py` PASS | n/a | already verified locally |
| R4 | calib jsonl path drift on platform | Likely refuted (prepare_model.sh resolves bundled jsonl first) | n/a | parse_quant_diagnostic confirms calib path/row count |
| R5 | Platform private set difficulty CEILING (not 0-vs-nonzero cliff) | Open; testable only after v23 != 0 | n/a | Per-task analysis after correctness fix lands |

## Fix commits (chronological, on `quant/w4a16` branch)

| Commit | Title | What |
|---|---|---|
| `d8aaaf1e4` | v21+v22 platform=0: harden fix_qzeros + pin gptqmodel + diagnostics | H1+H2 + DIAGNOSTIC prints |
| `7a53cd2ac` | tools: parse_quant_diagnostic for platform vs local log diff | Log parser tool + 22 tests |
| `5a1480678` | perf: cherry-pick chunked-prefill 65K + op-fusion verify patch | (later reverted from submission_*/, kept in run_sala.sh) |
| `509d341fd`, `450174e4d`, `2c3a941c9` | docs: handoff doc layout corrections | PLATFORM_DEBUG_HANDOFF.md path / autodl layout |
| `a4cbcfdc6` | defensive: revert chunked-prefill in submissions + local_eval sed-patch + handoff warning | Variable isolation for v23 |
| `57f9cef06` | copy_runtime_assets: overwrite GPTQModel's re-serialized tokenizer — H4 fix | H4 fix in source |
| `0d4496012` | tests + docs: 17 tests for check_tokenizer_compat + H4 version corrected | Tokenizer compat tool tests + version refinement |
| `2aa8f16e6`, `e79c2707c` | SUBMISSIONS.md: H4 tokenizer drift joins H1 qzeros as co-equal | Documentation |
| `ad3d08b5e` | pack_submission: extend validate_variant with 5 fix canaries + 16 tests | Preflight enforcement |
| `1c2c3e1c9` | eval: round-robin interleave perf_public_set.jsonl + add reorder tool | Partial-eval signal improvement |
| `fdf69c93c` | scripts/peek.sh — live status snapshot | Observability |
| `a965a8cab` | v24: independent fork dir + symlinks (not in-place mutation of v23) | v24-perf fork (chunked-prefill 65K, ready post-v23) |
| `65b5a3bf2` | pack_submission: enforce bundled sglang/python + defensive patches at preflight | Catches the v23-without-sglang issue going forward |
| `dda8fb130` (= origin's `67feb6b64` after rebase) | quant_config_validator: fix false-positive on perma-patched fp16 source | Validator now correctly handles "source already fp16" |
| `43dea6c29` | Revert fp16 sed patch on python/ — let prepare_env.sh do it on platform install | Server-side reverted perma-patch so platform sed actually fires |
| **`bc49fdf29`** | **Stream A: 5 contingency variants for v24/v25** | v24_no_dtype_key, v24_pin_transformers, v24_bits8, v25_calib_multi_adaptive, v25_g64 |
| **`572eb5fd9`** | **Stream D: decide_next_variant.py auto-iterates V24_PLAN.md** | One-line command that maps platform log → recommended next variant |
| **`7311b510e`** | **Stream E: STATE_OF_PLAY.md mega-doc** | this file (initial version; later updated in-place) |
| **`d077fef28`** | **Stream B-lite: AWQ feasibility assessment** | CONDITIONAL GO — wait for v23 outcome before building |
| **`13ad9e064`** | **scripts/v24_auto.sh** | One-shot wrapper: log → decide → preflight → pack |
| **`3089e65c9`** | **MORNING_PLAYBOOK.md + sanity_check.sh** | Single-page wake-up guide + 5-second infra verifier |
| **`866222047`** | **tests: 14 cases for decide_next_variant + regex fix** | extracts acc from JSON / dot / "Average Score" forms; greedy regex bug fixed |

(Watchdog `auto:` commits are excluded — they're snapshot artifacts, not feature commits.)

## Variant dir inventory

| Dir | Type | Tracked files | Inherited via symlink | Status |
|---|---|---|---|---|
| `submission_gptqmodel_calib_w4a16/` | v21/v23 source | README, prepare_env.sh, prepare_model.sh, quantize_gptqmodel_w4a16.py, perf_public_set.jsonl | (none — base) | **v23 packed + submitted** |
| `submission_gptq_v17_minconfig/` | v22 source (full-attn) | Same 5 files as v23, slightly different prepare_model.sh | (none — base) | v22 lineage; not submitted in this round |
| `submission_gptqmodel_calib_w4a16_v24/` | v24-perf (chunked-prefill 65K) | README + prepare_env.sh | rest → v23 | Ready, **do NOT submit until v23 acc >= 30** |
| `submission_gptqmodel_calib_w4a16_v24_no_dtype_key/` | drops dtype= config key | README + quantize_gptqmodel_w4a16.py | rest → v23 | Ready, submit if v23 acc=0 + qzeros OK + transformers>=4.47 |
| `submission_gptqmodel_calib_w4a16_v24_pin_transformers/` | forces transformers==4.57.1 | README + prepare_env.sh | rest → v23 | Ready, submit if v23 acc=0 + transformers<4.47 |
| `submission_gptqmodel_calib_w4a16_v24_bits8/` | --bits 8 (W8A16) | README + prepare_model.sh | rest → v23 | Ready, submit if W4A16 itself suspected broken |
| `submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive/` | multi-adaptive calib + 300 samples | README + prepare_model.sh | rest → v23 | Ready, submit if 0 < v23 acc < 60 |
| `submission_gptqmodel_calib_w4a16_v25_g64/` | --group-size 64 | README + prepare_model.sh | rest → v23 | Ready, submit if 0 < v23 acc < 70 |

## Open questions

1. **What's platform's transformers version?** Determines whether H4 is actually triggered. Will surface in v23's `[versions]` log block.
2. **What's platform's gptqmodel .x release?** Determines whether H1 hardened fix fired or no-oped. Will surface in v23's qzeros-fix sample lines.
3. **Why did v21 corrected_acc only reach 40% locally?** Even with H1+H4 hypothetically right, local local-AutoDL acc has a ~50pp gap to the platform's 80% gate. This is the R5 "ceiling" problem — need calibration improvements (v25_calib_multi_adaptive, v25_g64) AFTER correctness fix.
4. **Is op-fusion safe on GPU?** CPU-equivalence verified (`scripts/test_op_fusion_math.py` passed). GPU run with `MINICPM_FUSION_VERIFY=1` not yet done. Deferred to v25+ once correctness is established.
5. **Is AWQ a viable backup?** Untested, no official endorsement. May 14 llmcompressor artifact exists at `/root/autodl-fs/zyn/models/` but never benchmarked. Stream B-lite feasibility study not yet executed.

## Tooling inventory (post-2026-05-21)

| Tool | Purpose | Tests |
|---|---|---|
| `tools/analyze_predictions.py` | Per-task pass-rate + failure mode classification | n/a (pre-2026) |
| `tools/diff_predictions.py` | A/B transition matrix between two prediction sets | n/a |
| `tools/inspect_quant_artifact.py` | safetensors metadata + qzeros sanity | n/a |
| `tools/quant_config_validator.py` | Tarball-level preflight (sed targets, sed end-state) | n/a |
| `tools/parse_experiment_logs.py` | Bulk log → CSV/MD aggregator | n/a |
| `tools/build_calib_set.py`, `calib_set_preview.py` | Calib bucketing + visualization | 27 tests |
| `tools/pack_submission.py` | Variant dir → tarball, with 11-canary preflight | 18 tests |
| `tools/repetition_analyzer.py` | predictions.jsonl → loop substring patterns | n/a |
| `tools/parse_quant_diagnostic.py` | Quantize log → structured + diff | 22 tests |
| `tools/check_tokenizer_compat.py` | Base BF16 vs artifact tokenizer drift detection | 17 tests |
| `tools/compute_corrected_acc.py` | Per-task + corrected-aggregate (drop fwe) acc | n/a |
| `tools/dequant_one_tensor.py` | Math-layer dequant correctness | n/a (server-side) |
| `scripts/local_eval.sh` | One-shot variant → quant → server → eval | n/a |
| `scripts/peek.sh` | Live snapshot of running pipeline | n/a |
| `scripts/overwrite_tokenizer_with_base.sh` | Apply H4 fix to existing artifact | n/a |
| `scripts/reorder_eval_set.py` | Round-robin interleave perf_public_set.jsonl | n/a |
| `scripts/v23_full_pipeline.sh` | One-shot wrapper: preflight → pack → submit prep | n/a |
| `scripts/decide_next_variant.py` | v23 platform log → recommended v24 variant | **14 tests** |
| `scripts/v24_auto.sh` | **decide + preflight + pack in one command** | smoke-tested via dry-run |
| `scripts/sanity_check.sh` | 5-sec infra health verifier (tests + preflights + CLI + bash -n) | self-validating |

Total: **102 unit tests pass** (`python3 -m unittest discover tests`).

## Documents inventory

| Path | Purpose |
|---|---|
| `SUBMISSIONS.md` | Canonical chronological log of every submission + hard constraints |
| `experiments/PLATFORM_DEBUG_HANDOFF.md` | "Server-side Claude reads this first" cheat sheet |
| `experiments/V23_PLATFORM_LOG_CHECKLIST.md` | Block-by-block grep guide for the v23 platform log |
| `experiments/V23_VS_RTN_TARBALL_AUDIT.md` | File-level diff between RTN-pass and v22-fail tarballs |
| `experiments/V24_PLAN.md` | Decision tree by v23 outcome; per-branch concrete commands |
| `experiments/AWQ_FEASIBILITY_ASSESSMENT.md` | Stream B-lite output: CONDITIONAL GO on AWQ. Build only if v23 < 10. |
| `experiments/MORNING_PLAYBOOK.md` | **Half-awake guide.** 5 steps when v23 result arrives. Start here. |
| `experiments/PLAN.md` | Night-work runbook (partially stale; predates H4) |
| `experiments/op_fusion_verify.patch` | Drop-in patch for `perf/op-fusion` GPU validation |
| `experiments/STATE_OF_PLAY.md` | **(this file)** Single-source-of-truth for any new session |
| `docs/calibration_gotchas.md` | qzeros bug, truncation-side, repetition collapse, fp16 sed |
| `docs/awq_fallback_plan.md` | 3-step AWQ alt (pre-feasibility-check) |
| `NIGHT_WORK_SUMMARY.md` | 2026-05-21 overnight changes (multi-adaptive calib + bug fixes) |
| `W4A16_README.md` | High-level competition + variant intro |
| `memory/MEMORY.md` (auto) | Auto-loaded user memory index |
| `memory/*.md` | Auto-loaded per-fact memory files |

## What to do when v23 platform result arrives

**Easy path** (recommended): one command via the wrapper script:

```bash
cd /root/soar/sglang
git pull origin quant/w4a16

# Save the platform log to NAS first (survives instance swaps)
cp /path/to/downloaded_log.txt /root/autodl-fs/zyn/logs/platform_v23.log

# One-shot decision + pack
bash scripts/v24_auto.sh
# Use --dry-run first to preview, --yes to skip y/N confirm
```

The wrapper reads the log, applies V24_PLAN.md's decision tree, prints
the recommendation + rationale, asks y/N, then runs preflight + pack.
Output: a ready-to-upload tarball + the path.

**Detail path** (if v24_auto fails or you want to inspect): see
`experiments/MORNING_PLAYBOOK.md` for the 5-step manual walkthrough.

## What to NOT do

- Don't modify `submission_gptqmodel_calib_w4a16/` or
  `submission_gptq_v17_minconfig/` directly. v23 is packed from them
  and any future re-pack must produce a byte-equivalent tarball.
- Don't submit any variant before reading its `README_SUBMISSION.md`
  precondition — most are gated on a specific v23 outcome.
- Don't `git push --force` or rewrite history. Watchdog auto-commits are valuable as time-series snapshots.
- Don't try AWQ blind. Run Stream B-lite feasibility check first.

## Branch state (as of this writing)

- Current branch: `quant/w4a16`
- Most recent feature commit: `866222047 tests: 14 cases for decide_next_variant + regex fix`
- Watchdog `auto:` commits running every 5 min on the server side
- v23 tarball: submitted ~02:35 to SOAR platform
- Server stops at 03:20 — local re-quant validation should be in progress
- Platform result ETA: ~07:35

## Quick verification before any work

If anything in this doc feels stale, verify with:

```bash
bash scripts/sanity_check.sh
# expect: 102 tests + 8 variant preflights + 5 CLI + 6 shell all green
```

If sanity_check fails, the doc is more reliable than the code state.
Re-read the failing component's recent commits.
