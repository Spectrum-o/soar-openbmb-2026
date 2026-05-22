# Handoff — 2026-05-22 evening

> **You (zyn) are looking at this because you came back from being away.**
> Claude on branch `parallel/non-gptqmodel-paths` did some work for you.
> Read this first, then act.

## TL;DR — what to do FIRST

1. **Check 1849 platform result** on SOAR web UI. Save log to `/root/autodl-fs/zyn/logs/platform_v24_pin_transformers.log`.
2. **Submit BF16 v3 safetynet IMMEDIATELY regardless of 1849 result.** This is the rank-20 floor. Path:
   ```
   /home/zyn/.../sglang/soar_bf16_chunk32k_safetynet_submission_20260520_v3.tar.gz
   ```
3. Based on v3's result + 1849's result, pick next action from the decision tree below.

## State of this branch (`parallel/non-gptqmodel-paths`)

```
$ git status
On branch parallel/non-gptqmodel-paths
modified:   (same uncommitted v24_pin_transformers files as quant/w4a16)

untracked (new from this session):
    experiments/BF16_SAFETYNET_AUDIT_20260522.md     ← audit of v3 tarball
    experiments/HANDOFF_20260522_evening.md          ← THIS FILE
    scripts/test_llmcompressor_artifact.sh           ← runs the 5/14 W4A16 artifact
    submission_bf16_op_fusion/                       ← stretch variant, needs GPU validation
        ├── README_SUBMISSION.md
        ├── prepare_env.sh
        ├── prepare_model.sh
        └── sglang_overlay/minicpm.py
```

**No tarballs were packed.** Submissions are your call.

**`quant/w4a16` branch is untouched.** If you `git checkout quant/w4a16`, you get the same uncommitted v24_pin_transformers state you left it in. The 1849 tarball at repo root is unchanged.

## Decision tree (read after 1849 result is in)

### Case 1: 1849 returned acc=0 (still)

Most likely outcome statistically (per `feedback_h5_transformers_runtime_drift.md`, H5 is still untested and could be wrong). What it means:

- **transformers pin is not the fix.** We've exhausted the GPTQModel hypothesis space (H1 qzeros, H4 tokenizer, H5 transformers).
- **The W4A16 GPTQModel path is effectively dead** for our purposes.

What to do:
1. **Submit BF16 v3 safetynet now** to get any non-zero score. Expected ~25-30.
2. **Run `bash scripts/test_llmcompressor_artifact.sh`** on AutoDL to evaluate the 2026-05-14 llm-compressor artifact locally. If it scores > 42 locally, we have a viable alternative quant tool.
3. **Read `experiments/BF16_SAFETYNET_AUDIT_20260522.md`** for context on v3 risks + v4 fallback.

### Case 2: 1849 returned weight-load shape mismatch (same as v23b)

H5 partially falsified. transformers pin reaches SGLang load but shape still mismatches.

Next suspect (per `experiments/STATE_OF_PLAY.md` open questions):
- H4 `.py` overwrite (copy_runtime_assets writing modeling_minicpm_sala.py over GPTQModel's saved version)
- The hardened qzeros fix introducing a shape change we missed

What to do:
1. **Submit BF16 v3 safetynet** (still your floor)
2. Then: add diagnostic message to `python/sglang/srt/layers/parameter.py:285` to print tensor name on shape assert (proposed earlier this session, never committed). This is a quant/w4a16 commit, NOT here.
3. Then: build a "H4 narrow" variant (revert `.py` overwrite, keep tokenizer JSON overwrite) to test if the .py overwrite is the culprit.

### Case 3: 1849 returned non-zero acc!

🎉 Big win regardless of exact number.

What to do:
1. **DO NOT submit BF16 v3 yet** — it would overwrite this score if v3 < W4A16 acc
2. Instead: read `experiments/V24_PLAN.md` decision tree for what variant to submit next based on the acc value
3. If acc > 30: try `submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive/`

### Case 4: 1849 returned new failure mode

(install error / preparation crash / something we haven't seen)

What to do:
1. **Submit BF16 v3 safetynet** to lock the floor
2. Then: diagnose new failure with traceback. Most install failures are now fast-fail thanks to fix2's hard-guard + smoke test.

## All four submission options at your disposal (ordered by recommendation)

| Tarball | Risk | Expected final_score | Already packed? | Recommendation |
|---|---|---|---|---|
| `soar_bf16_chunk32k_safetynet_submission_20260520_v3.tar.gz` | low | 25-30 | ✅ | **Submit ASAP** unless 1849 returned non-zero |
| `soar_bf16_baseline_match_submission_20260520_v4.tar.gz` | very low | ~19 (matches baseline) | ✅ | Only as diagnostic if v3 fails |
| `submission_bf16_op_fusion/` | **medium-high** (needs GPU validation) | 30-37 (predicted) | ❌ NOT packed | Stretch goal after v3 confirms floor |
| `soar_gptqmodel_calib_w4a16_submission_20260522_1849_v24_pin_transformers.tar.gz` | tested-on-platform-now | unknown | ✅ | Already submitted as 1849 |

## llm-compressor exploration (zero submission cost)

If GPTQModel path is dead but you want one more shot at W4A16:

```bash
# On AutoDL (not local)
cd /root/autodl-tmp/zyn/sglang
git checkout parallel/non-gptqmodel-paths      # this branch
git pull origin parallel/non-gptqmodel-paths   # if pushed; otherwise sync manually
bash scripts/test_llmcompressor_artifact.sh --num-samples 150
```

This evaluates the 2026-05-14 artifact at `/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/`. The script:
1. Checks artifact exists + dumps metadata
2. Launches SGLang with **`--quantization compressed-tensors --dtype bfloat16`** (different from our GPTQModel path which uses `gptq_marlin` + `float16`)
3. Runs `eval_model.py` against perf_public_set
4. Prints decision guide based on acc

**If this artifact returns acc > 42 locally**, it's strong evidence that the GPTQModel pipeline itself is the problem and switching to llm-compressor is a viable rescue. Package it as a submission and platform-test.

## What's NOT done (by Claude in this session)

- ❌ `parameter.py:285` diagnostic message — proposed twice this session, never committed. If you go to Case 2, do this first.
- ❌ NVFP4 KV Cache (笔记 05) — large scope, not started. Future work.
- ❌ Op-fusion GPU validation — `submission_bf16_op_fusion/` prepared but the `MINICPM_FUSION_VERIFY=1` run is your responsibility.
- ❌ Bundling gptqmodel cp310 wheel into v24_pin_transformers — relevant if v24 install still fails on platform after fix2.

## Rank-20 goal realism check

Per `SUBMISSIONS.md` scoreboard, BF16 baseline = final_score 19.13.

| What you ship | Expected final_score | Rank target hit? |
|---|---|---|
| Nothing | 0 | ❌ |
| BF16 baseline reproduction (v4) | 19.13 | likely below rank 20 |
| **BF16 v3 (chunked-prefill 32K)** | **25-30** | **✅ likely hits rank 20** |
| BF16 + op-fusion (validated) | 30-37 | ✅✅ comfortable rank 18-20 |
| W4A16 cleared 80% gate (if ever) | ?? | bonus, unconstrained |

**Just shipping BF16 v3 is probably enough to hit your rank-20 goal.** Treat W4A16 as a high-variance bonus, not a requirement.

## When you commit Claude's work

The new files on this branch are all untracked. To preserve them:

```bash
git add experiments/BF16_SAFETYNET_AUDIT_20260522.md \
        experiments/HANDOFF_20260522_evening.md \
        scripts/test_llmcompressor_artifact.sh \
        submission_bf16_op_fusion/

git commit -m "parallel/non-gptqmodel: BF16 audit + op-fusion variant + llm-compressor exploration"
```

The branch is **not pushed**. If you want to keep working on it remotely:

```bash
git push -u origin parallel/non-gptqmodel-paths
```

If you decide the work is good and want it on `quant/w4a16` too:

```bash
git checkout quant/w4a16
git cherry-pick <commit-hash>
```

(Don't merge the whole branch — keep this branch as a "parallel paths" workspace.)

## Memory notes saved during this session

I saved 2 new memory entries (auto-loaded next session):

- `feedback_h5_transformers_runtime_drift.md` — H5 hypothesis (GPTQ uses transformers, RTN doesn't)
- `feedback_consult_past_failures_before_fix.md` — meta-rule: cite SUBMISSIONS.md/W4A16_README before any prepare_env diff

These are reference points for future Claude sessions, not new code.

---

## End of handoff. Good luck.
