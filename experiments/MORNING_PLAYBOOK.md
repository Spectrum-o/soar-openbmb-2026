# Morning playbook — v23 platform result in hand, what to do

> Single page. Read top-to-bottom, execute commands as you go.
> If anything fails, read STATE_OF_PLAY.md or
> experiments/V23_PLATFORM_LOG_CHECKLIST.md.

## Step 1 — locate the v23 platform result

The SOAR competition platform web UI shows the eval status + final
score + a downloadable eval log. Save the log to persistent NAS:

```bash
# On the AutoDL server (or wherever you have the log)
cp /path/to/downloaded_log.txt /root/autodl-fs/zyn/logs/platform_v23.log
```

If the platform shows just the score JSON without a log, copy the
visible state-machine + score lines into the file manually. Even a
truncated log lets `decide_next_variant.py` extract acc_ori.

## Step 2 — sync the repo (fresh AutoDL: bootstrap first)

```bash
# Existing instance:
cd /root/soar/sglang
git pull origin quant/w4a16

# Fresh instance (only if /root/soar/sglang is missing):
BRANCH=quant/w4a16 \
SSH_KEY_PATH=/root/autodl-fs/zyn/soar_deploy_key \
bash /root/autodl-fs/zyn/bootstrap_gpu.sh install 2>&1 \
  | tee /root/autodl-fs/zyn/logs/install_$(date +%Y%m%d_%H%M%S).log
```

## Step 3 — one-line decision

```bash
cd /root/soar/sglang
bash scripts/v24_auto.sh
```

This will:
1. Read `/root/autodl-fs/zyn/logs/platform_v23.log` by default
2. Apply the V24_PLAN.md decision tree
3. Print the recommended next variant + rationale
4. Ask y/N to proceed
5. Run preflight + pack the recommended tarball
6. Print the tarball path

Use `--dry-run` first to see what it would do without changing anything.

## Step 4 — submit the produced tarball

Upload to the SOAR platform web UI. Platform 5h clock starts. Save
the platform's submission ID + start time somewhere (for diff
analysis later).

## Step 5 (parallel) — read STATE_OF_PLAY.md for context

`experiments/STATE_OF_PLAY.md` has the full picture: which
hypotheses are tested, which variant dirs are ready, what's deferred.

If something looks wrong (recommendation doesn't match expectation,
acc detection fails, etc.), open STATE_OF_PLAY.md and
V23_PLATFORM_LOG_CHECKLIST.md — they have the deeper diagnosis steps.

## Likely decision outcomes (and what to expect)

### Case A: v23 platform acc >= 30 (success scenario)

H1+H4 worked. v24_auto recommends `v24_perf` (chunked-prefill 65K
config). Submit, expect:
- acc_ori ≈ v23's (chunked-prefill is a perf flag, NOT quality)
- benchmark_duration drops materially (S1 50%-60% faster)
- final_score rises substantially

After v24 lands, push for the 80% ceiling: run
`v25_calib_multi_adaptive` and/or `v25_g64`. See V24_PLAN.md.

### Case B: v23 platform acc == 0 (the bad case)

v24_auto branches by qzeros-fix state:

- **`[qzeros-fix] FATAL`**: gptqmodel layout we didn't anticipate.
  v24_auto says MANUAL_FIX. Read the FATAL message + sample hex
  values in the log; extend `fix_qzeros_for_marlin` in
  `submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py`
  to handle the new pattern. Repack as v23b.

- **`[qzeros-fix] OK` + platform's transformers < 4.47**:
  H4 wasn't applied (the `chat_template.jinja` sidecar would have
  been silently ignored). v24_auto recommends `v24_pin_transformers`.
  This force-reinstalls transformers==4.57.1, eliminating sidecar
  drift as a variable.

- **`[qzeros-fix] OK` + platform's transformers >= 4.47**:
  H1+H4 weren't sufficient. v24_auto recommends `v24_no_dtype_key`
  (cheapest single-variable swap). Submit; if still 0 try
  `v24_bits8` to rule out bit-depth issues.

### Case C: v23 acc partial (e.g., 0 < acc < 30)

Quality partial; calibration recipe is the bottleneck. v24_auto
recommends `v25_calib_multi_adaptive`. Submit. Expect significant
improvement on niah/cwe tasks if calib quality was the gate.

### Case D: v23 pipeline crashed before eval

Read V23_PLATFORM_LOG_CHECKLIST.md Block 6/7. Common culprits:
- `RuntimeError: query and key must have the same dtype`:
  sed-patch missed a file. Check the `find` in prepare_env.sh's
  fp16 sed block.
- `KeyError: model.layers.X...`: gptqmodel's `dynamic` skip
  pattern doesn't match what it actually wrote. Update the regex
  in `make_quant_config`.
- `ImportError: flash_attn`: bundled wheel install failed.
  Check the wheel's cu128torch2.9-cp310 tag matches the platform.

## Useful commands at a glance

```bash
# Snapshot of running processes / partial eval if anything's still going
bash scripts/peek.sh

# Re-quant + eval to validate scripts end-to-end (no GPU it skips)
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --force-requant \
    --eval-data submission_gptqmodel_calib_w4a16/perf_public_set.jsonl \
    --num-samples 150

# Parse a platform log structurally + compare to local baseline
python3 tools/parse_quant_diagnostic.py \
    --input /root/autodl-fs/zyn/logs/platform_v23.log \
    --compare scripts/logs/local_eval_quant_submission_gptqmodel_calib_w4a16_1779289794.log \
    --output-md /root/autodl-fs/zyn/logs/diff_v23.md

# Show per-task pass rate corrected for the fwe free-credit
python3 tools/compute_corrected_acc.py \
    --input outputs/<latest>/predictions.jsonl

# Sanity check: all 88 tests + all 8 variant preflights
python3 -m unittest discover tests
for d in submission_gptqmodel_calib_w4a16*; do
    [ -d "$d" ] || continue
    python3 tools/pack_submission.py --variant "$d" --check-only \
        2>&1 | tail -1 | awk -v d="$d" '{print d ": " $0}'
done
```

## Submission ID tracking

After each submission, log the platform's submission ID + tarball
filename + timestamp in `SUBMISSIONS.md`. The "Chronological
submission log" table is the canonical record.

## If you're truly stuck

1. Read `experiments/STATE_OF_PLAY.md` for the bird's-eye view.
2. Read `experiments/V23_PLATFORM_LOG_CHECKLIST.md` for log block-by-block diagnosis.
3. Read `experiments/V24_PLAN.md` for branch-by-branch decisions.
4. Read `SUBMISSIONS.md` for historical context.

If those don't unblock, ask the offline Claude (the one that built
all this) by sharing the platform log + the v24_auto output. They'll
have full context from the commits.
