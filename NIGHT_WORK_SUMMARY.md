# Night work summary — 2026-05-21

Built while you slept (~3 hours of CPU-only infra + debugging).
**Everything pushed to `quant/w4a16`** — `git pull` on the server to get
all of it.

---

## TL;DR — what you need to know first

1. **v21 acc=49 真因不是 calibration choice — 是 quant 导致 repetition collapse**。
   `tools/analyze_predictions.py` 揭示失败样本全部撞 `output_tokens=65546`
   max-cap，model 在 token 级 loop（"stall", "stall"... 或 "sky is blue..."
   重复）。chat-template/truncation 不是杠杆。
2. **fp16 sed-patch hypothesis 已排除**。v21 tarball 的 sed pattern 真的能
   match 7+3 lines (`tools/quant_config_validator.py`)。
3. **平台 0% 跟 local 49% 的 49pp 差距，主要由 hidden eval set 难度更高解释**。
   Local 的 fwe 30 题 100% 是 free credit（任务太简单），剔除后 local 真实
   ~36%。Platform 的 hidden set 应该没有这种 free credit。
4. **修了一个 production bug** —— multi-adaptive 切窗模式下，random window
   可能跟 centered mid window 重叠。unit test 抓到 (seed=7, n=13L)。两个
   variant 的 quantize 脚本 + `tools/calib_set_preview.py` 都已修。
5. **Exp B (chat tpl OFF) 在 server 上还没完** —— 卡在 147/150（3 个长样本
   被 repetition collapse 卡了 30+ 分钟/条）。watchdog 5min snapshot 中。

**第一件事 → 读 `experiments/PLAN.md`**。是醒来后的 runbook。

---

## 新增文件清单（已 commit + push）

### Tools — diagnostics (work on real artifacts)

| Path | 作用 | 已验证 |
|---|---|---|
| `tools/analyze_predictions.py` | 单 predictions.jsonl per-task 错误模式（找到了 repetition collapse） | ✓ on v21 + v18 outputs |
| `tools/diff_predictions.py` | 双 predictions.jsonl transition matrix | ✓ on v18 (30 sample) vs v21 (150 sample) |
| `tools/inspect_quant_artifact.py` | safetensors meta + qzeros 0x88888888 check | ✓ on mock GOOD + BUG artifacts |
| `tools/quant_config_validator.py` | tarball pre-flight (qzeros / sed / SGLANG_SERVER_ARGS) | ✓ on v21 tarball |

### Tools — experiment preparation

| Path | 作用 | 已验证 |
|---|---|---|
| `tools/build_calib_set.py` | jsonl 按 token length 分桶生成 short/medium/long/super/mixed/holdout calib sets | ✓ on perf_public_set with mock tokenizer |
| `tools/calib_set_preview.py` | 不跑 GPTQ 可视化 windowing 后实际样本 | ✓ syntax + mock data |
| `tools/pack_submission.py` | stage → cp -L → drop pycache → tar → verify → md5 | ✓ packs identical structure to v21/v22 |
| `tools/parse_experiment_logs.py` | logs dir → results.csv + summary.md | ✓ on scripts/logs/ (17 files) |

### Scripts

| Path | 作用 |
|---|---|
| `scripts/list_all_quant_artifacts.sh` | server inventory of `/root/autodl-fs/zyn/models/*-quantized/` |

### Docs

| Path | 内容 |
|---|---|
| `experiments/PLAN.md` | **tomorrow's runbook** — Exp C/D/E/F/G priority + decision tree |
| `docs/calibration_gotchas.md` | 沉淀知识：qzeros bug / truncation-side / task-length 相关 / fwe 误导 / repetition collapse / fp16 sed |
| `docs/awq_fallback_plan.md` | 如果 GPTQ 死活过不了 80，3 步 AWQ 验证（含 May 14 artifact 复用） |

### Tests

| Path | 覆盖 |
|---|---|
| `tests/test_calibration_tools.py` | 27 tests passing: slice_windows_for_prompt (6) + build_calib_set helpers (5) + parse_experiment_logs regex (9) + analyze_predictions classification (2) + quant_config_validator (4) — **caught a real production bug on first run** |

### Bug fixes baked into existing files

| File | Fix |
|---|---|
| `submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py` | `_slice_windows_for_prompt`: random window in multi-adaptive mode no longer overlaps centered mid window |
| `submission_gptq_v17_minconfig/quantize_gptqmodel_w4a16.py` | same fix |
| `tools/calib_set_preview.py` | same fix |

---

## Verified locally (CPU only) vs needs server verification

### ✓ Verified locally
- All 27 unit tests in `tests/test_calibration_tools.py`
- `analyze_predictions.py` ran cleanly on v21 + v18 outputs
- `diff_predictions.py` ran cleanly on v18 vs v21
- `parse_experiment_logs.py` ran cleanly on 17 real logs in `scripts/logs/`
- `inspect_quant_artifact.py` ran cleanly on mock safetensors (one with
  GOOD qzeros, one with BUG qzeros — both correctly classified)
- `quant_config_validator.py` ran cleanly on v21 tarball (passed
  all critical checks, flagged 4 informational missing items)
- `build_calib_set.py` ran cleanly on real perf_public_set with mock tokenizer
- `calib_set_preview.py` ran cleanly on real perf_public_set with mock tokenizer
- `pack_submission.py --check-only` on 3 variant dirs, full pack on calib_w4a16

### ⏳ Needs server verification before relying on
- `build_calib_set.py` with real `MiniCPM-SALA` tokenizer (mock counts ≠
  real BPE counts; the bucket distribution will shift)
- `calib_set_preview.py` with real tokenizer (decoded head/tail strings
  are mock-flavored locally)
- `inspect_quant_artifact.py` against the actual v21 quant artifact at
  `/root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized/`
- `scripts/list_all_quant_artifacts.sh` against the real models dir
- `quant_config_validator.py` against v22 tarball

These are all bare `python3 tools/X.py --help`-runnable on the server.

---

## Decisions baked in (look here if anything seems wrong)

- **fwe is real and excluded from PLAN.md's "real perf" estimate** —
  the metric the platform uses INCLUDES fwe. Don't try to remove fwe
  from eval; only mentally adjust expectations.
- **Tests use unittest, not pytest** — keeps zero external deps. Run:
  `python3 -m unittest tests.test_calibration_tools -v`
- **mock tokenizer is intentional** — local dev shouldn't require an
  HF model download. Real tokenization happens on server.
- **No HTML report tool** — markdown + CSV are enough; less to debug.
- **No GPTQ sweep yaml runner** — our experiments are sequential
  hypothesis tests, not parameter grids. Re-evaluate if/when we have
  3+ dimensions to scan.

---

## What I deliberately did NOT do

| Skipped | Why |
|---|---|
| Re-pack v23 tarball with multi-adaptive code | No platform slot to spend until we know acc beats 80% locally |
| Implement AWQ fully | Wrote `docs/awq_fallback_plan.md` instead; full impl is 3+ hours and needs validation against the actual May 14 artifact |
| Edit eval_model.py for repetition_penalty (Exp D) | It's in the SOAR-Toolkit tree, not ours; documented as a 1-line sed in PLAN.md |
| Fused-op correctness tests | Orthogonal to current acc problem; perf optimization belongs after 80% gate is cleared |
| GPTQ sweep yaml runner | Cartesian sweep wrong fit for "narrow hypothesis test" cadence |

---

## Quick commands for tomorrow morning

```bash
# Sync
cd /root/autodl-tmp/zyn/soar/sglang
git pull   # all watchdog snapshots + my work

# Did Exp B finish?
ls -d outputs/*/
wc -l outputs/*/predictions.jsonl

# If 20260521_011939 has 150 lines:
python3 tools/analyze_predictions.py \
    --input outputs/20260521_011939/predictions.jsonl \
    --top-failures 3

# Full A/B between v21 and Exp B
python3 tools/diff_predictions.py \
    --baseline outputs/20260520_234226/predictions.jsonl \
    --candidate outputs/20260521_011939/predictions.jsonl \
    --output-md /root/autodl-fs/zyn/logs/diff_v21_vs_expB.md

# Sanity-check the v21 quant artifact on disk
python3 tools/inspect_quant_artifact.py \
    --artifact /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized

# Inventory of every quant artifact on /root/autodl-fs/
bash scripts/list_all_quant_artifacts.sh

# Aggregate every log
python3 tools/parse_experiment_logs.py \
    --log-dir scripts/logs \
    --log-dir /root/autodl-fs/zyn/logs \
    --output-csv /tmp/all_results.csv \
    --output-md /tmp/all_summary.md
cat /tmp/all_summary.md
```

Then jump to `experiments/PLAN.md` for the next experiment to run.

---

## File diff stats

```
14 files added (8 tools + 1 shell + 3 docs + 1 test + 1 summary)
3 production files patched (slice_windows overlap fix)
0 files deleted
```

Watchdog (pid 25036 on server) is still running 5min snapshots; if it
died, restart with the command in `scripts/watchdog_commit.sh`.
