# SOAR W4A16 — Handoff (2026-05-20 night snapshot)

**Read this first if you are picking up the W4A16 work in a fresh session.**
The acc=0 root cause has been found and fixed. The current question is no
longer "why acc=0" but "**how far past acc=63 can we get to clear the
platform's `acc_ori >= 80` gate**".

---

## 1. Status snapshot

| Stage | acc on perf_public_set | Notes |
|---|---|---|
| v17 (full-attn GPTQ, gptqmodel 7.0) | **0.0** | platform run, 5h |
| v18 / v19 (MLP-only, gptqmodel 7.0) | **0.0** | platform run |
| RTN-scalefix (hand-written packing) | **42** | platform run; only non-zero W4A16 prior to qzeros fix |
| **v18 + qzeros in-place patch (2026-05-20 22:59 local)** | **63.33** on 30 of 150 | local_eval, ~13min wall |
| v18 + qzeros + calibration fix (running) | TBD | in flight, `--force-requant --num-samples 150` |

Platform correctness gate: `acc_ori >= 80` → `final_score = 0` below that.

---

## 2. Root cause (SOLVED)

`gptqmodel 7.0.0 + sym=True` writes uint4b8 `qzeros` as **7 per 4-bit slot**
(packed int32 `0x77777777 = 2004318071`). SGLang's Marlin dequant is
`weight = (q_unsigned - qzero) * scale`, which expects qzero = **8** for true
symmetric encoding (range `[-8, +7]`). With qzero=7 the range becomes
`[-7, +8]` — every weight gets a per-channel `+1 * scale` additive bias.
The bias accumulates through every Linear in every layer → garbage output
→ acc=0.

Hand-written RTN-scalefix writes qzero=8 explicitly → acc=42 (the only
non-zero W4A16 attempt prior to the qzeros fix).

**Fix landed in commit `324ea90d2`** as `fix_qzeros_for_marlin()` — a
post-`model.save()` step in both
`submission_gptqmodel_calib_w4a16/quantize_gptqmodel_w4a16.py` and
`submission_gptq_v17_minconfig/quantize_gptqmodel_w4a16.py`. It rewrites
every `.qzeros` tensor full of `0x77777777` to `0x88888888`. Idempotent.

Verification via `handoff/diagnose_qzeros.py` on a patched artifact:
```
qzeros  int32  (32, 2048)  unique[:6]=[-2004318072]   # = 0x88888888 ✓
```
The script's hint text shows the unsigned form (2290649224); int32 prints
the signed form (-2004318072). Same bit pattern, same correct value.

---

## 3. What is in flight (2026-05-20 ~23:00)

Re-running quantization from scratch with **two improvements** at once:

1. **Auto qzeros fix** (commit `324ea90d2`) — quant artifact will be correct
   without manual post-processing.
2. **Calibration fix** (commit `ff660de85`):
   - `tokenizer.truncation_side = "left"` → keep the TAIL of long
     perf_public_set rows (~30K-token median, question/needle/MCQ choices at
     the end) instead of 4K of haystack filler at the head.
   - Apply chat template if defined (SALA is instruction-tuned; serving wraps
     in `<用户>...<AI>`; raw `question` calibration sees a different boundary
     distribution than what the deployed model consumes).
   - `--max-calib-len` default 4096 → 8192.

Command currently in screen:
```bash
bash scripts/local_eval.sh \
    --variant submission_gptqmodel_calib_w4a16 \
    --force-requant \
    --num-samples 150 \
    2>&1 | tee /root/autodl-fs/zyn/logs/quant_v19_calib_$(date +%Y%m%d_%H%M%S).log
```

Expected wall time: ~14 min quant + ~1 h eval ≈ 1.5 h total.

---

## 4. Branch points after the in-flight run finishes

| 150-sample acc | Interpretation | Next move |
|---|---|---|
| ≥ 80 | Both fixes sufficient to clear platform gate | Package v20 tarball, submit |
| 70-80 | Major fixes working, residual gap | Add `o_gate`/`z_proj` quant back (P1 in v17_minconfig README), OR add attention quant back with fixed qzeros |
| 60-70 | Calibration fix had little impact vs 63.33 baseline | Multi-window slicing of long prompts (P2); revisit calibration source |
| < 60 | Calibration regressed or new bug | Diff predictions.jsonl vs the 63.33 run to find what changed |

---

## 5. AutoDL gotchas (we hit each of these today)

### Always after SSH reconnect / new screen — the 6-line preamble

```bash
cd /root/autodl-tmp/zyn/soar/sglang
source sglang_minicpm_sala_env/bin/activate    # prompt MUST show (sglang_minicpm_sala_env)
source /etc/network_turbo                       # AutoDL HTTPS accelerator
unset TMPDIR
export TMPDIR=/root/autodl-tmp/zyn/tmp && mkdir -p $TMPDIR
export TRITON_CACHE_DIR=/root/autodl-tmp/zyn/tmp/triton_cache && mkdir -p $TRITON_CACHE_DIR
export MAX_JOBS=4
```

User has added this to `~/.bashrc` so it runs on every login.

### Disk / mount specifics

| Path | Size | Notes |
|---|---|---|
| `/` overlay | 30 G | System disk. Build tmps WILL fill this if redirected here. |
| `/root/autodl-tmp` | 50 G ext4 | Repo + venv + build artifacts. Fast. |
| `/root/autodl-fs` | 14 T NAS | Models, logs, quant artifacts. ~50 MB/s read. |
| `/dev/shm` | 55 G RAM | **noexec** — runtime triton `dlopen` fails. Compile-time only. |

### Pin / compat conflicts already solved

- **gptqmodel 7.0.0 + transformers 4.57.1** needs two stubs (already in
  the quantize scripts as `_stub_transformers_for_gptqmodel_7`):
  1. `transformers.PreTrainedConfig = transformers.PretrainedConfig`
  2. `transformers.integrations.hub_kernels._gptqmodel_local_causal_conv1d_kernel = True`
     (trips gptqmodel's own early-return inside the causal_conv1d patch
     function, which otherwise crashes on transformers-4.x missing
     `lazy_load_kernel` / `_KERNEL_MODULE_MAPPING`).
- **`pip install gptqmodel` force-upgrades** `huggingface-hub` to 1.x,
  `transformers` to 5.x, `torchao` to 0.17. Always follow with:
  ```bash
  uv pip install --no-deps --force-reinstall \
      "transformers==4.57.1" "tokenizers>=0.22,<=0.23" "huggingface-hub==0.34.0"
  uv pip install --no-deps -e python/    # re-link patched local sglang source
  ```
- **sparse_kernel setup.py auto-detects CUDA but returns `(12, 0)`** on
  this AutoDL setup so sm_120 is never generated. Patch:
  ```bash
  sed -i 's/supported_archs = \["80"\]/supported_archs = ["80","90","120"]/' \
    3rdparty/sparse_kernel/setup.py
  ```
- **fp16 patch on sglang attention** is required for Marlin (GEMM emits
  fp16, sparse backend hardcodes bf16). After fresh install:
  ```bash
  bash scripts/fp16_patch.sh status     # check
  bash scripts/fp16_patch.sh apply      # if minicpm_backend.py is UPSTREAM
  ```

---

## 6. Commit map (recent → older on `quant/w4a16`)

```
324ea90d2  quant: fix gptqmodel 7.0 qzeros encoding bug — ROOT CAUSE
ff660de85  quant: backport calibration fix to MLP-only variant (v18)
cf4a7c058  quant: stub transformers 5.x symbols for gptqmodel 7.0 compat
2f659a23f  handoff: single-file context dump for W4A16 acc=0 investigation
e9acfa3fb  docs: update SUBMISSIONS.md
2a45c0177  quant: add v17-minconfig variant — isolate config-rewrite regression
871c08c62  quant: pivot submission_gptqmodel_calib_w4a16 to MLP-only (v18/v19)
70ad40d9a  scripts: local_eval auto-detect variant quant entrypoint
6cb0769fa  sglang: guard config/torchao paths against quant-tool side effects
```

## 7. Files / dirs that matter

| File | What |
|---|---|
| `handoff/README.md` (this file) | Single-file entry point |
| `handoff/diagnose_qzeros.py` | Standalone no-GPU safetensors inspector |
| `SUBMISSIONS.md` (repo root) | Chronological log of platform submissions |
| `submission_gptqmodel_calib_w4a16/` | Active MLP-only variant (v18 / v19 / next v20) |
| `submission_gptq_v17_minconfig/` | Sibling variant with full-attn + MLP module set (not submitted yet) |
| `submission_rtn_sym_w4a16/quantize_gptq_rtn_sym.py` | Reference acc=42 implementation; the "known good" qzeros layout |
| `scripts/local_eval.sh` | Quant → serve → eval pipeline (no platform slot) |
| `scripts/fp16_patch.sh` | bf16→fp16 sed-patch on sglang sparse backend |
| `/root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized/` | Current quantized artifact on AutoDL (~5.4 GB) |
| `/root/autodl-fs/zyn/soar_toolkit/{eval_model.py,perf_public_set.jsonl}` | SOAR eval harness + 150-row eval set |

## 8. Known follow-ups (do NOT do mid-run)

- Fix `scripts/local_eval.sh` parser to extract `Average Score: XX%` from
  current eval_model.py output (it currently prints `acc_ori=?` because
  it greps for an older format).
- If acc still < 80 after calibration fix: try restoring `o_gate` /
  `z_proj` quantization (P1 in v17_minconfig README — needs
  `layer_modules_strict=False` or a post-pass RTN).
- The May 14 llmcompressor artifact at
  `/root/autodl-fs/zyn/models/MiniCPM-SALA-W4A16/` (compressed-tensors
  format, actorder=static) is an unrelated alternative path that was
  never benchmarked. Could be a fallback if GPTQ-Marlin stalls below 80.

## 9. User context (for non-human readers)

User (zyn) is competing solo in SOAR 2026 (OpenBMB MiniCPM-SALA inference
optimization, RTX PRO 6000 96GB). Shares the AutoDL instance with labmates
but their quant artifacts at `/root/autodl-fs/models/MiniCPM-SALA-GPTQ-W4A16*`
should NOT be reused (independent work). Prefers concise / direct
technical answers, terse follow-ups, in Chinese. Writes throwaway commands
into `test/ddd.txt` for clipboard-friendly handoff.

Champion route (per "SOAR 周冠军笔记 04") is W4A16 GPTQ + Marlin with
calibration matched to eval task distribution. After the qzeros fix and
calibration fix we are now ON that route — the remaining 80-gate gap is
expected to be calibration tuning + possibly restoring attention quant.
