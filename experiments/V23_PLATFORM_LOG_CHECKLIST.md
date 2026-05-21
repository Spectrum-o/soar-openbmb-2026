# v23 platform-log checklist

> When v23 (or any subsequent GPTQModel submission) returns from the
> SOAR platform, run this checklist top-to-bottom to attribute the
> result to a hypothesis. Each block contains: a `grep` you can paste,
> the expected output(s), and the decision branch.

The log lives wherever the platform UI lets you download it (typically
the eval log + the prepare_env/prepare_model stdout pipe). Save it as
`/root/autodl-fs/zyn/logs/platform_v23.log` and pass that path below.

## TL;DR

```bash
LOG=/root/autodl-fs/zyn/logs/platform_v23.log
BASE=scripts/logs/local_eval_quant_submission_gptqmodel_calib_w4a16_1779289794.log

# One-shot: structured diff against the known-good local baseline
python3 tools/parse_quant_diagnostic.py \
    --input "$LOG" --compare "$BASE" \
    --output-md /root/autodl-fs/zyn/logs/diff_v23_vs_local.md
cat /root/autodl-fs/zyn/logs/diff_v23_vs_local.md
```

The diff report's "Verdict heuristic" section names the most likely
failure mode. If that's not enough, run the grep blocks below.

---

## Block 1 — Environment fingerprint (resolves H2 + H3)

```bash
grep -E "^\[versions\]|^\[prepare_env\] (gptqmodel|forcing|installing)" "$LOG"
```

Expected on a healthy v23:

```
[prepare_env] forcing gptqmodel==7.0.0 (currently: <some-version>)
[versions] python=3.10.<x>
[versions] torch=2.9.1
[versions] transformers=<X>.<Y>.<Z>
[versions] gptqmodel=7.0.0
[versions] flash-attn=2.8.3
```

| Observation | Conclusion |
|---|---|
| `python=3.12.<x>` (matches AutoDL) | H3 (python ABI mismatch) doesn't apply on this platform run |
| `python=3.10.<x>` | platform IS on 3.10 as suspected; H3 stays as a contributing factor |
| `transformers<4.47` | **H4 confirmed environmentally** — sidecar will not be read; inline-only must be correct |
| `transformers>=4.47` | platform reads sidecar like AutoDL; symptom direction needs re-thinking |
| `gptqmodel=7.0.0` after forcing line | pin took effect ✓ |
| `gptqmodel=7.0.<x>` where x != 0 AND no forcing line | pin was skipped (`python_pkg_exact_version` succeeded but for the wrong reason); patch `prepare_env.sh` |
| no `[versions]` block at all | print_versions never ran; prepare_env crashed earlier — search the log above for `FATAL` |

---

## Block 2 — Calibration data resolution

```bash
grep -E "^\[prepare_model\] DIAGNOSTIC: calib jsonl" "$LOG"
```

Expected:

```
[prepare_model] DIAGNOSTIC: calib jsonl path=/tmp/<somewhere>/perf_public_set.jsonl
[prepare_model] DIAGNOSTIC: calib jsonl size=23925601 bytes
[prepare_model] DIAGNOSTIC: calib jsonl row count=150
```

| Observation | Conclusion |
|---|---|
| `row count=150` | bundled calib was used ✓ |
| `row count=52` (the corrupt soar_toolkit copy) | calib data is broken — quality bad; not the platform=0 cause but degrades Hessian |
| `path=...autodl-fs/...soar_toolkit...` | platform somehow has the AutoDL path — unlikely; check |
| no DIAGNOSTIC lines | prepare_model.sh DIAGNOSTIC block not committed in this tarball — re-pack |

---

## Block 3 — Output dir layout after quantize

```bash
grep -E "^\[qzeros-fix\] +\S+\.(safetensors|json|model)" "$LOG"
grep -E "^\[qzeros-fix\] scanning" "$LOG"
```

Expected on a healthy 3-shard layout (matches AutoDL):

```
[qzeros-fix]   model-00001-of-00003.safetensors  (4,983,672,832 bytes)
[qzeros-fix]   model-00002-of-00003.safetensors  (3,213,440,128 bytes)
[qzeros-fix]   model-00003-of-00003.safetensors  (1,124,560,896 bytes)
[qzeros-fix] scanning 3 .safetensors weight file(s)
```

| Observation | Conclusion |
|---|---|
| 3 shards same names + close to AutoDL sizes | platform gptqmodel writes same layout — H1's "different glob" branch refuted |
| `model.safetensors` single file | confirms H1 first branch was real on old code; hardened fix should still catch it via `*.safetensors` glob |
| `pytorch_model-*.bin` only | gptqmodel switched to torch format — fix will raise `FATAL: found N torch .bin files` |
| no scanning line | reached fix function but exited before scanning; look for FATAL |

---

## Block 4 — qzeros sample inspection (resolves H1)

```bash
grep -E "^\[qzeros-fix\] +sample:" "$LOG"
```

Expected on a healthy v23 BEFORE the patch step:

```
[qzeros-fix]   sample: model-00002-of-00003.safetensors::layers.0.mlp.gate_proj.qzeros dtype=torch.int32 unique[:6]=[2004318071] hex=['0x77777777']
[qzeros-fix]   sample: model-00002-of-00003.safetensors::layers.0.mlp.up_proj.qzeros dtype=torch.int32 unique[:6]=[2004318071] hex=['0x77777777']
...
```

| Observation | Conclusion |
|---|---|
| `hex=['0x77777777']` uniform | classical bug; fix WILL patch → expect 96/96 patched below |
| `hex=['0x88888888']` already | gptqmodel on platform fixed itself; no patching needed; H1 wasn't it |
| `dtype=torch.uint32` | uint32 variant; hardened fix handles via `.view(torch.int32)` |
| `unique[:6]=[..., ...]` (non-uniform) | mixed qzeros — fix does per-element replacement; should still work |
| any other `dtype=...` | unknown encoding; fix will exit with `WARN ... unexpected dtype` and probably raise after; report it to extend the function |

---

## Block 5 — qzeros fix SUMMARY (the main verdict)

```bash
awk '/^\[qzeros-fix\] SUMMARY:/,/^\[qzeros-fix\] (OK|FATAL)/' "$LOG"
```

Healthy:

```
[qzeros-fix] SUMMARY:
[qzeros-fix]   qzeros tensors seen:        96
[qzeros-fix]   patched (had 0x77777777):   96
[qzeros-fix]   already good (0x88888888):  0
[qzeros-fix]   unknown dtype skipped:      0
[qzeros-fix]   unknown values not patched: 0
[qzeros-fix] POST-CHECK model-00002-of-00003.safetensors::layers.0.mlp.gate_proj.qzeros first_val=-2004318072 (0x88888888)
[qzeros-fix] OK
```

| Observation | Conclusion |
|---|---|
| `patched=96` + `POST-CHECK ... (0x88888888)` + `OK` | H1 fix worked. If platform still acc=0, H1 wasn't the cause; look elsewhere (H4 / H5 / etc.) |
| `patched=0`, `already good=96`, `OK` | qzeros were already correct from gptqmodel itself; H1 wasn't relevant on this platform. acc=0 means H4 or H5 |
| `seen=0` | **FATAL exit fired**: no .qzeros tensors detected; gptqmodel wrote a layout we don't recognize at all. Read the inventory in Block 3 |
| `unknown values not patched > 0` AND `patched=0` | **FATAL exit fired**: qzeros encoding isn't `0x77777777` nor `0x88888888`. Read the `sample:` lines in Block 4 and extend the fix |
| `POST-CHECK ... (0x77777777)` after non-zero patched | save path mismatch; in-memory patch never reached disk. **FATAL**: check the save_file path target |

---

## Block 6 — sed-patch verification

```bash
grep -E "^\[prepare_env\] patched (minicpm|sgl)" "$LOG"
```

Expected:

```
[prepare_env] patched minicpm_backend.py for fp16 quantized run
[prepare_env] patched minicpm_sparse_utils.py for fp16 quantized run
```

| Observation | Conclusion |
|---|---|
| both files patched | sed-patch worked ✓ |
| only one patched | the other file path is wrong in prepare_env.sh; quant will crash with `query and key must have the same dtype` later |
| neither patched | bundled SGLang has different file names — diff against current branch |

---

## Block 7 — SGLang server startup

```bash
grep -E "Server is listening|Initialization completed|RuntimeError|ImportError|AssertionError" "$LOG"
```

| Observation | Conclusion |
|---|---|
| `Server is listening` | server started; acc=0 then is purely model output quality |
| `RuntimeError: query and key must have the same dtype` | sed-patch missed at least one file; cross-check Block 6 |
| `AssertionError: Only flash_attention_2 is supported` | force_flash_attention_2 patch didn't apply; check `quantize_gptqmodel_w4a16.py` line touching `_attn_implementation` |
| `ImportError: ... flash_attn ...` | flash-attn wheel wasn't installed; check `prepare_env.sh` flash-attn block |
| `KeyError: model.layers.X.self_attn.Y.weight` | quant config `dynamic` skip rules don't match what gptqmodel actually emitted; cross-check Block 3 inventory |

---

## Block 8 — Bench timing (sanity)

```bash
grep -E "S1=|S8=|Smax=|benchmark_duration|final_score" "$LOG"
```

Reference points (from prior submissions):

```
v17/v21/v22:  S1=626  S8=998  Smax=2290     (broken-but-serving)
RTN-scalefix: S1≈625  S8≈997  Smax≈2289     (very close — same kernel path)
baseline BF16: S1=N/A   (different model, different timing)
```

| Observation | Conclusion |
|---|---|
| timings ≈ v17/v21/v22 | model is generating tokens at the same speed; if acc=0 with these timings then content is the problem, not the kernel |
| timings way off | sglang setup differs materially (chunked-prefill or KV cache config); cross-check Block 7 |

---

## Decision tree (combining all blocks)

```
qzeros SUMMARY shows OK + POST-CHECK 0x88888888
│
├── platform acc ≥ 30        → H1 or H4 was real; ship v24 / push harder on calibration
│                              (R5: still need Exp E/F/G to reach 80 ceiling)
│
├── platform acc ≈ 0          → H1+H4 both irrelevant on this platform run
│   │
│   ├── transformers ≥ 4.47 in Block 1
│   │     → H4 mechanism never applied; look at H5/H6 (next):
│   │       - is `dtype: "float16"` config key being mis-parsed?
│   │       - is `auto_map` mis-resolving the custom modeling code?
│   │
│   └── transformers < 4.47 in Block 1
│         → H4 SHOULD have applied. Check that
│           submission_*/quantize_gptqmodel_w4a16.py is actually the
│           hardened version (grep "CRITICAL 2026-05-22" in tarball).
│           If yes, the inline chat_template in BASE's
│           tokenizer_config.json may itself be wrong/empty — fetch
│           and read it.
│
qzeros SUMMARY raised FATAL
│
└── read the FATAL message + Block 4 sample hex values
      → extend fix_qzeros_for_marlin() to handle the new layout
      → repack as v24
```

---

## After matching: what to do next

| Outcome | Action |
|---|---|
| acc=0 still | gather sample of v23 output text from the eval log (`grep -A 3 "task=" "$LOG"` or whatever the harness uses); compare against v17/v21/v22 outputs in `outputs/20260520_234226/predictions.jsonl` — if outputs are byte-identical, the FIX never took; if outputs DIFFER but are still garbage, a new failure mode is in play |
| 30 ≤ acc < 80 | Read `tools/analyze_predictions.py --output-md` on the eval predictions to see per-task movement. Use that to plan Exp E (multi-adaptive calib), F (alternative group-size / desc-act), G (AWQ) per `experiments/PLAN.md` |
| acc ≥ 80 | ship v24 (full-attn) for parallel-track perf gain; add chunked-prefill 65536 back to submission tarball SGLANG_SERVER_ARGS |
| Pipeline crashed before serving | the FATAL line names the file & function; usually fixable in <1h |

---

## Common pitfalls reading this log

1. The log on the platform side may have ANSI color escapes or progress-bar redraws. Pipe through `sed 's/\x1b\[[0-9;]*[a-zA-Z]//g'` if greps look weird.
2. `gptqmodel` itself logs a lot. Our `[qzeros-fix]` and `[prepare_model] DIAGNOSTIC:` prefixes are anchored at line start to disambiguate.
3. `parse_quant_diagnostic.py` is backward-compatible with the legacy `[qzeros-fix] total N qzeros ...` format. If the SUMMARY block is missing but the legacy line is present, the parser will still report `patched=N, ended_ok=True`. So an old-format log with `patched=96` is still good news.
4. The `[versions]` block is printed BEFORE quantize runs, so if quant crashed midway, you still have the env fingerprint.
