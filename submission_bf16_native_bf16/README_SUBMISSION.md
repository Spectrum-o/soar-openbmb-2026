# SOAR BF16 Native Variant (v5g) — drop fp16 conversion

> Built 2026-05-23 08:11 after v5d's result revealed quantization wasn't the
> acc bottleneck. v5d (BF16 + fp16 sed-patch + `--dtype float16`) returned
> `acc_ori=46.64` — basically identical to 1849's W4A16 `acc_ori=46.89`.
> That falsifies "quant hurts acc" and points the suspicion at the
> **fp16 conversion** that BOTH 1849 and v5d apply.

## Hypothesis

Baseline (BF16, no extra args, **native bfloat16**) gets `final_score=19.13`
on the platform. To clear 19.13, baseline must have `acc ≥ correctness gate`
(commonly ≥80%). v5d's `acc=58.31` got `final_score=0` because it's below
the gate.

The ONLY material difference between v5d's pipeline and baseline's:
- v5d applies `sed s/torch.bfloat16/torch.float16/g` to `minicpm_backend.py`
  and `minicpm_sparse_utils.py`
- v5d uses `--dtype float16` (SGLang converts BF16 weights to fp16 at load)

→ **The fp16 conversion lossy-roundtrips activations through SALA's
recurrent lightning attention**, and that's where acc loses ~30-40pp vs
native bf16.

v5g tests this hypothesis with a single change.

## Diff vs v5d

| Component | v5d | **v5g** |
|---|---|---|
| Quantization | None (BF16 identity) | None (BF16 identity) |
| fp16 sed-patch on sparse backend | **YES** (inherited from 1849 prepare_env) | **❌ skipped** |
| `--dtype` flag | `--dtype float16` | **`--dtype bfloat16`** |
| `--quantization` flag | (none) | (none) |
| auto_map.AutoConfig strip | YES | YES |
| transformers 4.57.1 pin | YES | YES |
| bundled flash_attn + SGLang | YES | YES |
| chunked-prefill | 8192 | 8192 (unchanged for variable isolation) |
| --enable-mixed-chunk | NO | NO |

## Expected outcome

| Result | acc_ori | final_score | Conclusion |
|---|---|---|---|
| ✅ acc_ori jumps to ~80+ | clears gate | first non-zero final_score! | **fp16 conversion WAS the killer** |
| 🟡 acc_ori 50-70 | still under gate | 0 | fp16 partial blame; investigate more |
| ⛔ acc_ori ~47 (same as v5d) | 0 | fp16 isn't the bottleneck; model caps at ~47 on this eval |
| ⛔ crashes | — | — | bfloat16 path has its own incompatibility |

## Why this is THE highest-EV next submission

If hypothesis is right: jumps from 0 → non-zero final_score in ONE submission,
no quant change, no calibration tuning, just removing a workaround that was
specific to the gptq_marlin loader.

If hypothesis is wrong: we learn that the model itself caps at acc=47 on
this eval set, which forces us to either accept rank 20 via BF16+throughput
optimizations (no acc gate needed) OR find a completely different lever
(LoRA/QAT/AWQ).

## Tarball

Run:
```bash
bash scripts/full_preflight.sh --variant submission_bf16_native_bf16 --pack
```

Expected size ~252 MB (same env bundle as v5d).
