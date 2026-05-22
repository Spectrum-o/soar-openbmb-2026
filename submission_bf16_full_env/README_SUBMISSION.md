# SOAR BF16 Full-Env Submission (v5)

> **Why this exists**: BF16 v2 + v3 both crashed in ~13s on the platform with
> the transformers model_type list dump. The 2026-05-20 hypothesis
> ("bundled SGLang causes v2 crash, so v3 doesn't bundle") was FALSIFIED:
> v3 (no bundle) ALSO crashed with the same error.
>
> 1849 (W4A16, full env bundle) was the only submission to load SALA
> successfully on the platform's current env (transformers 5.9.0 base).
> v5 = "1849 env, minus quantization" = BF16 with the full env scaffolding
> that's verified to work.

## Diff vs other BF16 variants

| | safetynet v3 | baseline-match v4 | **v5 full_env** | 1849 W4A16 |
|---|---|---|---|---|
| Bundled SGLang | ❌ | ❌ | **✅** | ✅ |
| transformers pin to 4.57.1 | ❌ | ❌ | **✅** | ✅ |
| Bundled flash_attn wheel | ❌ | ❌ | **✅** | ✅ |
| Quantization | none (BF16) | none (BF16) | **none (BF16)** | gptq_marlin |
| chunked-prefill-size | 32768 | 8192 | **32768** | 8192 |
| --enable-mixed-chunk | ✅ | ❌ | **✅** | ❌ |
| platform result | 11s crash | (untested) | **expected ~25-30 final_score** | acc=58.61, final_score=0 |

## Expected outcome

- correctness == baseline (no model change, math identical to BF16 baseline 19.13)
- throughput boost from chunked-prefill 32K + mixed-chunk: +59% local bench
- expected final_score: **25-30** (BF16 19.13 × ~1.5x throughput effect)

## SGLANG_SERVER_ARGS

```
--disable-radix-cache
--attention-backend minicpm_flashinfer
--chunked-prefill-size 32768
--max-prefill-tokens 32768
--enable-mixed-chunk
--skip-server-warmup
--dense-as-sparse
```

NO `--quantization`, NO `--dtype` — native BF16 path.

## Why all four env pieces are necessary

1. **Bundled SGLang** — our `python/sglang/srt/models/minicpm.py` has
   MiniCPMSALAForCausalLM. Platform's base env SGLang may lack this class
   (theory: v3 model_type list dump suggests transformers registers
   `minicpm_sala` but SGLang can't dispatch to a model implementation).

2. **transformers 4.57.1 pin** — verified necessary by 1849. transformers
   5.9.0 (platform default) has known incompatibilities with SALA's
   modeling code (RoPE param warnings). 4.57.1 is the repo's tested pin
   per W4A16_README.

3. **Bundled flash_attn 2.8.3+cu128torch2.9-cp310** — SALA's modeling code
   asserts `_attn_implementation == "flash_attention_2"`. transformers
   >= 5.0 does an eager import of flash_attn at PreTrainedModel.__init__.
   Bundling avoids platform-network download.

4. **chunked-prefill 32K + mixed-chunk** — throughput optimization that
   works ONLY because we bundle SGLang (our minicpm_backend.py supports
   mixed forward mode at line 377+).

## Risks

- ~5%: platform's bundled-SGLang install path differs from 1849's →
  fallback to chunked-prefill 8192 in v5b.
- ~5%: BF16 dtype implicit handling fails (we don't set --dtype) →
  try --dtype bfloat16 in v5b.
- ~5%: unknown env drift since 1849's 2026-05-22 success →
  re-run smoke test in prepare_env (already included).

## Pack command

```bash
python3 tools/pack_submission.py --variant submission_bf16_full_env --check-only
python3 tools/pack_submission.py --variant submission_bf16_full_env \\
    --output soar_bf16_full_env_$(date +%Y%m%d_%H%M).tar.gz
```

Tarball size: ~250 MB (mostly bundled flash_attn wheel).
