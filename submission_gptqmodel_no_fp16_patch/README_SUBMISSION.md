# SOAR GPTQModel without fp16 sed-patch (v5j)

> Built 2026-05-23 while v5g (BF16 native) is on platform. The single-variable
> diagnostic to disambiguate **whether the fp16 sed-patch is the killer or
> whether `--dtype float16` itself is**.

## What this tests

The user pointed out (correctly): **official OpenBMB README + SOAR Week 4
champion both recommend GPTQ + Marlin + fp16 path on SALA**. Yet our 1849
GPTQ + Marlin + fp16 only got acc_ori=46.89 (way below acc gate).

The difference between official/champion and ours: our prepare_env applies
this sed-patch:

```bash
sed -i 's/torch.bfloat16/torch.float16/g' minicpm_backend.py
sed -i 's/torch.bfloat16/torch.float16/g' minicpm_sparse_utils.py
```

This hard-converts the sparse backend (including lightning attention
recurrent state h_t) from bf16 to fp16. fp16 has range ±65504; bf16 has
±3.4e38. Lightning recurrence h_t = g_t × h_{t-1} + ... can accumulate
large values → fp16 loses precision / overflows.

Official/champion's setup doesn't include this patch, presumably letting
SGLang's auto-cast at Linear boundaries handle the dtype mismatch (Marlin
GEMM uses fp16 internally, but the output gets cast back to bf16 for
sparse backend).

## Diff vs 1849

**Single change in prepare_env.sh**: skip the fp16 sed-patch block.

Everything else byte-identical to 1849:
- `quantize_gptqmodel_w4a16.py` (symlink to base)
- gptqmodel 7.0.0 install
- transformers 4.57.1 pin + hub<1.0 cascade
- flash_attn bundled wheel
- bundled SGLang
- `SGLANG_SERVER_ARGS = --quantization gptq_marlin --dtype float16 --chunked-prefill-size 8192 --dense-as-sparse ...`
- Same H1 hardened qzeros, H4 tokenizer overwrite, H5 transformers pin

## Why v5j > v5g for this hypothesis

`v5g` removes TWO things at once:
- `--dtype float16` → `--dtype bfloat16`
- fp16 sed-patch

If v5g acc jumps, we don't know if it's `--dtype` or sed-patch.

`v5j` removes ONLY the sed-patch (keeps `--dtype float16` + quantization).
If v5j acc jumps significantly, **sed-patch is the bug**, and the
official Marlin + fp16 recipe is fine.

## Submission decision matrix

| v5j vs 1849 (46.89) | Conclusion | Next |
|---|---|---|
| acc_ori jumps to 70+ | sed-patch was the bug; Marlin + fp16 is fine | Use this for all future GPTQ variants. Add chunked-prefill 32K next. |
| acc_ori 50-70 | sed-patch is partial bug; some fp16 leakage remains | Still bigger fix; iterate |
| acc_ori ≈ 47 (no change) | sed-patch isn't the bug; `--dtype float16` itself or quant or something else | Pivot to compressed-tensors loader (AWQ variant) |
| crash | sed-patch was actually necessary | Re-enable it; theory wrong |

## Tarball

```bash
bash scripts/full_preflight.sh --variant submission_gptqmodel_no_fp16_patch --pack
```

Reuses 1849's flash_attn wheel + sglang + perf_public_set via symlinks.
Expected size ~260 MB (full env bundle, same as 1849).

## When to submit

**Submit v5j IF**:
- v5g returns acc_ori ≥ 60 (suggesting fp16-related issue) AND
- You want to isolate whether sed-patch or dtype is the bug

**Submit v5j BEFORE v5g result IF**:
- You're confident the sed-patch is the prime suspect
- You want a more direct comparison to "official recipe"

**SKIP v5j IF**:
- v5g returns acc_ori ≈ 47 (rules out the fp16 family as bottleneck)
