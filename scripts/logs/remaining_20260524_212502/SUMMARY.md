# GPU Remaining Smoke — 20260524_212502

Autonomous run of 3 variants left after chunk32k_safe + fp8kv (2026-05-24).

- Baseline: v5j_dtype_bf16 platform acc_ori = 82.18, final_score = 23.18
- chunk32k_safe local smoke (this session): acc_ori = 78.67
- chunk32k_fp8kv: CRASH (FlashAttention rejects fp8 K/V; Path D insufficient)

## Results

| Variant | Status | acc_ori | duration | Notes |
|---|---|---|---|---|
| gptqmodel_no_fp16_patch_dtype_bf16_chunk65k_safe | CRASH_other | 0.00 | 104s |  |
| gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_opfusion | CRASH_other | 82.33 | 875s |  |
| w4a16_lightning_skip_bf16 | CRASH_other | — | 7s | [prepare_model] FATAL: quantize_gptqmodel_w4a16.py exited 1 |

## Recommendations

No variant passed smoke cleanly. Submit `chunk32k_safe` (78.67) blindly.
