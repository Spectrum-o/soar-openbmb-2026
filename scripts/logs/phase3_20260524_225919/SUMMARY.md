# Phase 3 Smokes — 20260524_225919

After chunk32k_safe platform success (acc_ori=80.31, final_score=22.9).

## Results

| Experiment | Status | acc_ori | duration | Notes |
|---|---|---|---|---|
| E1_opfusion_real | OK | 82.67 | 813s | overlay applied;  |
| E2_fp8kv_path_e | CRASH_FA | — | 18s | Path E dequant fallback; FA still rejects (Path E insufficient) |
| E3_full_attn_bf16 | CRASH_quant | — | 25s | full-attn q/k/v/o quantized; gptqmodel crashed on missing self_attn in lightning layers |
