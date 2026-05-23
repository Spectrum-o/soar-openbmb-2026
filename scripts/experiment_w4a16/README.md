# W4A16 迁移实验 — 今晚自动化包

> 2026-05-23 11:30 准备完成。今晚 GPU 一到位就能 3 步跑完。

## 背景

v5g (BF16 native, chunked-prefill 8192) 在 SOAR 平台拿到 `acc_ori=83.04`，
final_score=15.76（首次非零）。这证明：
- bf16 family 是对的，fp16 sed-patch 是过去 acc 卡在 ~47 的根因
- v5g 的"地基" fix (auto_map 剥除、transformers 4.57.1 pin、tokenizer 覆盖等) 全部要保留

**官方推荐的 W4A16 完整配方** (按 `W4A16_README.md` + `run_sala_w4a16.sh`)：
1. llm-compressor + GPTQ → compressed-tensors 格式 (NOT gptq_marlin)
2. SGLang 用 `--quantization compressed-tensors`
3. Serving args: `--chunked-prefill-size 65536 --max-prefill-tokens 65536 --max-running-requests 32 --mem-fraction-static 0.80`
4. 不显式设置 `--dtype` (默认 bfloat16)

我们一直在错的 family 跑 (gptqmodel + gptq_marlin + fp16 sed-patch)。
今晚把 v5g 的 bf16 native 成功路径 + 官方 serving args + 量化做完整迁移。

## 今晚的 3 步流程

### Step 1：30 秒环境 sanity (no GPU)

```bash
cd /root/autodl-tmp/zyn/sglang
git pull origin quant/w4a16
source sglang_minicpm_sala_env/bin/activate
bash scripts/experiment_w4a16/smoke.sh
```

期望全 PASS。任何 `✗` 都要先解决。常见问题：
- transformers 未安装 → `bash submission_bf16_official_args/prepare_env.sh` 跑一遍补齐
- BF16 模型路径不在 → 检查 `/root/autodl-fs/models/OpenBMB/MiniCPM-SALA`
- sed-patch 泄漏 (`torch.bfloat16` 不在 minicpm_backend.py) → 之前 gptq_marlin 跑过没清；
  `git restore python/sglang/srt/layers/attention/minicpm_backend.py python/sglang/srt/layers/attention/minicpm_sparse_utils.py`

### Step 2：15 分钟 pipeline smoke (有 GPU, 验证全链路)

```bash
bash scripts/experiment_w4a16/run_plan.sh --smoke
```

- Phase 1: BF16 + 官方 args，5 samples eval (~3 min)
- Phase 2: AWQ quant w/ 8 calib samples + 5 samples eval (~10-12 min)
- 末尾打印 decision，应该是 "completed" 而不是 "FAILED"
- acc 不重要 (5 samples 噪声大)，**只看 pipeline 是否走通**

如果 Phase 1 或 Phase 2 报 FAILED：读 `scripts/logs/w4a16_<session>/master.log` 看具体 error。

### Step 3：3 小时自动化全跑 (无人值守)

```bash
nohup bash scripts/experiment_w4a16/run_plan.sh \
    > /tmp/w4a16_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "started pid=$! at $(date)" > /tmp/w4a16_pid
disown
```

- Phase 1: BF16 + 官方 args，200 samples，~15-20 min，期望 acc_ori ≈ 83
- Phase 2: AWQ quant (256 calib, 8192 max_len) + 200 samples eval，~1.5-2h
- 每个 phase 结束自动 commit + push 到 `quant/w4a16`
- 全部跑完会有 `scripts/experiment_w4a16/results/<session>.txt` 写出 DECISION

监控（任何时候可以 grep）：

```bash
# 看 master log
tail -f scripts/logs/w4a16_*/master.log

# 看进程
ps -ef | grep run_plan.sh | grep -v grep
nvidia-smi
```

提前杀掉：

```bash
pkill -f run_plan.sh
# 杀 sglang 服务（如果 phase 2 正在 eval）
bash scripts/killall_sglang.sh
```

## 决策矩阵

`run_plan.sh` 结束后自动写到 results 文件，按这个表：

| Phase 1 acc | Phase 2 acc | 提交 |
|---|---|---|
| ≈ 83 | **≥ 78** | `submission_awq_official_args` (W4A16 + 官方，主目标) |
| ≈ 83 | 70-77 | `submission_bf16_official_args` (BF16 + 官方，安全) |
| ≈ 83 | < 70 | `submission_bf16_official_args` (W4A16 量化损失太大，BF16 兜底) |
| < 78 | — | `submission_bf16_native_chunk32k` (v5h 兜底；官方 args 在平台 break acc 了) |

历史本地↔平台落差 ~2-3 pp，所以本地 78 ≈ 平台 75+。

## 提交打包

确认 decision 后：

```bash
# 自动 preflight + pack
bash scripts/full_preflight.sh \
    --variant <picked_variant> --pack

# 输出 .tar.gz + .md5 在 release/ 或 cwd
```

或者直接用 `tools/pack_submission.py` 手动打包。

## 文件清单

新增文件：

- `submission_bf16_official_args/` — v5g + 官方 serving args 控制变体
  - `prepare_env.sh` (new) — SGLANG_SERVER_ARGS 改成 65K/0.80
  - `prepare_model.sh` (copy from v5g) — auto_map.AutoConfig strip
  - `README_SUBMISSION.md`
  - sglang/, flash_attn 通过 symlink 共享 v5g 的资产
- `submission_awq_official_args/` — AWQ + 官方 serving args W4A16 主变体
  - `prepare_env.sh` (new) — SGLANG_SERVER_ARGS 改成 65K/0.80 + compressed-tensors
  - `prepare_model.sh` → symlink `submission_awq_llmcompressor/prepare_model.sh`
  - `quantize_llmcompressor_awq.py` → symlink
  - `perf_public_set.jsonl` → symlink
  - `README_SUBMISSION.md`
  - sglang/, flash_attn → symlink
- `scripts/experiment_w4a16/run_plan.sh` — 自动化 runner (2-phase)
- `scripts/experiment_w4a16/smoke.sh` — 30s pre-flight check
- `scripts/experiment_w4a16/README.md` — 本文件
- `scripts/experiment_w4a16/results/<session>.txt` — 跑完后自动生成

未改动文件：

- `submission_bf16_native_bf16/` (v5g, 已成功提交)
- `submission_awq_llmcompressor/` (AWQ chunked-prefill 8K 旧版，留作 fallback)
- `scripts/local_eval.sh` (今晚 runner 直接调用)
- `scripts/experiment_5h/commit_push.sh` (复用其 git push 逻辑)

## 故障排除

**Phase 1 报 acc=?**：local_eval.sh 的 acc 提取 regex 失败。
读 `scripts/logs/w4a16_*/phase1_*.log` 找 `ori_accuracy` 或 `acc_ori` 字符串。

**Phase 2 quant OOM**：MAX_CALIB_LEN=8192 在 9B 模型 + 256 calib 上可能爆显存。
退路：`NUM_CALIB=128 MAX_CALIB_LEN=4096 bash run_plan.sh`，质量略降但跑得通。

**Phase 2 quant 卡很久** (> 2h)：llm-compressor 在 SALA 的 hybrid attention 上可能慢。
不正常卡死可以 `pkill -9 python` 然后用 `MLP_ONLY=1` (默认就是) 跑，绕开 attention 量化。

**push 失败**：commit_push.sh 已有 rebase 重试逻辑。如果还失败，跑完不影响，
人工 `git pull --rebase origin quant/w4a16 && git push` 即可。

## 不需要做的事

- ❌ 不要手动跑 quantize_to_w4a16.py（runner 通过 prepare_model.sh 调用）
- ❌ 不要直接 source `submission_*/prepare_env.sh`（它会装很多包；local 已经装好了）
- ❌ 不要提交 phase 1 或 phase 2 的 variant 到平台前没先看 results.txt
