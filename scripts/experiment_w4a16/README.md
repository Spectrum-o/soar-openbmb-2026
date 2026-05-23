# W4A16 迁移实验 — 6 phases，无人值守

> 2026-05-23 准备完成。GPU 一到位 3 步跑完，覆盖 6 个 ablation。

## 背景

v5g (BF16 native, chunked-prefill 8192) 平台 `acc_ori=83.04`，首次非零
final_score=15.76。证明：
- bf16 family 正确，fp16 sed-patch 是 acc 卡在 ~47 的根因（已永久去除）
- v5g 的"地基"（auto_map 剥除、transformers 4.57.1、tokenizer 覆盖等）必须保留

**官方 W4A16 配方** (`W4A16_README.md` + `run_sala_w4a16.sh`)：
1. **llm-compressor + GPTQ** → compressed-tensors 格式（NOT gptq_marlin）
2. SGLang `--quantization compressed-tensors`
3. Serving: `--chunked-prefill-size 65536 --max-prefill-tokens 65536 --max-running-requests 32 --mem-fraction-static 0.80`
4. 不设 `--dtype`（默认 bfloat16）

## 6 个实验 phases

| # | 变体 / 修改 | 改的变量（vs 上一个）| 预算 | 信息价值 |
|---|---|---|---|---|
| **P1** | `submission_bf16_official_args` | （vs v5g）+ 官方 65K/0.80 serving args | 25 min | 控制：65K args 不破 acc 吗？ |
| **P2** | `submission_awq_official_args` | （vs P1）+ AWQ W4A16 + compressed-tensors | 1h25 | **今天最后一发主目标**：W4A16 路径走通否 |
| **P3** | `submission_gptq_official_args` | （vs P2）AWQ → GPTQ（官方推荐）| 1h25 | A/B：AWQ vs GPTQ 谁更适合 SALA |
| **P4** | `submission_awq_official_args` w/ `NUM_CALIB=512` | （vs P2）calib 256→512 | 1h35 | calibration 还能压分吗（F1 hint）|
| **P5** | `submission_awq_official_args` w/ `MLP_ONLY=0` | （vs P2）量化 attention（含 lightning）| 1h55 | lightning 递推能否承受 4-bit |
| **P6** | apply_lightning_skip_overlay.py on P2 artifact | 把 P2 的 lightning MLPs 反量化回 BF16 | 30 min | 全 BF16 lightning + W4A16 dense 复合 |

**Total: ~6h35min**（200 samples eval per phase）

## 3 步工作流（登服务器之后）

### Step 1：30 秒环境 sanity（无 GPU）

```bash
cd /root/autodl-tmp/zyn/sglang
git pull origin quant/w4a16
source sglang_minicpm_sala_env/bin/activate
bash scripts/experiment_w4a16/smoke.sh
```

期望 **ALL 39 CHECKS PASSED**（多了 GPTQ 变体和 lightning-skip 工具的检查）。

如果 llmcompressor 缺 → 跑：

```bash
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple "llmcompressor>=0.7"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple --force-reinstall "transformers==4.57.1"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple "huggingface-hub>=0.34.0,<1.0" "tokenizers>=0.22.0,<0.23.0"
```

### Step 2：15 分钟 pipeline smoke（验证 P1+P2 端到端）

```bash
bash scripts/experiment_w4a16/run_plan.sh --smoke
```

- Smoke 模式**只跑 P1+P2**（P3-P6 需要完整 quant artifact，smoke 跑了浪费）
- 末尾打印 `completed`（不是 FAILED）就 OK

### Step 3：6.5 小时全跑（无人值守）

```bash
nohup bash scripts/experiment_w4a16/run_plan.sh \
    > /tmp/w4a16_$(date +%Y%m%d_%H%M%S).log 2>&1 &
disown
echo "PID: $!"
```

每个 phase 跑完自动 commit + push。结束写 `scripts/experiment_w4a16/results/<session>.txt`。

## 决策矩阵（自动算）

runner 末尾跑这套逻辑（本地→平台落差 +3pp，本地 75 ≈ 平台 78）：

| 条件 | 推荐提交 |
|---|---|
| P1 < 75（控制 fail，serving args 破坏 acc）| v5h 兜底（chunked-prefill 32K） |
| best of (P2,P3,P4,P5,P6) ≥ 75 | 那个最高的变体 |
| best of (P2,P3,P4,P5,P6) 70-74 | BF16 + 官方 args（W4A16 边界，保守稳进）|
| best of (P2,P3,P4,P5,P6) < 70 | BF16 + 官方 args（W4A16 全失败）|

## 监控（不用守着）

```bash
# 实时 master log
tail -f scripts/logs/w4a16_*/master.log

# GPU 用量
watch -n 5 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv'

# 进程
ps -ef | grep run_plan.sh | grep -v grep
```

提前杀：

```bash
pkill -f run_plan.sh
bash scripts/killall_sglang.sh
```

## Resume 支持

如果某个 phase 跑挂了，重启从下一个 phase 开始：

```bash
bash scripts/experiment_w4a16/run_plan.sh --from 4   # 跳过 P1,P2,P3
```

注意：**P6 依赖 P2 的 quant artifact**——如果 P2 失败或还没跑过，P6 会自动 skip 并写 `skipped: no P2 artifact`。

## 已交付的内容

新增/更新文件：

```
submission_bf16_official_args/      # P1 控制变体（v5g + 65K args）
submission_awq_official_args/       # P2/P4/P5 主变体（AWQ + 65K + compressed-tensors）
submission_gptq_official_args/      # P3 新增变体（GPTQ via llmcompressor）
├── prepare_env.sh
├── prepare_model.sh
├── quantize_llmcompressor_gptq.py  # 镜像 AWQ 脚本但用 GPTQModifier
└── README_SUBMISSION.md

scripts/experiment_w4a16/
├── run_plan.sh                     # 6-phase 自动 runner
├── smoke.sh                        # 39 项 sanity（GPU 在时 100% PASS）
├── README.md                       # 本文件
└── results/<session>.txt           # runner 跑完自动写

tools/apply_lightning_skip_overlay.py  # 已存在（2026-05-23 凌晨），P6 用
```

## 几点提醒

- **不要直接 source `submission_*/prepare_env.sh`**——venv 已有大部分包，让 prepare_env 装容易污染版本
- **不要在 P2 还没跑完就启动 P6**——会自动 skip 但仍占 phase 序号
- **不要在没读 `results/<session>.txt` 之前 yolo 提交**
- **如果 P5 卡很久**：full-attention 量化可能 OOM 或慢；超过 3h 不动就 `pkill -9 python` 强杀，跳过 P5（P5 失败不影响 P3/P4/P6）

## 5h 实验的旧结果不直接受用

凌晨的 5h pipeline 给了：F1_multi_adaptive 51.50 (+7.5pp 旧路径) 等。这些**全在 gptqmodel + gptq_marlin + fp16 sed-patch 老路径上**，与今晚的 compressed-tensors + bf16 路径正交。只有"calibration 多样本可能有用"这个信号被 P4 (NUM_CALIB=512) 间接验证。

如果今晚 P4 也观察到 calibration 是杠杆，**明天的 follow-up** 可以考虑把 F1_multi_adaptive 的多窗口 adaptive calibration 移植到 `quantize_llmcompressor_awq.py`。今晚不做。
