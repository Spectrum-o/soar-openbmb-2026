# POSTQ FP8KV 失败清单 / 代码审查 (2026-06-02)

> **范围**：`submission_..._fp8kv_hp224_postq_*` 这一系 FP8KV「POSTQ」包。它们反复在
> `prepare_model.sh` **step-3 的「第二次 `GPTQModel.load` 重载」** 里崩溃，至今**从未跑到
> platform eval 阶段**（全部 ⛔ 死在 `prepare_model`，eval slot 未被消耗，但每次烧一次提交 +
> ~54min prepare）。
>
> ⚠️ **该变体目录正由 Codex 与本会话并发编辑，工作树是移动目标**（observed: 16:58→17:37 多次
> 变更）。本文档以**已确认的 platform 报错**和**reload 机制层面的缺陷**为主——这些不随行号漂移；
> 涉及工作树当前状态处均标注「截至 HH:MM」。
>
> **环境约束**：本机无 GPU、无 `transformers`、无 `gptqmodel`、无 gptqmodel 源码 →
> 本文档为**纯静态分析 + platform 实测日志**。证据等级见 §6。

---

## 0. 一句话结论

POSTQ 是一条「**修一个、露下一个**」的 gptqmodel-7 × transformers-4.57.1 兼容 gauntlet。
`#1 → #6` 已逐个被撞出。最新提交的 **`DYNAMIC_FIX_153306` 死于 #6（config-class 不一致，
已 platform 实测确认）**；其修复（reload 时临时恢复 remote AutoConfig）已于 **~17:37 由 Codex 在
工作树接线，但未经 GPU 验证**。**修通 #6 后还有 #7/#8/#9 三个尚未被触达的预测错误。** 即便全部
修通，POSTQ 的 `final_score` 天花板 **~21.0–21.3 仍 < 已封存的 21.79** → 见 §5 战略建议。

---

## 1. 两个产物 / 当前状态（截至 17:37）

| | **TARBALL `DYNAMIC_FIX_153306`**（15:36 build，已提交） | **工作树变体目录**（Codex 在改） |
|---|---|---|
| `quantize` md5 | `652c2651…`（静态、不再变） | `3f6a62b8…`@17:37（移动目标） |
| #1–#5 | ✅ 全过 | ✅ 全过 |
| #6 config-class | ❌ **死于此**（16:36 实测） | 🔧 修复已接线（待 GPU 验证） |
| NameError（16:58 中途态） | 无 | ✅ 已消除（`register_minicpm_sala_hf_config` def=0/call=0） |
| 可提交性 | 会撞 #6，**勿再交** | 需重新打包；但先解决 §3/§4 |

**时间线**：`15:36 build → 15:41 submit → 16:36 FAILED(#6) → 16:58 起 Codex 改（中途出现 NameError 半成品）→ 17:37 #6 修复接线完成`。

---

## 2. 失败 gauntlet（按 reload 执行顺序）

`run_post_quant_kv_calibration()` 启动一个**独立子进程**重载最终 artifact。顺序：
`register/stub → GPTQModel.load（config 解析 → 模块构建 → 权重加载）→ forward 采样 → 写 KV scale → inject`。

证据等级：✅`CONFIRMED_PLATFORM`（实测 traceback） / 🔶`STATIC_CERTAIN`（代码可证、无需运行） / 🔮`PREDICTED`（需 GPU 确认，附置信度）。

### § A 历史链 —— 之前的包已撞、已修（**勿回退**）

| # | 包 / 日期 | 错误 | 根因 | 修复（所在） | 等级 |
|---|---|---|---|---|---|
| 1 | `POSTQ_0339` | reload 无法解析 `minicpm_sala` | `config.json` 故意删了 `auto_map.AutoConfig`，reload 端没手动注册 | 加 `AutoConfig.register('minicpm_sala')`（后演化为 §B #6 的修复） | 🔶/superseded |
| 2 | `POSTQ_0418`；DENSEQKV `0115` | `TypeError: __init__() got unexpected kwarg 'offload_to_disk'` | `offload_to_disk` 被当作 `GPTQModel.load()` kwarg → 透传进 `MiniCPMSALAForCausalLM.__init__` | offload 只放 `QuantizeConfig`，不传 `load()`（`load_model` docstring ~796-802） | ✅（兄弟 0115 实测） |
| 3 | `MODEL_MAP_FIX_113604`（submit 11:41 → **FAIL 12:36**） | `ImportError: cannot import name 'PreTrainedConfig' from 'transformers'` | gptqmodel 7 写的是 `from transformers import PreTrainedConfig`；4.57.1 只 lazy 导出 `Pretrainedconfig`（小写 c）。旧 shim 只设属性、没补 LazyModule 映射 | `_stub_transformers_for_gptqmodel_7` 补 `_objects/__all__/_import_structure/_class_to_module` + 校验 import（`quantize…py:65-133`） | ✅ |
| 4 | `COMPAT_FIX_132839`（submit 13:55 → **FAIL 14:49**） | `AttributeError: 'bool' object has no attribute 'items'` in `gptqmodel/quantization/config.py:_normalize_dynamic_layer_config` | 我方 dynamic 写 `"-:regex": true`；gptqmodel 7 reload 对每个 value 调 `.items()`（期望 dict） | 所有负规则 value 改 `{}`（`write_sglang_compatible_quant_config` dynamic dict）；对 SGLang 语义透明（`utils.py:254-258` 只看 key 前缀） | ✅ |
| 5 | （即 `DYNAMIC_FIX` 本身） | — | — | `#4` 的修复，不是独立错误 | — |

### § B 最新包 `DYNAMIC_FIX_153306` 的致命错误（本次重点）

**#6 — config-class 不一致** ✅`CONFIRMED_PLATFORM`（submit 15:41 → **FAIL 16:36**）

```
ValueError: The model class you are passing has a `config_class` attribute that is not
consistent with the config class you passed (model has
 <class '...configuration_minicpm_sala.MiniCPMSALAConfig'>  and you passed
 <class 'sglang.srt.configs.minicpm.MiniCPMHybridConfig'>. Fix one of those so they match!

run_post_quant_kv_calibration → GPTQModel.load → from_quantized → loader.from_config
  → transformers/models/auto/auto_factory.py:449 cls.register(config.__class__, model_class)
  → auto_factory.py:624  raise ValueError(...)
```

- **根因**：reload 端原先调 `register_minicpm_sala_hf_config()`，把 `model_type=minicpm_sala`
  注册到 **SGLang 的 `MiniCPMHybridConfig`**。于是 `config.__class__ = MiniCPMHybridConfig`。
  但 `config.json` **保留了 `auto_map.AutoModelForCausalLM`** → trust_remote_code 解析出**远程
  `MiniCPMSALAForCausalLM`**，其 `config_class = 远程 MiniCPMSALAConfig`。两者不一致 → `cls.register` 抛错。
- **修复**（作者/Codex 思路，已于 ~17:37 接线，**未经 GPU 验证**）：
  `_temporarily_restore_remote_autoconfig_for_postq_reload(output_dir, Path(args.input))` —— 仅在
  reload 期间把 `auto_map.AutoConfig` 改回**远程 `MiniCPMSALAConfig`**，让 transformers 看到一致的
  `远程Config → 远程ForCausalLM`；`finally` 里 `_restore_postq_reload_config` 还原 SGLang serving 配置。
  （其 docstring 原话就点名了上面的 ValueError。）
- **状态**：tarball 153306 **没有**此修复 → 必撞 #6；工作树**已接线** → 需重新打包后 GPU 验证。

### § C 修通 #6 后，**下一个会崩的地方**（尚未被触达 —— 本文档最有价值的部分）

> reload 在 platform 上**从未越过 config 解析阶段**，以下都在「权重加载 / forward」更深处，
> Codex 若只盯着 #6，修完仍会一个个撞上。建议**一次性预判处理**。

**#7 — skip 名不匹配 → 还原层 `.qweight` 缺失** 🔮`PREDICTED_HIGH`
- selective overlay 把 **last-7 lightning attn 的 q/k/v/o 还原成 BF16 并物理删除其 `.qweight/.qzeros/.scales/.g_idx`**，
  同时往 `quantize_config.json` 写跳过规则——用的是 **SGLang 融合名** `-:…self_attn.**qkv_proj**$`、`-:…mlp.**gate_up_proj**$`
  （`apply_lightning_skip_overlay.py:211-216`）。
- 但 gptqmodel reload 的 `layer_modules/module_tree` 用 **HF 拆分名** `q_proj/k_proj/v_proj`、`gate_proj/up_proj`
  （`quantize…py:540-555`）。`re.match("…qkv_proj$", "…q_proj")` **不匹配** → gptqmodel 认为 q/k/v 仍是量化的 →
  去找已被删除的 `.qweight` → **KeyError / 缺权重**。（`o_proj$`、`down_proj$` 名字一致，能匹配，不受影响。）
- **作者自己的注释已警告此类**（`quantize…py:1136-1142`）：「quant-time 与 load-time skip 必须**精确匹配**，否则
  GPTQModel 给某层写了 `.qweight` 而另一侧当 BF16 加载 → **v14 式 KeyError**……**必须在 GPU 上验证**」。
- **不确定性**：若 gptqmodel 改用「张量存在性」判断（见到 `.weight` 就当 BF16）则可能不崩。**需 GPU 实测。**
- **预修**：给 reload 端补 HF 名跳过规则 `-:…q_proj$/k_proj$/v_proj$/gate_proj$/up_proj$`（针对 last-7 还原层），
  或确认 gptqmodel 的回退行为。

**#8 — forward 走 HF SALA modeling，需稀疏 kernel / flash_attn** 🔮`PREDICTED_MED`
- `qmodel(ids)`（`quantize…py:~1404-1419`）跑的是 **HF `modeling_minicpm_sala` 前向**（不是 SGLang
  `minicpm_flashinfer` 后端）。dense 层需 **InfLLM-v2 稀疏 CUDA kernel**、`flash_attn`、以及 `flash_attention_2`
  断言（已由 `force_flash_attention_2_for_sala` 处理）。
- 任一 dense 层前向失败 → 逐 prompt 被 `except` 吞；若**全失败** → `ran_forwards==0` 硬抛；若**某 dense 层 hook 没触发**
  → `unvisited` 硬抛（`quantize…py:~1430-1437`）。
- **需 GPU 实测**：prepare 环境是否齐备稀疏 kernel（SOAR base env 装了 flash_attn，但稀疏 kernel 需确认）。

**#9 — qzeros `0x88`(for Marlin) artifact 回喂 gptqmodel** 🔮`PREDICTED_LOW–MED`
- artifact 的 qzeros 已被 `0x77→0x88` 打补丁（**为 Marlin**）。gptqmodel reload 若用**非-Marlin** 后端反量化，
  zero-point 约定可能差 1 → 反量化偏置 → 前向激活失真。KV 标定只取 |max| 量级，**可能**仍出可用 scale，
  但也可能 gptqmodel 校验 qzeros 而报错。**需 GPU 实测。**

**其它待查**：tokenizer reload（`AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)`
是否拿到正确 chat_template）、`KV_CALIB_TIMEOUT_MIN` 是否够、device/offload 是否影响 forward、
`kv_calibrate.py --inject` 写入的 key 名（`model.layers.N.self_attn.attn.k_scale/v_scale`）是否与运行期一致。

### § D 工程卫生 / 流程（多为 🔶`STATIC_CERTAIN`）

| 项 | 说明 | 风险 |
|---|---|---|
| 变体目录 **git-untracked、修复 uncommitted** | `git ls-files` 该目录为空；`git log --all` 该 quantizer 路径为空 | `git clean` 或换机即**全丢**；建议固定一个提交点 |
| 工作树残留 `__pycache__/*.pyc` | 坏文件被 import 过留下的 | 若直接 `--pack` 该目录，audit `FAIL no __pycache__` |
| **兄弟变体仍是坏版本** | `postq_offload`/`postq_perhead` 等仍停在 #3 弱 shim（`hasattr`-only ~82-83）+ #4 `: true`（~979-983） | 打错兄弟包 → 静默回退到 #3/#4 |
| **audit 是纯字符串匹配** | `audit_fp8kv_realcalib_package.py` 全是 substring 检查 | **抓不到** NameError / #6 / 任何 reload 行为；**audit PASS ≠ 能跑** |
| **Codex 并发编辑** | 工作树持续变动 | 打包前务必快照确认；勿与 Codex 同时改同一文件 |

---

## 3. 一次性修复建议（本地可做的）

1. **#6**：保持 Codex 已接线的 `_temporarily_restore_remote_autoconfig_for_postq_reload`（load 前）+
   `_restore_postq_reload_config`（`finally`）。**确认 `finally` 覆盖到异常路径**（reload 抛错时也要还原 config）。
2. **#7（预修，省一次 GPU 往返）**：在 reload 读取的 `quantize_config.json` 里，为 last-7 还原层**补 HF 名跳过规则**
   `-:model.layers.{i}.self_attn.q_proj$` 等（与现有融合名规则并存）；或在 `register_minicpm_sala_with_gptqmodel`
   里确认 gptqmodel 对「config 量化但张量缺失」的回退行为。
3. **卫生**：删 `__pycache__`；把工作树修复**提交**（至少固定一个 commit 再打包）；把 #3/#4/#6 修复**回灌到 base quantizer**，
   避免兄弟变体回退。
4. **本地能验证的极限**：`python3 -m py_compile`；AST「调用是否有定义」自查（本会话已做，当前无未定义调用）；
   **真正的 reload 行为（#7/#8/#9）必须 GPU 上 `GPTQModel.load` 复现**——见 §4。

---

## 4. 本地可修 vs 必须 GPU 验证

| 错误 | 本地（无 GPU）能否定/修 | 必须 GPU 复现确认 |
|---|---|---|
| #1–#4 | ✅ 已修，静态可核 | 否 |
| #6 config-class | ✅ 修复可静态接线 | ✅ 接线后须 `GPTQModel.load` 复现 |
| #7 skip 名 / 缺 qweight | 🔶 可预修（补 HF 名规则） | ✅ gptqmodel 回退行为只能 GPU 确认 |
| #8 forward / 稀疏 kernel | ❌ | ✅ 必须 GPU |
| #9 qzeros 回喂 | ❌ | ✅ 必须 GPU |

**GPU 一到就跑的最小复现**（不烧 platform slot，几分钟暴露 #6…#N）：在 `transformers 4.57.1 + gptqmodel 7.0.0`
环境里，对一个真实的「qzeros-fixed + overlay-mutated」artifact 直接
`python3 quantize_gptqmodel_w4a16.py --post-quant-kv-only --input <bf16> --output <artifact> …`，
或最小化地 `GPTQModel.load(<artifact>, trust_remote_code=True)`。这比每次 ~54min 的 platform 试错快两个数量级。

---

## 5. EV 与战略建议

- **天花板 < 21.79**：FP8KV 在 SALA 的 `--dense-as-sparse` 路径上**无速度增益**（S1 650–656 > 非-FP8 记录的 648），
  而速度主导相对分；POSTQ 相对 DENSEQKV（78.67/20.87）只多**零点几 pp** 的 scale-source。要破 21.79 需「acc +~1pp **且**
  S1 持平 648」同时成立，趋势不支持。realistic ceiling ~21.0–21.3。
- `SUBMISSIONS.md` 第 19 行自陈：「**W4 scale recalibration 才是 scoring route**」。
- **建议**：POSTQ 当作「把 reload 这条工程链一次性跑通」的工程目标即可；**真正的得分精力应放到非-FP8 记录上的
  W4 scale recalibration A/B**（本地、不烧 slot、speed-neutral）。详见 `experiments/W4_SCALE_RECALIB_RUNBOOK_20260601.md`
  与记忆 `project_full_selective_bf16`。

---

## 6. 证据等级与出处

- ✅`CONFIRMED_PLATFORM`：#2（兄弟 0115 实测）、#3（113604, FAIL 12:36）、#4（132839, FAIL 14:49）、**#6（153306, FAIL 16:36，traceback 见 §B）**。来源：`SUBMISSIONS.md:174-185` + 用户粘贴的 entrypoint.log。
- 🔶`STATIC_CERTAIN`：#7 的名字不匹配证据（`apply_lightning_skip_overlay.py:211-216` 融合名 vs `quantize…py:540-555` HF 名）、§D 全部。
- 🔮`PREDICTED`：#7（HIGH，含作者 `quantize…py:1136-1142` 自警）、#8（MED）、#9（LOW–MED）—— 均需 GPU `GPTQModel.load` 复现确认。
- 分析方法：本机无 GPU/transformers/gptqmodel → 纯静态读码 + platform 日志；gptqmodel 内部行为处均标 PREDICTED，未臆造。

> 维护提示：本文档与并发的 Codex 编辑解耦——§A/§B（已确认错误）与 §C（reload 机制层缺陷）不随工作树行号漂移；
> §1/§3 涉及工作树当前态处已标「截至 17:37」，复核时以 `SUBMISSIONS.md` + 最新工作树为准。

---

## 7. 第二轮复查（17:38–18:17）—— #6 实测确认、#7 已修打新包、serve 期新隐患

### 7.1 平台时间线更新
- **`DYNAMIC_FIX_153306`：15:41 提交 → 16:36 实测 FAILED** with `ValueError: ... config_class ... MiniCPMSALAConfig ... not consistent with ... MiniCPMHybridConfig`（`auto_factory.py:624`）→ **#6 由 PREDICTED 升为 CONFIRMED_PLATFORM**，与本文档 §B 预测一致。
- Codex 随后产出 **`RELOAD_CONFIG_FIX_20260602_173814`**（md5 `d4a57c07`）：修 #6（`_temporarily_restore_remote_autoconfig_for_postq_reload`，reload 时临时把 `auto_map.AutoConfig` 改回远程 `MiniCPMSALAConfig`），audit 全 PASS。**但它没碰 #7。**

### 7.2 #7 升级为 CONFIRMED-REAL（两路独立分析一致）
两条独立证据链都确认 #7 是 `RELOAD_CONFIG_FIX` 修完 #6 后**下一个会崩的点**（prepare step3 的 `GPTQModel.load`，在 forward 之前）：
- 运行期 `MiniCPMLightningMixer`/`MiniCPMAttention` 都是融合 `qkv_proj`（`overlay/minicpm.py:204,268`），checkpoint 是 HF 拆分 `q_proj/k_proj/v_proj`（loader 映射 `:616-618`）；gptqmodel `module_tree` 对**所有层**用拆分名（`quantize…py:615-621`，无按层类型分支）。
- overlay 物理删除还原层的 q/k/v/o 量化张量 + 只写**融合名**跳过规则 `qkv_proj$`/`o_proj$`（`apply_lightning_skip_overlay.py:188-194,211-214`）。
- reload 端不翻译名字 → `qkv_proj$` 匹配不上 gptqmodel 的 `q_proj/k_proj/v_proj` → 还原层 q/k/v 被当成量化却缺 `.qweight`。
- **非对称细化（STATIC_CERTAIN）**：`o_proj$` 能匹配 gptqmodel 的 `o_proj`（正确跳过），但 `qkv_proj$` 匹配不上 q/k/v → **崩点必在某还原 lightning 层的 q/k/v（不是 o_proj）**。
- 名字不匹配是 **STATIC_CERTAIN**；gptqmodel 对「config 说量化但张量缺失」是报错还是回退 = **PREDICTED_HIGH**（仅 GPU 可定）。

### 7.3 修复 + 新包（本会话产出）
**修法（reload 侧，而非 overlay 侧）**：在 `quantize_gptqmodel_w4a16.py` 新增 `_augment_dynamic_skip_aliases_for_reload(output_dir)`，在 `GPTQModel.load` 前（且在 auto_map 临时补丁**捕获 `original_config_text` 之前**）由现有融合规则**派生 HF 名跳过别名**（`qkv_proj$→q/k/v_proj$`、`gate_up_proj$→gate/up_proj$`），同时写入 `quantize_config.json` 与 `config.json[quantization_config]`。
- **为何放 reload 侧而非 overlay**：overlay 的 `sglang_dynamic_skip_rules_for_layer` 被单测**锁死**（`test_apply_lightning_skip_overlay.py:491-522` 断言「必须融合名、不是 HF 名」+ `assertEqual(new_rules_count, 6/4)` 精确计数）。在 overlay 加 HF 名会破这些单测、违背 SGLang 设计。reload 侧派生别名则：对 SGLang 无害（融合模块匹配不上 HF 名别名）、不动 overlay/单测、纯超集、幂等。
- **验证**：`py_compile` OK；别名逻辑独立验证（从 `qkv_proj$` 正确派生 q/k/v，不误伤 `o_gate/z_proj`，`o_proj` 同名不重复，幂等）；`pack_submission --check-only` OK；新包 audit **22 PASS / 0 FAIL**；回归 `test_apply_lightning_skip_overlay`(28) + `test_fp8kv_realcalib_tools`(48) 全 OK。
- **最终提交包**：**`soar_fp8kv_POSTQ_MULTI8K300_HP224_RELOAD_ALIAS_FIX_20260602_183237.tar.gz`**，md5 `ecab79cc5fb1a263650842a43130d4f9`，248MB。= Codex 的 #1–#6 + #7 修复，并 supersede 早先的 `181703`/`183152` 中间包。

### 7.4 新发现的 serve 期隐患（PREDICTED_MED，非崩溃）
**POSTQ 的测量源 ≠ 服务源**：POSTQ 用 gptqmodel **HF-Marlin reload 前向**测 K/V abs-max，但服务端用 **SGLang gptq_marlin** 内核填 fp8 KV——两套 dequant 实现不同 → 标定出的 scale 可能与运行期 KV 分布错配（可能更好也可能更差）。这是 POSTQ 相对 DENSEQKV 的**唯一数值差异**：DENSEQKV 测 BF16 base（测量源==服务源），已实测 78.67/20.87。**含义：POSTQ 未必胜过 DENSEQKV，必须当 A/B 看，别假设更优**（`safe_max=224` 有 2x 头空间，不会硬饱和，只是 acc 偏移）。

### 7.5 已核对正确（CLEARED，valuable negatives）
1. **scale 方向无反转**：写入 `cache/k_scale`（`memory_pool.py:1014-1027`），读取乘回；`k_scale=k_max/safe_max` 一致。
2. scale 确实接进 fp8 路径（非默认 1.0）；`self_attn.attn.k_scale` key 名一致 + `maybe_remap`。
3. **BF16 还原 × fp8-KV 共存 OK**：还原的 last-7 是 **lightning 层（递归状态、无 KV cache、无 RadixAttention）**，被 KV 测量与注入**双双跳过**；只有 dense 层写 fp8 KV——与服过 78.67 的家族一致。
4. `safe_max=224` vs e4m3 448 = 2x 头空间，吸收 RoPE ≤1.41x 增长（最坏 ~316<448）。
5. `kv_calibrate --inject` 物理写 shard 0 + 更新 index，正确。
6. per-head scale 路径是**潜伏死分支**（flashinfer 读核硬编码 k_scale=1.0）——当前标量 scale **不触发**；仅未来 per-head 变体需防。
7. `lastN-lightning` 选择器在层数<N 时静默全选——32 层 SALA **不触发**。

### 7.6 仍存的 GPU-only 风险（本地无法定）+ EV
- **#7 修复本身未经 GPU 验证**（Pareto 安全：若 gptqmodel 本就回退则无害，若会崩则修好）。
- **#8**：reload 前向走 HF SALA modeling，需 InfLLM-v2 稀疏 kernel + flash_attn 可用，否则 `ran_forwards==0` 硬抛。
- **#9**：qzeros `0x88`(Marlin) artifact 回喂 gptqmodel 的反量化约定。
- **EV 仍为负**：POSTQ 天花板 ~21.0–21.3 < 21.79，且 7.4 表明未必胜 DENSEQKV。POSTQ 只校 FP8 KV scale，不会重写 W4 `.scales`；下一条得分主线应是非-FP8 `last7attn_rmsopfusion_calib300_multi120` 上的 W4 scale recalibration A/B。

### 7.7 验证工具（无需烧 platform）
新增 **`scripts/postq_reload_smoke.sh`**：在 GPU 上隔离运行 prepare **step3**（`--post-quant-kv-only`），对一个已 prepare 的 artifact 几分钟跑完，明确报出撞到 #6/#7/#8 的哪一个（并预先 grep artifact 的 dynamic 规则判断 #7 guard 是否在位）。**有 GPU 时先跑它，别再盲交 platform。**
