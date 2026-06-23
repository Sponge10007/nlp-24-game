# Qwen2.5-1.5B 24 点 RLVR 实验日志

本文档记录从数据划分、低显存 GRPO、协议奖励修复、SFT 预热，到混合不可解训练的完整实验过程。实验时间为 2026 年 6 月，主要训练环境为 AutoDL 单张 RTX 4090 24GB。

## 1. 任务与评估规则

模型输入 3–4 个整数和目标值，输出：

```text
<think>...</think>
<answer>...</answer>
```

可解题要求 `<answer>`：

- 只包含 ASCII 数字、空格与 `+ - * / ( )`；
- 每个给定数字必须且只能使用一次；
- 表达式结果严格等于目标值；
- 不允许等号、解释文字、Unicode 运算符或数字拼接。

不可解题要求精确输出：

```text
<answer>UNSOLVABLE</answer>
```

主要评估方式：

- greedy pass@1：`temperature=0`；
- sampled pass@4：`temperature=0.7`、`top_p=0.95`，最多采样 4 次；
- 指标包括正确率、R1 格式率和错误类型分布。

## 2. 数据划分

原始 `nlile/24-game` 与 `test-time-compute/game-of-24` 当前版本包含同一套 1362 个四数组合，不能直接采用“全量 nlile 训练 + ToT 测试”。最终划分为：

| 数据集 | 数量 | 用途 |
| --- | ---: | --- |
| `data/train.jsonl` | 1162 | 可解训练题 |
| `data/test_hard_900_1000.jsonl` | 100 | ToT paper hard 测试 |
| `data/test_low_solved_rate.jsonl` | 100 | solved rate 最低测试 |
| `data/test_all_nonoverlap.jsonl` | 200 | 两个可解测试集的去重并集 |
| `data/unsolvable_test.jsonl` | 100 | 本地枚举的不可解 holdout |

数据生成顺序是先固定 hard/low 测试题，再从 nlile 训练集中剔除对应组合，因此可解训练集与测试集零重叠。不可解测试集从 1–13 的四数字组合中确定性枚举，并与后续不可解训练数据严格隔离。

## 3. 公共训练设置

- 基座模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 本地模型路径：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 微调方式：LoRA
- 24GB 主配置：
  - `lora_rank=32`
  - `lora_alpha=64`
  - `bf16=True`
  - `load_in_4bit=False`
- GRPO 主配置：
  - `num_generations=8`
  - `per_device_batch_size=2`
  - `grad_accum_steps=4`
  - `kl_coef/beta=0.05`
- SFT 使用 conversational prompt-completion 数据，并启用 `completion_only_loss=True`。

## 4. 结果总览

### 4.1 精确评估结果

| 版本 | 阶段 | Hard pass@1 | Low pass@1 | Unsolvable pass@1 | Hard pass@4 | Low pass@4 | 格式率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | Compact SFT | 5% | 1% | 0% | — | — | 100% |
| v3 | Compact SFT → GRPO | **10%** | **2%** | 0% | **16%** | **5%** | 100% |
| v4 | 多解 + 300 不可解 SFT | 2% | 0% | 1% | — | — | 100% |
| v4.1 | 单一规范答案 + 150 不可解 SFT | 1% | 2% | 0% | — | — | 100% |
| v5 | 从 v3 继续混合 GRPO | 11% | 0% | 0% | 16% | 2% | 100% |

可解测试集综合结果：

| 版本 | greedy pass@1 | sampled pass@4 |
| --- | ---: | ---: |
| v3 最佳模型 | **6.0%** | **10.5%** |
| v5 混合续训 | 5.5% | 9.0% |

综合来看，当前最佳模型是：

```text
outputs/experiments/v3_compact_sft_grpo/grpo/adapter
```

### 4.2 早期实验

早期 24GB GRPO 的用户汇总正确率约为 2%，但当时没有按当前版本目录保存完整分数据集 summary，因此不与上表中的精确结果混用。更早的低显存 LoRA8/G2 实验最后 100 reward step 平均 batch accuracy 约 0.5%，格式率约 84.9%，主要用于定位协议和 reward 问题。

## 5. 实验过程

### 5.1 早期低显存 GRPO：格式与正确性脱节

最初采用 4-bit、LoRA rank 8、`num_generations=2` 的低显存配置。模型逐渐学会 `<think>/<answer>` 外层格式，但正确率长期低于 1%。

主要错误：

- `<answer>` 包含 `=24`；
- 使用 `×`、`÷`、全角括号；
- 引入题目之外的数字；
- 漏用或重复给定数字；
- 表达式合法但结果错误。

这说明格式率上升不能直接视为解题能力上升。

### 5.2 Reward 与 prompt 协议修复

随后依次进行了：

1. `protocol_reward_lora8_g2`
2. `no_example_leak_lora8_g2`
3. `strict_ascii_answer_lora8_g2`
4. `number_usage_lora8_g2`

关键发现：

- 在 prompt 中加入具体正确示例会导致模板泄漏，模型大量复制 `8/(3-8/3)`；
- 删除具体示例后，复制问题消失；
- 加强 ASCII 协议惩罚后，格式率提高到约 93%；
- 此后主要瓶颈从非法字符转为 `number_mismatch` 和 `wrong_value`。

严格协议能够教会模型“怎么回答”，但不能自动教会模型“怎么算对”。

### 5.3 v2：DeepSeek 长 CoT SFT 失败

使用 `deepseek-reasoner` 对 500 道训练题做拒绝采样，每题最多尝试 16 次，仅保留本地裁判验证正确的 completion。500 道题最终全部获得正确样本。

第一版直接使用 DeepSeek 长 reasoning 训练。数据长度统计：

| 指标 | 数值 |
| --- | ---: |
| 平均长度 | 869.984 tokens |
| P50 | 715 |
| P90 | 1651 |
| 最大长度 | 2652 |
| 超过 768 tokens | 227/500 |
| 768 内没有完整 `</answer>` | 222/500 |
| 768 内没有完整 `</think>` | 220/500 |

结果表现为模型反复生成长篇推理，直到 `max_new_tokens` 截断，无法输出 `<answer>`。该实验说明：

- 训练 loss 正常下降不代表训练目标正确；
- 长 CoT 与固定长度截断会让模型主要学习“继续思考”；
- 必须确保答案标签完整可见，并让 loss 聚焦 assistant completion。

该失败版本保存在：

```text
outputs/experiments/v2_deepseek500_sft_grpo
```

### 5.4 v3：短算式 SFT → GRPO，当前最佳

将 500 条 DeepSeek 正确答案转换成纯算术轨迹，例如：

```text
<think>1+1=2; 2+1=3; 8*3=24</think>
<answer>8*(1+1+1)</answer>
```

同时修改 SFT：

- 使用 TRL 原生 conversational prompt-completion；
- `completion` 为 assistant 消息列表；
- `completion_only_loss=True`；
- `max_length=512`；
- 不再训练长篇自然语言 reasoning。

SFT 结果：

| 测试集 | pass@1 | 格式率 | 主要错误 |
| --- | ---: | ---: | --- |
| Hard | 5% | 100% | `wrong_value` |
| Low | 1% | 100% | `wrong_value` |
| Unsolvable | 0% | 100% | `fabricated_unsolvable` |

从 SFT adapter 继续 GRPO 后：

| 测试集 | greedy pass@1 | sampled pass@4 |
| --- | ---: | ---: |
| Hard | 10% | 16% |
| Low | 2% | 5% |
| Unsolvable | 0% | 0% |

GRPO 使 hard 和 low 的 greedy accuracy 相对 SFT 均翻倍，同时保持 100% 格式率。pass@4 的可解综合正确率达到 10.5%，说明模型偶尔能够生成正确解，但单次生成稳定性不足。

### 5.5 v4：每题多解 + 300 不可解 SFT

扩展到全部 1162 道可解训练题，每题确定性搜索最多 3 个答案，同时加入 300 道与测试集无重叠的不可解题。

SFT 结果：

| 测试集 | pass@1 | 主要错误 |
| --- | ---: | --- |
| Hard | 2% | 34% `number_mismatch` |
| Low | 0% | 33% `number_mismatch` |
| Unsolvable | 1% | 99% `fabricated_unsolvable` |

同一个 prompt 对应多个不同表达式，使 1.5B 模型出现答案模式混合；不可解能力只提高到 1%，却破坏了可解题能力。因此停止该版本的 GRPO。

### 5.6 v4.1：每题单一规范答案 + 150 不可解 SFT

为消除多解干扰，改为：

- 优先复用 v3 的 500 条教师答案；
- 其余题目由确定性搜索补齐；
- 每题只保留一个括号更少、长度更短的规范表达式；
- SFT 加入 150 道不可解题；
- 只训练 1 epoch，学习率 `1e-5`。

训练日志：

- 1312 条左右 SFT 数据；
- 164 optimizer steps；
- RTX 4090 上约 76.7 秒；
- loss 从 1.291 降至约 0.279；
- mean token accuracy 最终约 89.8%。

评估：

| 测试集 | pass@1 | 格式率 |
| --- | ---: | ---: |
| Hard | 1% | 100% |
| Low | 2% | 100% |
| Unsolvable | 0% | 100% |

尽管训练 loss 很低，泛化没有提升，`number_mismatch` 仍约 25–28%。这再次说明 SFT 拟合文本不等于学会组合搜索。

### 5.7 v5：从 v3 最佳模型继续混合 GRPO

不再重复 SFT，直接从 v3 最佳 GRPO adapter 出发，使用：

- 1162 道可解题；
- 100 道与测试集无重叠的不可解题；
- 3 epochs；
- learning rate `2e-6`；
- `num_generations=8`；
- `max_completion_length=512`。

结果：

| 测试集 | greedy pass@1 | sampled pass@4 |
| --- | ---: | ---: |
| Hard | 11% | 16% |
| Low | 0% | 2% |
| Unsolvable | 0% | 0% |

Hard pass@1 提高 1 个百分点，但 low 和综合结果退化，不可解能力仍为 0%。因此 v5 不是新的最佳模型。

## 6. 主要结论

1. **短算式 SFT 是有效预热。**  
   它将格式率稳定到 100%，并使错误从协议问题集中到真正的算术错误。

2. **SFT → GRPO 比直接 GRPO 更有效。**  
   v3 从 SFT 的 5%/1% 提升到 GRPO 后的 10%/2%。

3. **更多 SFT 数据不一定更好。**  
   多解标签和确定性搜索生成的大量表达式会导致小模型输出模式混合。

4. **不可解判断没有通过少量混合样本学会。**  
   v4、v4.1、v5 均未在不可解测试集取得稳定提升。

5. **当前瓶颈是算术搜索，不再是格式。**  
   最佳模型格式率 100%，数字不匹配很少，错误主要为 `wrong_value`。

6. **pass@4 明显高于 pass@1。**  
   v3 可解综合正确率从 6.0% 提升到 10.5%，说明能力存在但生成不稳定。

7. **1.5B 模型与最终答案奖励已出现平台期。**  
   后续若追求明显提升，应考虑更强基座、搜索增强、过程奖励或两阶段可解性判断，而不是继续堆相同训练轮数。

## 7. 最终推荐版本

报告主结果与演示应使用 v3：

```text
outputs/experiments/v3_compact_sft_grpo/grpo/adapter
```

报告中可将 v4、v4.1、v5 作为消融实验，说明：

- 多解 SFT 的负面影响；
- 训练 loss 与泛化能力不一致；
- 少量不可解混合训练不能解决可解性判断；
- 保守续训仍可能造成困难样本遗忘。

## 8. 结果文件

运行：

```bash
bash scripts/archive_lightweight_results.sh
```

脚本会把可提交的轻量结果复制到：

```text
reports/results/
```

归档内容包括可用的：

- `summary.json`
- `config.json` / `sft_config.json`
- 数据集 `summary.json` 与 `sha256.txt`
- `accuracy_curve.png`

不会归档：

- LoRA 权重；
- checkpoint；
- Hugging Face 模型；
- 完整逐样本生成结果；
- 大型训练日志。

