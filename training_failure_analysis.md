# GRPO 训练失败现象分析

本文档整理 `grpo_qwen25_1_5b_lora8_g2` 这次训练 run 的现象，供报告撰写使用。核心结论是：模型学会了一部分 R1 输出格式，但没有稳定学会产出合法且正确的 24 点表达式。

## 1. 实验设置

- 训练集：`data/train.jsonl`，1162 条可解 24 点题。
- 基座模型：`Qwen/Qwen2.5-1.5B-Instruct`。
- 训练方式：4-bit 量化加载基座模型，使用 LoRA 训练适配器。
- 主要训练参数：
  - `lora_rank=8`
  - `num_generations=2`
  - `max_completion_length=384`
  - `num_train_epochs=6`
  - `learning_rate=5e-6`
  - `gradient_accumulation_steps=8`
- 主要产物：
  - `outputs/final_model/`：最终 LoRA adapter，不是完整 Qwen 模型。
  - `outputs/checkpoints/`：训练中间 checkpoint，可用于恢复训练或比较不同 step。
  - `outputs/runs/grpo_qwen25_1_5b_lora8_g2/`：本次 run 的配置、训练指标、样例日志和曲线图。

## 2. 量化结果

主要依据：

- `outputs/runs/grpo_qwen25_1_5b_lora8_g2/training_metrics.csv`
- `outputs/runs/grpo_qwen25_1_5b_lora8_g2/accuracy_curve.png`

本次完整 run 共记录 1740 个 reward step。最后 100 个 reward step 的均值如下：

| 指标 | 数值 |
| --- | ---: |
| 平均 batch accuracy | 0.50% |
| 平均 format rate | 84.88% |
| 平均 correctness reward | -0.4469 |

最后 100 个 reward step 的主要错误类型累计如下：

| 错误类型 | 次数 | 说明 |
| --- | ---: | --- |
| `illegal_character_count` | 405 | `<answer>` 中包含等号、中文乘号、文字等裁判不允许的字符 |
| `number_mismatch_count` | 231 | 没有严格使用题目给定的四个数字，或凭空引入其他数字 |
| `missing_answer_count` | 83 | 没有可提取的 `<answer>` 内容 |
| `wrong_value_count` | 65 | 表达式合法、数字也匹配，但计算结果不是 24 |
| `false_unsolvable_claim_count` | 12 | 可解题误报为不可解 |
| `correct_count` | 4 | 正确样本很少 |
| `syntax_error_count` | 0 | 语法崩溃不是主要问题 |
| `division_by_zero_count` | 0 | 除零不是主要问题 |

整体均值也偏低：

| 指标 | 全 run 均值 |
| --- | ---: |
| `batch_accuracy` | 0.2874% |
| `smoothed_accuracy` | 0.2767% |
| `format_rate` | 76.6236% |
| `mean_correctness_reward` | -0.4626 |
| 最高 `smoothed_accuracy` | 0.88% |

这些数据说明：训练确实提高了格式服从程度，但 correctness reward 长期偏低，模型几乎没有稳定获得正奖励。报告里不应把这次结果表述为 RL 显著提升，而应描述为一次低显存 GRPO 设置下的不稳定训练或失败训练。

## 3. 错误类型分析

### 3.1 格式率提升不代表解题能力提升

最后 100 step 的平均 `format_rate` 已接近 85%，说明模型越来越经常输出 `<think>...</think>` 和 `<answer>...</answer>`。但是 `batch_accuracy` 仍只有约 0.5%，说明模型只学到外层标签，未学会 `<answer>` 内部必须是一个可验证表达式。

### 3.2 最大问题是 answer 通道约束不稳

`illegal_character_count` 是最大错误来源。典型问题包括：

- 在 `<answer>` 里写完整等式，例如 `3*6+7-13=24`。
- 使用中文乘号 `×`，裁判只接受 ASCII `*`。
- 在 `<answer>` 中写解释性文字或不属于表达式的符号。

当前裁判要求 `<answer>` 只能包含数字、`+ - * / ( )` 和空白。如果答案写成 `表达式 = 24`，即使人能理解，也会被判为非法字符。

### 3.3 数字使用约束没有学稳

`number_mismatch_count` 是第二大错误来源。模型经常凭空引入题目没有给出的数字，或者没有把给定数字各用一次。这说明它还没有稳定遵守 24 点任务的核心约束。

### 3.4 算术本身不是唯一瓶颈

`wrong_value_count` 也存在，但低于非法字符和数字不匹配。这说明在当前阶段，模型还没有进入“主要差在算术搜索”的状态；更前置的问题是生成格式和数字约束没有被学稳。

## 4. 定性样例

以下样例来自训练末尾的 `train_log.txt`，代表训练结束时模型仍然存在的问题。

### 4.1 `<answer>` 中包含等号

题目：`nums=[3, 6, 7, 13]`

模型输出：

```text
<answer>3*6+7-13=24</answer>
```

裁判结果：`illegal_character`

原因：`<answer>` 中包含 `=`。当前裁判只接受最终表达式，不接受完整等式。即使忽略等号，这个表达式本身也不等于 24。

暴露问题：prompt 和 reward 没有让模型稳定学会“answer 里只写表达式，不写等号和解释”。

### 4.2 使用中文乘号

题目：`nums=[2, 2, 5, 6]`

模型输出：

```text
<answer>(2 × 2 + 2) × 6</answer>
```

裁判结果：`illegal_character`

原因：中文乘号 `×` 不在允许字符集内，裁判只接受 ASCII `*`。

暴露问题：模型倾向使用自然语言中的数学符号，而不是程序裁判要求的严格表达式语法。

### 4.3 凭空引入数字

题目：`nums=[1, 6, 6, 12]`

模型输出：

```text
<answer>4*6 / (1*6) + 1</answer>
```

裁判结果：`number_mismatch`

原因：题目没有数字 `4`，并且给定数字 `12` 没有被使用。

暴露问题：模型没有稳定遵守“每个给定数字必须且只能使用一次”的核心约束。

### 4.4 可解题误报不可解

题目：`nums=[2, 2, 5, 8]`

模型输出：

```text
<answer>UNsolvable</answer>
```

裁判结果：`false_unsolvable_claim`

原因：训练集中的题目都是可解题，模型却输出了不可解声明。

暴露问题：模型在没有找到解时会倾向放弃，且对 `UNSOLVABLE` 的使用场景没有学清楚。

### 4.5 表达式合法但值不对

题目：`nums=[2, 5, 5, 11]`

模型输出：

```text
<answer>(5*5-11)/2</answer>
```

裁判结果：`wrong_value`，计算值为 `7.0`

原因：表达式合法，也使用了给定数字，但结果不是 24。

暴露问题：在格式和数字使用都正确时，模型仍缺少可靠的算术搜索能力。

## 5. 报告结论建议

建议报告采用如下表述：

> 在当前低显存 GRPO 设置下，模型较快学会了 R1 风格标签格式，但 correctness reward 没有形成稳定上升趋势。训练末尾的错误主要集中在 `<answer>` 的严格语法约束和数字使用约束上，而不是单纯的除零或表达式解析失败。因此，本次实验更适合作为 RLVR 训练不稳定性的案例分析，而不是作为性能提升结果。

不建议写成：

> GRPO 显著提升了模型 24 点求解能力。

因为当前训练指标不支持这个结论。

## 6. 后续方向

后续如果继续优化模型，建议优先级如下：

1. 先做正式评估：分别评估 base model 和当前 LoRA 在 hard、low solved-rate、unsolvable split 上的表现，确认当前 LoRA 是否优于 base。
2. 强化 answer 约束：prompt 中更明确要求 `<answer>` 只包含表达式，不写等号、不写解释、不用中文数学符号。
3. 调整 reward shaping：对接近合法表达式的情况给更细的反馈，例如区分等号错误、中文符号错误、数字不匹配、算错值。
4. 增大采样信号：当前 `num_generations=2` 对 GRPO 来说信号较弱，显存允许时可尝试 `num_generations=4`。
5. 观察 `train_log.txt` 而不只看曲线：如果格式率继续上升但正确率不动，说明需要先处理输出约束，而不是单纯延长训练。

## 7. 报告引用材料位置

- 训练曲线：`outputs/runs/grpo_qwen25_1_5b_lora8_g2/accuracy_curve.png`
- 训练指标：`outputs/runs/grpo_qwen25_1_5b_lora8_g2/training_metrics.csv`
- 定性样例：`outputs/runs/grpo_qwen25_1_5b_lora8_g2/train_log.txt`
- 最终 LoRA：`outputs/final_model/`
- 中间 checkpoint：`outputs/checkpoints/`

## 8. protocol_reward_lora8_g2 后续发现

`protocol_reward_lora8_g2` 使用了更强的 answer 协议 prompt 和 reward shaping。该 run 说明协议修复方向有效，但暴露了新的 prompt 示例泄漏问题：

- 最后 100 个 reward step 的平均 `format_rate` 提升到约 `92.25%`。
- 最后 100 个 reward step 的平均 `batch_accuracy` 仍只有约 `0.125%`。
- `illegal_character_count` 相比旧 run 下降，但 `number_mismatch_count` 成为主要错误。
- 最后 800 个答案中，`8/(3-8/3)` 出现 260 次，单独输出 `24` 出现 96 次。

原因是 prompt 中的具体正例 `GOOD: <answer>8/(3-8/3)</answer>` 被模型当作通用答案模板复制。由于大多数题目的数字不是 `[3, 3, 8, 8]`，复制该表达式会触发 `number_mismatch`。

因此后续修复策略调整为：

- 删除可复制的具体 GOOD 表达式，只保留规则描述。
- 加重 `number_mismatch` 惩罚。
- 单独统计 `bare_target_count` 和 `copied_prompt_example_count`。
- 先跑 `no_example_leak_lora8_g2`，确认模板背诵和裸目标值输出下降后，再考虑 `num_generations=4`。

## 9. no_example_leak_lora8_g2 中期结论

`no_example_leak_lora8_g2` 跑到约 step 500 时，模板背诵问题已经明显修复，但训练仍卡在 answer 协议错误：

- `copied_prompt_example_count=0`，说明 `8/(3-8/3)` 泄漏已经压住。
- 最近 100 step 的 `format_rate` 约 `90%`。
- 最近 100 step 的 `batch_accuracy` 约 `0.125%`。
- 最近 100 step 中 `illegal_character_count=644`，`equal_sign_count=511`，`unicode_operator_count=335`。
- `number_mismatch_count` 明显下降，但 `=24`、`×/÷`、全角括号和多个 `<answer>` 成为主要问题。

因此后续策略改为进一步强化“单一 ASCII answer”：

- prompt 中用更硬的英文协议要求 exactly one `<answer>`。
- `<answer>` 内只允许 ASCII 数字、空格和 `+ - * / ( )`。
- reward 中对等号、Unicode 运算符、全角括号、文本 answer、多 answer 给强惩罚。
- 新增 `fullwidth_paren_count` 和 `multiple_answer_count`，下一轮 `strict_ascii_answer_lora8_g2` 重点观察这些指标。
