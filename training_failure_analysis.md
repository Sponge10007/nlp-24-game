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

## 10. strict_ascii_answer_lora8_g2 完整 run 结论

`strict_ascii_answer_lora8_g2` 是目前协议修复效果最好的一轮。完整 run 共 1740 个 reward step，最后 100 step 的均值和累计错误如下：

| 指标 | 数值 |
| --- | ---: |
| 平均 batch accuracy | 0.625% |
| 平均 smoothed accuracy | 0.9315% |
| 平均 format rate | 93.25% |
| 平均 correctness reward | -0.5448 |
| `correct_count` | 5 |
| `number_mismatch_count` | 358 |
| `wrong_value_count` | 272 |
| `legal_expr_wrong_value_count` | 272 |
| `illegal_character_count` | 130 |
| `equal_sign_count` | 98 |
| `fullwidth_paren_count` | 25 |
| `unicode_operator_count` | 13 |
| `bare_target_count` | 38 |
| `multiple_answer_count` | 0 |
| `copied_prompt_example_count` | 0 |

这说明 strict ASCII prompt/reward 对格式协议是有效的：模板背诵、多 answer、Unicode 运算符等问题明显下降。但准确率仍很低，主要瓶颈转为数字使用约束和算术搜索。

典型失败包括：

- 裸目标值或中间数：`<answer>24</answer>`、`<answer>12</answer>`、`<answer>6</answer>`。
- 拼接数字：例如题目 `nums=[6,9,9,12]` 时输出 `<answer>69 - 12 / 9 * 9</answer>`。
- 漏用或多用数字：例如题目 `nums=[4,11,12,13]` 时输出 `<answer>4 + 11 + 11 - 12 - 13</answer>`。
- 数字匹配但算错：例如 `<answer>12 * (7 - 2) + 10</answer>` 使用了给定数字但结果为 70。

因此下一轮不建议直接增加 `num_generations=4`。更合适的方向是先细分并强化 `number_mismatch` 的 reward shaping：

- 新增单数字答案、漏用、多用、题外数字、重复次数错误、拼接大数字等诊断指标。
- 对裸目标值、单数字答案、拼接/题外大数字给更强惩罚。
- 保留数字完全匹配但算错的 `0.1` 小正奖励，鼓励模型先进入可裁判表达式阶段。
- 下一轮 run 建议为 `number_usage_lora8_g2`；只有当数字使用错误明显下降后，再跑 `number_usage_lora8_g4 --num-generations 4` 做对比。

## 11. number_usage_lora8_g2 后的策略调整

`number_usage_lora8_g2` 完整跑到 1740 step 后，相比 `strict_ascii_answer_lora8_g2` 有小幅改善，但仍不是成功训练：

| 指标 | strict_ascii 最后 100 step | number_usage 最后 100 step |
| --- | ---: | ---: |
| `batch_accuracy` | 0.625% | 1.125% |
| `correct_count` | 5 | 9 |
| `format_rate` | 93.25% | 91.625% |
| `number_mismatch_count` | 358 | 325 |
| `wrong_value_count` | 272 | 306 |
| `illegal_character_count` | 130 | 110 |
| `bare_target_count` | 38 | 28 |

全 run 中，`number_usage_lora8_g2` 的总 correct 从 100 增至 119，`number_mismatch` 和 `illegal_character` 都下降，但 `wrong_value` 上升到 4200。这说明模型更常输出可裁判且数字匹配的表达式，但算术搜索仍不稳定。

因此下一阶段不再继续主要加重协议惩罚，而改为参考解 SFT warmup：

- 使用 `nlile/24-game` 原始 `solutions` 字段提供明确的 24 点解法模式。
- 只保留 `data/train.jsonl` 中的 1162 个训练组合，避免 hard/low 测试泄漏。
- 将 `×` 等符号规范化为 ASCII，并用当前严格裁判验证。
- 已生成 `data/sft_train.jsonl`：2655 条 SFT 样本，覆盖 1162 个训练组合。

后续实验建议顺序：

1. 跑 `python train_sft.py`，得到 `outputs/models/sft_ref_lora8`。
2. 评估 SFT adapter 在 hard、low solved-rate、unsolvable 上的 first@1/pass@4。
3. 从 SFT adapter 接续 GRPO：`python train.py --run-name sft_then_grpo_lora8_g2 --init-adapter-path outputs/models/sft_ref_lora8 --output-dir outputs/models/sft_then_grpo_lora8_g2`。
4. 如果 SFT+GRPO g2 明显优于当前 `number_usage_lora8_g2`，再跑 g4 做对比。

## 12. SFT+GRPO 测试结论

`sft_then_grpo_lora8_g2` 已完整跑到 1740 step，并保存到 `outputs/models/sft_then_grpo_lora8_g2`。这轮实验的主要结论是：SFT warmup 明显改善了格式和可裁判性，但 solved rate 仍然很低，模型主要卡在算术搜索。

训练内最后 100 step 指标如下：

| 指标 | 数值 |
| --- | ---: |
| `batch_accuracy` | 1.5% |
| `format_rate` | 100.0% |
| `correct_count` | 12 |
| `number_mismatch_count` | 201 |
| `wrong_value_count` | 561 |
| `legal_expr_wrong_value_count` | 561 |
| `illegal_character_count` | 25 |
| `missing_answer_count` | 0 |
| `bare_target_count` | 0 |
| `single_number_answer_count` | 0 |

相比 `number_usage_lora8_g2` 的全 run 结果，SFT+GRPO 有明确的协议改进：

| 指标 | `number_usage_lora8_g2` | `sft_then_grpo_lora8_g2` |
| --- | ---: | ---: |
| 总 correct | 119 | 253 |
| 最高 `smoothed_accuracy` | 1.88% | 3.12% |
| 最终 `smoothed_accuracy` | 1.12% | 1.50% |
| 总 `number_mismatch_count` | 5986 | 3696 |
| 总 `illegal_character_count` | 2583 | 382 |
| 总 `missing_answer_count` | 989 | 4 |
| 总 `bare_target_count` | 887 | 32 |
| 总 `wrong_value_count` | 4200 | 9546 |

`wrong_value_count` 大幅上升不是简单退化，而是说明模型更常产出格式合法、数字也更接近要求的表达式；但是多数表达式没有算到 24。典型错误包括：

```text
nums=[4, 5, 6, 10]
<answer>(10-6)*4+5</answer>
value=21

nums=[1, 2, 4, 7]
<answer>(7-1)*2*4</answer>
value=48
```

正式测试集结果如下：

| Split | first@1 | pass@4 | 主要错误 |
| --- | ---: | ---: | --- |
| `test_hard_900_1000` | 1/100 | 6/100 | `wrong_value` |
| `test_low_solved_rate` | 0/100 | 0/100 | `wrong_value` |
| `unsolvable_test` | 0/100 | 0/100 | `fabricated_unsolvable` |

不可解测试集完全失败：100 条不可解题都输出了表达式，全部被判为 `fabricated_unsolvable`。例如：

```text
nums=[3, 3, 3, 13]
<answer>(3+3)*13-3</answer>
```

因此报告中建议采用如下表述：

> SFT+GRPO 显著改善了输出格式和可裁判性，使模型更稳定地产生 `<answer>` 表达式，并大幅降低非法字符、缺失答案和裸目标值问题。但 held-out 测试 solved rate 仍然很低，主要错误从输出协议错误转为算术搜索错误；模型尚未稳定学会求解 24 点。不可解拒答能力也没有形成，所有不可解样本均被胡编为表达式。

不建议写成：

> SFT+GRPO 显著提升了 24 点求解能力。

因为当前 first@1/pass@4 测试结果不支持这个结论。更准确的结论是：当前模型已从“不可裁判输出”推进到“可裁判但算术不可靠”的阶段，后续应考虑 verifier-guided decoding、更多候选采样、显式搜索或不可解拒答训练。
