# 奖励函数说明

项目采用两类奖励函数，并在 `src/game24.py` 中使用统一裁判，保证训练奖励和最终评估口径一致。

## 1. 格式奖励 `format_reward`

模型回复必须包含：

```text
<think>...</think>
<answer>...</answer>
```

- 格式完整：`+0.5`
- 缺少任一标签：`-1.0`

## 2. 正确性奖励 `correctness_reward`

从 `<answer>` 中提取最终答案后，裁判会检查：

- 是否为空。
- 是否声明 `UNSOLVABLE`。
- 是否只包含数字、`+ - * /` 和括号。
- 是否每个给定数字必须且只能使用一次。
- 是否可以被安全 AST 求值。
- 结果是否等于目标值，默认 24，允许 `1e-6` 浮点误差。

奖励规则：

- 可解题算对：`+2.0`
- 不可解题诚实输出 `UNSOLVABLE`：`+2.0`
- 可解题误报 `UNSOLVABLE`：通常 `-0.5`
- 不可解题胡编算式：`-1.5`
- 空答案、数字不匹配、非法字符、语法错误、除零等：通常 `-0.5`
- 数字和语法合法但结果不等于目标值：`0.0`

## 3. 记录的训练指标

`training_metrics.csv` 会记录：

- `batch_accuracy`
- `smoothed_accuracy`
- `mean_correctness_reward`
- `format_rate`
- `correct_count`
- `unsolvable_honest_count`
- `false_unsolvable_claim_count`
- `missing_answer_count`
- `number_mismatch_count`
- `illegal_character_count`
- `syntax_error_count`
- `division_by_zero_count`
- `wrong_value_count`
- `fabricated_unsolvable_count`

这些指标可直接用 `plot_curve.py` 画成训练曲线，并用于报告中的错误类型分析。
