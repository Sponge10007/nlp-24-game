# 测试集生成与后续记录

本文档记录当前 24 点项目的数据 split 修复工作，方便后续实验和报告撰写时引用。之后如果继续调整数据、评估脚本或实验设置，请把关键变更追加到本文档。

## 2026-06-13 测试集生成修复

### 问题背景

原来的 `data/prepare_data.py` 先把 `nlile/24-game` 的全部可解样本写入训练集，再从 `test-time-compute/game-of-24` 中剔除所有与训练集重叠的组合。实际检查后发现：

- 当前 Hugging Face 缓存中的 `nlile/24-game` 有 1362 行，且 `solvable=True` 为 1362 行，没有 `solvable=False` 行。
- `test-time-compute/game-of-24` 也有 1362 行，四数组合与 `nlile/24-game` 完全相同。
- 因此如果先把 `nlile/24-game` 全部放进训练集，再要求 ToT 测试集不与训练集重叠，ToT 测试集会被全部删空。
- ToT 数据集的 solved rate 字段名是 `Solved rate`，值形如 `99.20%`，旧脚本没有正确解析这个百分数字符串。

### 当前修复策略

现在的数据生成流程改为：

1. 先读取 `test-time-compute/game-of-24`。
2. 保留两类测试题：
   - 官方 hard split：ToT 数据集零基下标 `900..999`，也就是 `Rank=901..1000` 的 100 题。
   - low solved-rate split：全表 solved rate 最低的 100 题。
3. 将 hard split 和 low solved-rate split 去重合并，作为训练集需要排除的测试 key。
4. 再读取 `nlile/24-game`，剔除上述测试 key 后写入训练集。
5. 因为当前 `nlile/24-game` 没有 `solvable=False` 行，所以本地枚举 1 到 13 的四数字组合，用确定性求解器筛出不可解组合，默认随机种子 `20240613` 抽取 100 条作为不可解测试集。

### 当前数据文件含义

- `data/train.jsonl`：训练集。来自 `nlile/24-game`，剔除了 ToT hard 和 low solved-rate 测试题后剩余 1162 条。
- `data/test_hard_900_1000.jsonl`：ToT 官方 hard split，100 条，对应 ToT 数据集零基下标 `900..999`。
- `data/test_low_solved_rate.jsonl`：ToT solved rate 最低的 100 条。
- `data/test_all_nonoverlap.jsonl`：hard split 和 low solved-rate split 的去重并集，当前 200 条。该文件与训练集无重叠。
- `data/test.jsonl`：兼容别名，内容等同于 `data/test_hard_900_1000.jsonl`。它不是没用；默认评估脚本 `evaluate.py` 在未显式指定 `--test-data-path` 时会读取 `data/test.jsonl` 和 `data/unsolvable_test.jsonl`。
- `data/unsolvable_test.jsonl`：本地枚举生成的不可解 holdout，当前 100 条，全部 `solvable=false`。
- `data/countdown_ood.jsonl`：Countdown 3-4 数字任意目标值 OOD 扩展集，当前 200 条。
- `data/dataset_summary.json`：本次数据生成摘要，记录各 split 数量、训练集剔除数量、不可解枚举总数等。

### 当前数量

```text
1162 data/train.jsonl
 100 data/test.jsonl
 100 data/test_hard_900_1000.jsonl
 100 data/test_low_solved_rate.jsonl
 200 data/test_all_nonoverlap.jsonl
 100 data/unsolvable_test.jsonl
 200 data/countdown_ood.jsonl
```

### 生成命令

```bash
./venv/bin/python data/prepare_data.py --with-countdown --countdown-size 200
```

可选参数：

```bash
./venv/bin/python data/prepare_data.py \
  --low-solved-rate-size 100 \
  --unsolvable-size 100 \
  --unsolvable-seed 20240613 \
  --with-countdown \
  --countdown-size 200
```

### 验证结果

已验证：

- `data/train.jsonl` 与 `data/test_hard_900_1000.jsonl` 无重叠。
- `data/train.jsonl` 与 `data/test_low_solved_rate.jsonl` 无重叠。
- `data/train.jsonl` 与 `data/test_all_nonoverlap.jsonl` 无重叠。
- `data/test_all_nonoverlap.jsonl` 等于 hard split 和 low solved-rate split 的 key 去重并集。
- `data/unsolvable_test.jsonl` 的 100 条样本都无法用四则运算凑出 24。

已运行测试：

```bash
./venv/bin/python -m unittest tests/test_game24.py tests/test_prepare_data.py
```

结果：13 个测试通过。

### 报告建议

报告中描述数据集时建议明确说明：

- `nlile/24-game` 和 `test-time-compute/game-of-24` 当前版本包含同一套 1362 个四数组合，所以不能直接用“全量 nlile 训练 + ToT 非重叠测试”的旧逻辑。
- 本项目采用先固定 ToT hard/low 测试题，再从 nlile 训练集中剔除这些组合的策略，保证评估样本没有出现在训练集中。
- `nlile/24-game` 当前版本没有不可解样本，因此不可解测试集来自本地完整枚举，而不是 HF 数据集字段。

## 后续变更记录

后续如果继续修改数据、训练或评估，请在这里追加：

- 修改日期。
- 改动文件。
- 改动目的。
- 重新生成或验证命令。
- 关键结果或风险。

# 2026-06-13 训练脚本兼容新版 TRL

- 改动文件：`train.py`。
- 问题：本地环境安装的是 `trl 1.5.1`，`GRPOConfig` 已经不再支持旧参数 `max_prompt_length`，并且旧参数 `kl_coef` 在新版中对应 `beta`。
- 处理：新增 `build_grpo_config()`，根据当前安装的 `GRPOConfig.__init__` 签名过滤不支持的参数，并在需要时把 `--kl-coef` 映射为 `beta`。
- 验证：
  - `./venv/bin/python -m py_compile train.py`
  - 用 `bf16=False` 做了轻量 `GRPOConfig` 构造验证，确认 `beta=0.05`、`max_completion_length=384`、`num_generations=2` 能正确写入配置。
- 注意：当前 Codex 工具环境无法初始化 NVML/GPU，所以 `bf16=True` 的轻量配置验证会报 “setup doesn't support bf16/gpu”。用户本机训练命令刚才已经能加载 CUDA/bitsandbytes 权重，因此实际训练环境应以用户终端为准。

# 2026-06-14 当前训练失败现象分析

- 新增文件：`training_failure_analysis.md`。
- 目的：整理 `grpo_qwen25_1_5b_lora8_g2` 这次完整 run 的失败现象，作为报告素材。
- 主要结论：模型学会了一部分 `<think>/<answer>` 外层格式，但没有稳定学会合法且正确的 24 点表达式；最后 100 个 reward step 的平均正确率约 `0.5%`，平均格式率约 `84.9%`。
- 主要错误：`illegal_character` 和 `number_mismatch`，说明最大瓶颈是 `<answer>` 严格语法和数字使用约束没有学稳。
- 未做事项：没有改训练代码，没有重跑训练。

# 2026-06-14 可验证答案协议改进

- 改动文件：`src/prompts.py`、`src/rewards.py`、`tests/test_rewards.py`。
- 目的：下一轮训练先修 `<answer>` 协议，让模型稳定输出 ASCII、无等号、只用给定数字一次的可裁判表达式。
- Prompt 改动：加入短反例和正例，明确 `<answer>` 只能写 ASCII 算式或精确 `UNSOLVABLE`，禁止 `=24`、解释文字、中文符号、近似词。
- Reward 改动：保持 `judge_answer()` 严格不变，但在训练 reward 中增加协议塑形，区分等号、Unicode 运算符、文本、数字不匹配、合法但算错等情况。
- 新增指标：`equal_sign_count`、`unicode_operator_count`、`answer_text_count`、`legal_expr_wrong_value_count`。
- 新增测试：覆盖 `=24`、`×`、数字不匹配、合法但算错值四类情况。
- 下一轮建议 run name：`protocol_reward_lora8_g2`；显存允许时可试 `protocol_reward_lora8_g4`。

# 2026-06-14 修复 GOOD 示例泄漏和裸目标值输出

- 改动文件：`src/prompts.py`、`src/rewards.py`、`tests/test_rewards.py`、`training_failure_analysis.md`。
- 问题：`protocol_reward_lora8_g2` 的格式率提升到约 `92.25%`，但最后 100 step 准确率仍约 `0.125%`；日志显示模型大量背诵 prompt 中的 `GOOD: <answer>8/(3-8/3)</answer>`，最后 800 个答案中该模板出现 260 次，单独输出 `24` 出现 96 次。
- Prompt 改动：删除可复制的 `GOOD: <answer>8/(3-8/3)</answer>`，改为规则描述，强调不得复用示例数字，必须使用本题给定数字且每个只用一次。
- Reward 改动：将普通 `number_mismatch` 惩罚从 `-0.4` 加重到 `-0.8`，对裸目标值和复制旧 GOOD 示例加重到 `-0.9`。
- 新增指标：`bare_target_count`、`copied_prompt_example_count`。
- 新增/更新测试：覆盖复制 `8/(3-8/3)`、裸 `24`、加重 number mismatch、合法但算错仍保留小正奖励。
- 下一轮建议先跑 `python train.py --run-name no_example_leak_lora8_g2`；若 number mismatch 明显下降，再跑 `--num-generations 4` 对比。

# 2026-06-14 强化单一 ASCII answer 约束

- 改动文件：`src/prompts.py`、`src/rewards.py`、`tests/test_rewards.py`、`training_failure_analysis.md`。
- 问题：`no_example_leak_lora8_g2` 到 step 500 时模板背诵已消失，但最近 100 step 仍有大量 `=`, `×/÷`, 全角括号和多个 `<answer>`；准确率仍接近 0。
- Prompt 改动：改为更硬的英文协议，明确 `<think>` 可中文，但 `<answer>` 必须 ASCII-only，且只能输出一个 `<answer>`。
- Reward 改动：对等号、Unicode 运算符、全角括号、answer 文本给 `-1.2` 强惩罚；多个 `<answer>` 也给 `-1.2`。
- 新增指标：`fullwidth_paren_count`、`multiple_answer_count`。
- 新增测试：覆盖等号强惩罚、Unicode 强惩罚、全角括号、多 answer、合法但算错保留小正奖励。
- 下一轮建议 run name：`strict_ascii_answer_lora8_g2`，到 step 300-500 检查协议错误是否下降。
