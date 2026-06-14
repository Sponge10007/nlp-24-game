# Qwen-24Game-GRPO: 基于强化学习的 24 点推理模型

本项目使用 `Qwen/Qwen2.5-1.5B-Instruct` 作为小规模开源基座模型，通过 TRL `GRPOTrainer` 做可验证奖励强化学习（RLVR），让模型以 R1 风格输出：

```text
<think>...</think>
<answer>...</answer>
```

目标是给定 4 个 1-13 的整数，输出一个只使用这些数字一次、由 `+ - * / ( )` 组成且结果等于 24 的算式；不可解样本要求输出 `UNSOLVABLE`。代码也预留了 3-4 数字任意目标值的 OOD 扩展入口。

## 项目结构

```text
nlp-24-game/
├── data/
│   └── prepare_data.py          # 生成训练集、多个测试 split、不可解 holdout 和摘要
├── src/
│   ├── game24.py                # 统一裁判：答案提取、安全求值、错误类型分类
│   ├── prompts.py               # system prompt 和用户题目模板
│   └── rewards.py               # GRPO 奖励函数、训练指标和样例日志
├── tests/
│   └── test_game24.py           # 裁判单元测试
├── train.py                     # GRPO 训练入口
├── evaluate.py                  # base/LoRA 评估，支持 pass@k 和结果导出
├── play_24.py                   # 交互式演示脚本
├── plot_curve.py                # 训练曲线与错误类型图
├── rewards_inform.md            # 奖励函数说明
└── requirements.txt
```

## 环境准备

推荐 Linux/WSL2 + CUDA。6GB 显存可使用默认 4-bit + LoRA 配置；更高显存可增大 `--num-generations`、`--max-completion-length` 和 `--lora-rank`。

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

国内网络可设置 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 数据准备

```bash
python data/prepare_data.py
```

默认生成：

```text
data/train.jsonl                  # nlile/24-game 中 的训练集减去所有test的部分
data/unsolvable_test.jsonl         # 枚举出并验证的不可解的数据集
data/test_all_nonoverlap.jsonl     # 硬 split 和最低 solved-rate 的并集
data/test_hard_900_1000.jsonl      # Tree of Thoughts 论文 hard split 去重后样本
data/test_low_solved_rate.jsonl    # solved_rate 最低的一批样本
data/test.jsonl                    # hard split 的兼容别名
data/dataset_summary.json          # 数据数量和重叠剔除摘要
```

可选加分项数据：

```bash
python data/prepare_data.py --with-countdown --countdown-size 200
```

会尝试生成 `data/countdown_ood.jsonl`，用于 3-4 数字任意目标值 OOD 验证。

## 训练

低显存默认配置：

```bash
python train.py
```

常用可调参数：

```bash
python train.py \
  --run-name lora16_g4_len768 \
  --num-generations 4 \
  --max-completion-length 768 \
  --lora-rank 16
```

输出：

```text
outputs/final_model/                         # LoRA 权重
outputs/checkpoints/                         # 训练 checkpoint
outputs/runs/<run_name>/config.json          # 本次训练配置
outputs/runs/<run_name>/training_metrics.csv # accuracy/reward/format/error 曲线数据
outputs/runs/<run_name>/train_log.txt        # 每轮样例输出与判定
```

## 绘制训练曲线

```bash
python plot_curve.py \
  --metrics outputs/runs/grpo_qwen25_1_5b_lora8_g2/training_metrics.csv \
  --output outputs/runs/grpo_qwen25_1_5b_lora8_g2/accuracy_curve.png
```

图中包含 batch accuracy、滑动正确率、平均 correctness reward、格式率和主要错误类型计数。

## 评估

评估训练后的 LoRA：

```bash
python evaluate.py \
  --test-data-path data/test_hard_900_1000.jsonl \
  --test-data-path data/test_low_solved_rate.jsonl \
  --test-data-path data/unsolvable_test.jsonl \
  --output-jsonl outputs/eval_results.jsonl \
  --summary-json outputs/eval_summary.json
```

评估 base model 作为 zero-shot baseline：

```bash
python evaluate.py --base-only --test-data-path data/test_hard_900_1000.jsonl
```

评估 pass@k：

```bash
python evaluate.py --pass-k 4 --temperature 0.7 --test-data-path data/test_hard_900_1000.jsonl
```

指标包括 first@1、pass@k、格式正确率、正确数、不可解 honest rate、fabrication/error 类型分布，并保存逐题回复，方便报告做定性分析。

## 交互式展示

```bash
python play_24.py 3 3 8 8
```

或进入交互模式：

```bash
python play_24.py
```

任意目标值扩展示例：

```bash
python play_24.py 2 3 7 --target 17
```

## 报告建议

报告里建议至少包含：

1. 任务定义和 RLVR/GRPO 背景。
2. 数据构造：训练集、hard OOD、low solved-rate、不可解 holdout 的数量和去重策略。
3. 奖励函数：格式奖励、可验证 correctness reward、不可解诚实性奖励和错误分类。
4. 实验设置：base zero-shot、GRPO LoRA、可选超参数消融。
5. 定量结果：solved rate、format rate、honest/fabrication rate、pass@k。
6. 定性分析：成功样例、数字偷换/格式错误/算错值/不可解胡编样例。
7. 局限性：RL 训练不稳定、低显存下 `num_generations=2` 的限制、复杂搜索题仍依赖采样。

## DeepSeek API 拒绝采样、SFT 与 GRPO 训练

如果直接运行 `python train.py`，就是从 base model 直接做 GRPO，可作为 zero-RL baseline。但当前实验中直接 GRPO 容易长期拿不到正确性奖励，因此推荐主流程改为：先调用 DeepSeek API 做拒绝采样生成 SFT 数据，再进行 SFT 训练，最后从 SFT adapter 继续 GRPO。

GRPO 阶段的 correctness reward 也加入了距离惩罚：当模型输出的表达式合法、数字匹配、可计算但值不等于目标值时，会按 `abs(value - target)` 给平滑负奖励；离目标越近惩罚越轻，正确答案仍保持最高奖励。

### 新增脚本与模块

- `generate_rejection_sft.py`：调用 DeepSeek API 对题目并行拒绝采样，并用本地裁判函数筛选可用于 SFT 的正确 completion；默认处理前 500 题、8 个并行 worker。
- `train_sft.py`：读取 `data/rejection_sft_train.jsonl`，使用 4-bit + LoRA 做低显存 SFT 训练，默认输出到 `outputs/sft_model`。
- `src/rejection_sampling.py`：拒绝采样核心逻辑，包括 DeepSeek 响应转 `<think>/<answer>`、候选 completion 校验、选择首个通过样本、构造 SFT 数据行。

程序生成 warmup 和 solver fallback 已移除。预热数据完全依赖 DeepSeek API 拒绝采样；如果某道题多次采样仍未通过裁判，该题会被跳过，不再用程序解兜底。

### API 配置

DeepSeek API 使用 OpenAI-compatible 调用方式。运行前需要设置 API key：

```bash
export DEEPSEEK_API_KEY=你的_api_key
```

默认配置如下：

```text
base_url: https://api.deepseek.com
api_model: deepseek-reasoner
api_key_env: DEEPSEEK_API_KEY
```

`deepseek-reasoner` 会返回 `reasoning_content` 和 `content`。脚本会把它们整理为 SFT completion：

```text
<think>reasoning_content</think>
<answer>content</answer>
```

如果 `content` 已经包含 `<answer>...</answer>`，则直接作为候选 completion 使用。无论哪种情况，最终都会经过 `extract_answer` 和 `judge_answer` 校验。

### 拒绝采样机制

拒绝采样流程如下：

1. 对每道 24 点题目调用 DeepSeek API 多次，最多 `--attempts-per-case` 次。
2. 将 API 返回内容整理为 `<think>...</think>` 和 `<answer>...</answer>`。
3. 使用 `extract_answer` 提取 `<answer>` 中的最终答案。
4. 使用 `judge_answer` 校验数字使用、字符合法性、表达式值和不可解声明。
5. 只保留第一个通过裁判的 completion，写入 SFT 数据。
6. 如果所有采样都失败，跳过该题，并在 summary 中统计为 `rejected_all`。
7. 默认只处理前 500 题以控制 API 成本；这个限制只影响拒绝采样脚本，不会截断 `data/train.jsonl`。

生成的 SFT 数据写入 `data/rejection_sft_train.jsonl`，该文件被 `.gitignore` 忽略，需要在本地生成。

### 推荐训练流程：DeepSeek 拒绝采样 SFT -> GRPO

第一步，准备基础数据：

```bash
python data/prepare_data.py --with-countdown --countdown-size 200
```

第二步，生成 DeepSeek 拒绝采样 SFT 数据：

```bash
python generate_rejection_sft.py \
  --input-data-path data/train.jsonl \
  --output-data-path data/rejection_sft_train.jsonl \
  --api-model deepseek-reasoner \
  --attempts-per-case 16 \
  --max-tokens 2048 \
  --limit 500 \
  --max-workers 8
```

可选参数：

```text
--api-key-env DEEPSEEK_API_KEY      API key 所在环境变量
--base-url https://api.deepseek.com DeepSeek API 地址
--sleep-seconds 0.2                 每次 API 调用后的等待时间
--max-workers 8                     并行处理题目的 worker 数
--temperature 0.6                   采样温度
--top-p 0.95                        nucleus sampling 参数
--limit N                           只处理前 N 条样本，默认 500，用于控制 API 成本
```

第三步，执行 SFT 训练：

```bash
python train_sft.py \
  --train-data-path data/rejection_sft_train.jsonl \
  --run-name deepseek_rejection_sft_lora8
```

第四步，从 SFT adapter 继续 GRPO：

```bash
python train.py \
  --adapter-init-path outputs/sft_model \
  --run-name deepseek_sft_then_grpo_lora8_g2
```

第五步，评估最终 LoRA：

```bash
python evaluate.py \
  --adapter-path outputs/final_model \
  --test-data-path data/test_hard_900_1000.jsonl \
  --test-data-path data/test_low_solved_rate.jsonl \
  --test-data-path data/unsolvable_test.jsonl \
  --output-jsonl outputs/eval_results.jsonl \
  --summary-json outputs/eval_summary.json
```

### SFT 数据格式

生成的每行 JSONL 保持如下结构：

```json
{
  "target_nums": [3, 3, 8, 8],
  "target_value": 24,
  "solvable": true,
  "prompt": ["..."],
  "completion": "<think>...</think>\n<answer>8/(3-(8/3))</answer>",
  "answer": "8/(3-(8/3))",
  "solution": "8/(3-(8/3))",
  "rejection_source": "deepseek_api",
  "accepted_attempt": 3,
  "judge_code": "correct"
}
```

`<answer>` 中只能包含最终表达式或 `UNSOLVABLE`，不能包含 `=24`、中文数学符号、解释文字或其他多余内容。未通过这些约束的 API 输出会被拒绝。
