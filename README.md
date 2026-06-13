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
data/train.jsonl                  # nlile/24-game 中 solvable=True 的训练集
data/unsolvable_test.jsonl         # nlile/24-game 中 solvable=False 的不可解 holdout
data/test_all_nonoverlap.jsonl     # test-time-compute/game-of-24 非训练重叠全集
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
