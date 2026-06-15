# Qwen-24Game-GRPO

本项目使用 `Qwen/Qwen2.5-1.5B-Instruct` 作为基座模型，通过 TRL `GRPOTrainer` 做可验证奖励强化学习（RLVR），训练模型用 R1 风格回答 24 点题目：

```text
<think>...</think>
<answer>...</answer>
```

目标是给定 3-4 个整数和目标值，输出一个只使用这些数字一次、由 `+ - * / ( )` 组成且结果等于目标值的表达式；不可解样本要求输出 `UNSOLVABLE`。

## 项目结构

```text
data/prepare_data.py     # 生成训练集、测试 split、不可解 holdout 和摘要
src/game24.py            # 答案提取、判题、安全求值和错误分类
src/prompts.py           # system prompt 和用户题目模板
src/rewards.py           # GRPO 奖励函数、训练指标和样例日志
tests/                   # 单元测试
train.py                 # GRPO 训练入口
evaluate.py              # base/LoRA 评估，支持 pass@k 和结果导出
play_24.py               # 交互式演示脚本
plot_curve.py            # 训练曲线与错误类型图
rewards_inform.md        # 奖励函数说明
```

## 环境准备

推荐 Linux/WSL2 + CUDA。当前默认配置面向 24GB 显存：bf16 全精度加载基座模型、高配 LoRA、`num_generations=8`、`max_completion_length=768`。如果显存不足，可以使用下方低显存回退命令。

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
data/train.jsonl
data/unsolvable_test.jsonl
data/test_all_nonoverlap.jsonl
data/test_hard_900_1000.jsonl
data/test_low_solved_rate.jsonl
data/test.jsonl
data/dataset_summary.json
```

可选生成 Countdown OOD 数据：

```bash
python data/prepare_data.py --with-countdown --countdown-size 200
```

## 训练

24GB 默认高配 LoRA：

```bash
python train.py
```

默认关键参数：

```text
load_in_4bit=False
bf16=True
lora_rank=32
lora_alpha=64
num_generations=8
max_prompt_length=256
max_completion_length=768
per_device_batch_size=2
grad_accum_steps=4
gradient_checkpointing=False
optim=adamw_torch
```

低显存回退配置：

```bash
python train.py \
  --load-in-4bit \
  --lora-rank 8 \
  --lora-alpha 16 \
  --num-generations 2 \
  --max-prompt-length 128 \
  --max-completion-length 384 \
  --per-device-batch-size 1 \
  --grad-accum-steps 8 \
  --gradient-checkpointing \
  --optim paged_adamw_8bit
```

常用自定义示例：

```bash
python train.py \
  --run-name lora32_g8_len768 \
  --num-generations 8 \
  --max-completion-length 768 \
  --lora-rank 32 \
  --lora-alpha 64
```

输出：

```text
outputs/final_model/                         # LoRA 权重
outputs/checkpoints/                         # 训练 checkpoint
outputs/runs/<run_name>/config.json          # 本次训练配置
outputs/runs/<run_name>/training_metrics.csv # accuracy/reward/format/error 曲线数据
outputs/runs/<run_name>/train_log.txt        # 样例输出与判定
```

## 绘制训练曲线

```bash
python plot_curve.py \
  --metrics outputs/runs/grpo_qwen25_1_5b_lora32_g8_len768/training_metrics.csv \
  --output outputs/runs/grpo_qwen25_1_5b_lora32_g8_len768/accuracy_curve.png
```

图中包含 batch accuracy、滑动正确率、平均 correctness reward、格式率和主要错误类型计数。

## 评估

评估脚本默认使用 bf16 非 4-bit 加载，并默认生成最多 768 tokens。

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

低显存评估可手动加回 4-bit：

```bash
python evaluate.py --load-in-4bit --test-data-path data/test_hard_900_1000.jsonl
```

## 交互式演示

交互脚本默认使用 bf16 非 4-bit 加载；低显存时可加 `--load-in-4bit`。

```bash
python play_24.py 3 3 8 8
```

进入交互模式：

```bash
python play_24.py
```

任意目标值示例：

```bash
python play_24.py 2 3 7 --target 17
```

## 报告建议

报告里建议至少包含：任务定义与 RLVR/GRPO 背景、数据构造、奖励函数、实验设置、定量结果、成功与失败样例、局限性和后续改进。
