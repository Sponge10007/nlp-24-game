# Countdown-GRPO Arithmetic Solver

本项目使用 `Qwen/Qwen2.5-1.5B-Instruct` 作为基座模型，通过 TRL `GRPOTrainer` 做可验证奖励强化学习（RLVR）。默认任务严格跟随 TinyZero Countdown：给定 3-4 个数字和任意目标值，模型需要输出一个只使用这些数字一次、由 `+ - * / ( )` 组成且结果等于目标值的表达式。

模型按 R1 风格回答：

```text
<think>...</think>
<answer>...</answer>
```

## 项目结构

```text
data/prepare_data.py     # 生成 TinyZero Countdown train/test JSONL，保留 legacy 24 点模式
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

推荐 Linux/WSL2 + CUDA。当前默认配置面向 24GB 显存：bf16 全精度加载基座模型、高配 LoRA、`num_generations=8`、`max_completion_length=768`。

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

默认生成 TinyZero Countdown 原版顺序切分：

```bash
python data/prepare_data.py --task countdown
```

默认输出：

```text
data/train.jsonl           # TinyZero Countdown 前 327680 条训练样本，均为 solvable=true
data/test.jsonl            # TinyZero Countdown 随后 1024 条测试样本
data/unsolvable_test.jsonl # 本地不可解 holdout，仅用于额外鲁棒性评估
data/dataset_summary.json  # 数据数量、来源和 split 摘要
```

小规模 smoke test：

```bash
python data/prepare_data.py \
  --task countdown \
  --train-size 20 \
  --test-size 5 \
  --unsolvable-size 3
```

可选非 TinyZero 原版不可解训练混入：

```bash
python data/prepare_data.py \
  --task countdown \
  --include-unsolvable-train \
  --unsolvable-train-ratio 0.1
```

保留旧 24 点数据模式：

```bash
python data/prepare_data.py --task game24 --with-countdown --countdown-size 200
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

## 评估

默认评估只读取 TinyZero Countdown test split：

```bash
python evaluate.py \
  --output-jsonl outputs/eval_results.jsonl \
  --summary-json outputs/eval_summary.json
```

额外评估本地不可解 holdout：

```bash
python evaluate.py --test-data-path data/unsolvable_test.jsonl
```

评估 base model 作为 zero-shot baseline：

```bash
python evaluate.py --base-only --test-data-path data/test.jsonl
```

评估 pass@k：

```bash
python evaluate.py --pass-k 4 --temperature 0.7 --test-data-path data/test.jsonl
```

## 交互式演示

默认 target 为 24，任意目标值可用 `--target` 指定。

```bash
python play_24.py 3 3 8 8
python play_24.py 2 3 7 --target 17
```

进入交互模式：

```bash
python play_24.py --target 42
```

## 报告建议

报告里建议至少包含：Countdown 任务定义、TinyZero/RLVR 背景、数据构造、奖励函数、实验设置、定量结果、成功与失败样例、局限性和后续改进。
