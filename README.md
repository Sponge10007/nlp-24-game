# Countdown-GRPO Arithmetic Solver

本项目使用 `Qwen/Qwen2.5-1.5B-Instruct` 作为基座模型，通过 TRL `GRPOTrainer` 做可验证奖励强化学习（RLVR）。默认任务参考 TinyZero 的 Countdown 方向：给定 3-4 个数字和任意目标值，模型需要输出一个只使用这些数字一次、由 `+ - * / ( )` 组成且结果等于目标值的表达式。

模型按 R1 风格回答：

```text
<think>...</think>
<answer>...</answer>
```

不可解样本要求输出 `UNSOLVABLE`。训练数据默认混入少量本地生成的不可解题，避免模型只学可解表达式。

## 项目结构

```text
data/prepare_data.py     # 生成 Countdown 训练/测试集，保留 legacy 24 点数据模式
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

默认生成 TinyZero Countdown 风格数据：

```bash
python data/prepare_data.py
```

默认行为：

```text
data/train.jsonl          # Countdown 可解题 + 少量本地不可解题
data/test.jsonl           # Countdown held-out 测试题
data/unsolvable_test.jsonl # 本地不可解 holdout
data/dataset_summary.json # 数据数量、来源和 split 摘要
```

常用参数：

```bash
python data/prepare_data.py \
  --task countdown \
  --countdown-size 2000 \
  --test-size 200 \
  --unsolvable-train-ratio 0.1 \
  --unsolvable-size 100
```

`--countdown-size -1` 表示读取全部 `Jiayi-Pan/Countdown-Tasks-3to4` 数据。`--unsolvable-train-ratio 0.1` 表示按可解训练样本数量的约 10% 混入不可解训练题。

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

输出：

```text
outputs/final_model/                         # LoRA 权重
outputs/checkpoints/                         # 训练 checkpoint
outputs/runs/<run_name>/config.json          # 本次训练配置
outputs/runs/<run_name>/training_metrics.csv # accuracy/reward/format/error 曲线数据
outputs/runs/<run_name>/train_log.txt        # 样例输出与判定
```

## 评估

评估脚本默认读取 `data/test.jsonl` 和 `data/unsolvable_test.jsonl`，使用 bf16 非 4-bit 加载，并默认生成最多 768 tokens。

```bash
python evaluate.py \
  --output-jsonl outputs/eval_results.jsonl \
  --summary-json outputs/eval_summary.json
```

评估 base model 作为 zero-shot baseline：

```bash
python evaluate.py --base-only --test-data-path data/test.jsonl
```

评估 pass@k：

```bash
python evaluate.py --pass-k 4 --temperature 0.7 --test-data-path data/test.jsonl
```

低显存评估可手动加回 4-bit：

```bash
python evaluate.py --load-in-4bit --test-data-path data/test.jsonl
```

## 交互式演示

默认 target 仍为 24，任意目标值可用 `--target` 指定。

```bash
python play_24.py 3 3 8 8
python play_24.py 2 3 7 --target 17
```

进入交互模式：

```bash
python play_24.py --target 42
```

## 报告建议

报告里建议至少包含：Countdown 任务定义、TinyZero/RLVR 背景、数据构造、不可解题比例、奖励函数、实验设置、定量结果、成功与失败样例、局限性和后续改进。
