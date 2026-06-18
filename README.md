# Qwen-24Game-GRPO

本项目使用 `Qwen/Qwen2.5-1.5B-Instruct` 作为基座模型，通过 DeepSeek API 拒绝采样生成 SFT 预热数据，再用 TRL `GRPOTrainer` 做可验证奖励强化学习（RLVR）。任务仍是 24 点：给定 4 个 1-13 的整数，输出一个只使用这些数字一次、由 `+ - * / ( )` 组成且结果等于 24 的表达式；不可解样本要求输出 `UNSOLVABLE`。

模型按 R1 风格回答：

```text
<think>...</think>
<answer>...</answer>
```

## 项目结构

```text
data/prepare_data.py        # 生成训练集、测试 split、不可解 holdout 和摘要
generate_rejection_sft.py   # DeepSeek API 拒绝采样，生成 SFT 数据
train_sft.py                # SFT 预热训练入口
train.py                    # GRPO 训练入口，可从 SFT adapter 继续
evaluate.py                 # base/LoRA 评估，支持 pass@k 和结果导出
play_24.py                  # 交互式演示脚本
src/game24.py               # 答案提取、判题、安全求值和错误分类
src/prompts.py              # system prompt 和用户题目模板
src/rewards.py              # GRPO 奖励函数、训练指标和样例日志
src/rejection_sampling.py   # 拒绝采样核心逻辑
tests/                      # 单元测试
```

## 环境准备

推荐 Linux/WSL2 + CUDA。当前默认配置面向 24GB 显存：bf16 非 4-bit 加载、高配 LoRA、较大的 GRPO 采样分支和上下文长度。低显存仍可通过命令行参数手动回退。

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

国内网络可设置 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

DeepSeek 拒绝采样需要配置：

```bash
export DEEPSEEK_API_KEY=your_api_key
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

## DeepSeek 拒绝采样 SFT 数据

生成 SFT 预热数据：

```bash
python generate_rejection_sft.py \
  --input-data-path data/train.jsonl \
  --output-data-path data/rejection_sft_train.jsonl \
  --max-cases 500 \
  --samples-per-case 4 \
  --num-workers 8
```

脚本会调用 `deepseek-reasoner`，将 `reasoning_content` 和 `content` 整理成 `<think>...</think><answer>...</answer>` completion，并用本地裁判只保留第一个正确样本。生成的 `data/rejection_sft_train.jsonl` 被 `.gitignore` 忽略，需要在本地生成。

## SFT 预热训练

默认 24GB 高配 LoRA：

```bash
python train_sft.py \
  --train-data-path data/rejection_sft_train.jsonl \
  --run-name rejection_sft_lora32_len768
```

默认关键参数：

```text
load_in_4bit=False
bf16=True
lora_rank=32
lora_alpha=64
max_seq_length=768
per_device_batch_size=2
grad_accum_steps=4
gradient_checkpointing=False
optim=adamw_torch
```

低显存 SFT 回退：

```bash
python train_sft.py \
  --train-data-path data/rejection_sft_train.jsonl \
  --load-in-4bit \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-seq-length 512 \
  --per-device-batch-size 1 \
  --grad-accum-steps 8 \
  --gradient-checkpointing \
  --optim paged_adamw_8bit
```

## GRPO 训练

从 SFT adapter 继续 GRPO：

```bash
python train.py \
  --adapter-init-path outputs/sft_model \
  --run-name deepseek_sft_then_grpo_lora32_g8_len768
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

从 base model 直接做 GRPO：

```bash
python train.py --run-name grpo_base_lora32_g8_len768
```

低显存 GRPO 回退：

```bash
python train.py \
  --adapter-init-path outputs/sft_model \
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

## 输出

```text
outputs/sft_model/                          # SFT LoRA 权重
outputs/final_model/                        # GRPO LoRA 权重
outputs/sft_checkpoints/                    # SFT checkpoint
outputs/checkpoints/                        # GRPO checkpoint
outputs/runs/<run_name>/config.json         # GRPO 配置
outputs/runs/<run_name>/sft_config.json     # SFT 配置
outputs/runs/<run_name>/training_metrics.csv
outputs/runs/<run_name>/train_log.txt
```

## 绘制训练曲线

```bash
python plot_curve.py \
  --metrics outputs/runs/deepseek_sft_then_grpo_lora32_g8_len768/training_metrics.csv \
  --output outputs/runs/deepseek_sft_then_grpo_lora32_g8_len768/accuracy_curve.png
```

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

低显存评估可手动加回 4-bit：

```bash
python evaluate.py --load-in-4bit --test-data-path data/test_hard_900_1000.jsonl
```

## 交互式演示

```bash
python play_24.py 3 3 8 8
python play_24.py
python play_24.py 2 3 7 --target 17
```

## 报告建议

报告里建议至少包含：任务定义、DeepSeek 拒绝采样 SFT、RLVR/GRPO 背景、数据构造、奖励函数、实验设置、定量结果、成功与失败样例、局限性和后续改进。
