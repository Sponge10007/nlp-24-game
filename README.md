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
scripts/run_sft_grpo.sh     # 按实验版本隔离执行 SFT -> GRPO
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
  --limit 500 \
  --attempts-per-case 16 \
  --max-workers 8
```

脚本会调用 `deepseek-reasoner`，将 `reasoning_content` 和 `content` 整理成 `<think>...</think><answer>...</answer>` completion，并用本地裁判只保留第一个正确样本。生成的 `data/rejection_sft_train.jsonl` 被 `.gitignore` 忽略，需要在本地生成。

默认启用断点续生成：如果输出文件已经存在，会重新验证已有样本，只请求缺失或无效的题目。需要从头生成时使用 `--no-resume`。旧参数名 `--max-cases`、`--samples-per-case`、`--num-workers` 仍可作为兼容别名使用。

### 生成短算式推理 SFT 数据

不要直接用 DeepSeek 的长篇原始 reasoning 训练。将已经严格验证的答案转换为短算式轨迹：

```bash
python data/prepare_compact_sft_data.py \
  --input-path outputs/datasets/deepseek500_v1/train.jsonl \
  --output-path outputs/datasets/deepseek500_compact_v2/train.jsonl
```

例如：

```text
<think>1+1=2; 2+1=3; 8*3=24</think>
<answer>8*(1+1+1)</answer>
```

转换过程不会调用 API，并会再次使用本地裁判验证每个答案。输出使用 TRL conversational prompt-completion 格式，训练时只对 completion 计算 loss。

## SFT 预热训练

默认 24GB 高配 LoRA：

```bash
python train_sft.py \
  --train-data-path outputs/datasets/deepseek500_compact_v2/train.jsonl \
  --run-name compact_sft_lora32_len512
```

默认关键参数：

```text
load_in_4bit=False
bf16=True
lora_rank=32
lora_alpha=64
max_seq_length=512
completion_only_loss=True
learning_rate=5e-5
num_train_epochs=3
per_device_batch_size=2
grad_accum_steps=4
gradient_checkpointing=False
optim=adamw_torch
```

## v4：多解 SFT + 不可解训练

在 v3 已经稳定掌握格式后，可生成覆盖全部可解训练题、每题最多 3 个确定性正确表达式，并加入与不可解测试集严格隔离的 300 道不可解训练题：

```bash
python data/prepare_v4_training_data.py \
  --train-path data/train.jsonl \
  --solvable-test-path data/test_all_nonoverlap.jsonl \
  --unsolvable-test-path data/unsolvable_test.jsonl \
  --output-dir outputs/datasets/v4_multi_solution_unsolvable \
  --max-solutions 3 \
  --unsolvable-train-size 300 \
  --seed 20260622
```

输出：

```text
outputs/datasets/v4_multi_solution_unsolvable/
├── sft_train.jsonl
├── grpo_train.jsonl
├── summary.json
└── sha256.txt
```

生成器会自动验证：

- 所有可解 SFT 表达式都通过 `judge_answer()`；
- 可解训练题与 hard/low 测试题零重叠；
- 不可解训练题与 `unsolvable_test.jsonl` 零重叠；
- SFT 使用短算式轨迹和 conversational completion；
- GRPO 每道题只保留一行，包含可解与不可解题。

运行 v4 SFT：

```bash
export MODEL_NAME=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
export SFT_DATA_PATH=outputs/datasets/v4_multi_solution_unsolvable/sft_train.jsonl
export TRAIN_DATA_PATH=outputs/datasets/v4_multi_solution_unsolvable/grpo_train.jsonl
export SFT_NUM_TRAIN_EPOCHS=2
export SFT_LEARNING_RATE=2e-5
export SFT_MAX_SEQ_LENGTH=512

bash scripts/run_sft_grpo.sh v4_multi_solution_unsolvable sft
```

评估 v4 SFT 后，再运行 GRPO：

```bash
export GRPO_NUM_TRAIN_EPOCHS=4
export GRPO_LEARNING_RATE=3e-6
export GRPO_NUM_GENERATIONS=8
export GRPO_MAX_COMPLETION_LENGTH=512

bash scripts/run_sft_grpo.sh v4_multi_solution_unsolvable grpo
```

上述环境变量会被写入对应的 `sft_config.json` 和 `config.json`，结果统一保存在：

```text
outputs/experiments/v4_multi_solution_unsolvable/
```

## v4.1：单一规范答案 SFT

如果多解 SFT 导致同一 prompt 的不同答案互相干扰，使用 v4.1：

- 优先复用 v3 的 500 条 DeepSeek 精简正确答案；
- 其余训练题由确定性搜索补齐；
- 每道可解题只保留一个规范答案；
- SFT 仅加入 150 道不可解题；
- GRPO 仍使用 300 道不可解题；
- v3、v4 和 v4.1 使用独立目录，互不覆盖。

生成数据：

```bash
python data/prepare_v41_training_data.py \
  --train-path data/train.jsonl \
  --teacher-sft-path outputs/datasets/deepseek500_compact_v2/train.jsonl \
  --solvable-test-path data/test_all_nonoverlap.jsonl \
  --unsolvable-test-path data/unsolvable_test.jsonl \
  --output-dir outputs/datasets/v4_1_canonical_solution \
  --unsolvable-sft-size 150 \
  --unsolvable-grpo-size 300 \
  --search-candidates 16 \
  --seed 20260622
```

检查 `summary.json` 中：

```text
solvable_train_puzzles=1162
unique_solvable_sft_puzzles=1162
teacher_solution_rows=500
deterministic_solution_rows=662
missing_solution_rows=0
unsolvable_sft_puzzles=150
unsolvable_grpo_puzzles=300
solvable_test_overlap=0
unsolvable_test_overlap=0
sft_rows=1312
grpo_rows=1462
```

训练 v4.1 SFT：

```bash
export MODEL_NAME=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
export SFT_DATA_PATH=outputs/datasets/v4_1_canonical_solution/sft_train.jsonl
export TRAIN_DATA_PATH=outputs/datasets/v4_1_canonical_solution/grpo_train.jsonl
export SFT_NUM_TRAIN_EPOCHS=1
export SFT_LEARNING_RATE=1e-5
export SFT_MAX_SEQ_LENGTH=512

bash scripts/run_sft_grpo.sh v4_1_canonical_solution sft
```

确认 SFT 评估没有相对 v3 明显退化后再运行 GRPO：

```bash
export GRPO_NUM_TRAIN_EPOCHS=4
export GRPO_LEARNING_RATE=3e-6
export GRPO_NUM_GENERATIONS=8
export GRPO_MAX_COMPLETION_LENGTH=512

bash scripts/run_sft_grpo.sh v4_1_canonical_solution grpo
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

SFT 和 GRPO 都支持从 checkpoint 恢复：

```bash
python train_sft.py ... --resume-from-checkpoint latest
python train.py ... --resume-from-checkpoint latest --no-reset-metrics
```

也可以把 `latest` 换成具体的 `checkpoint-xxx` 路径。

## 实验版本管理

推荐使用统一脚本，避免不同训练版本覆盖：

```bash
export MODEL_NAME=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct

# 一次执行完整 SFT -> GRPO
bash scripts/run_sft_grpo.sh v3_compact_sft_grpo all

# 或分阶段执行
bash scripts/run_sft_grpo.sh v3_compact_sft_grpo sft
bash scripts/run_sft_grpo.sh v3_compact_sft_grpo grpo
```

每个实验使用一个不可重复的名称。若目标 adapter 已存在，脚本会拒绝覆盖；应优先使用新版本名。确实需要覆盖时显式设置 `ALLOW_OVERWRITE=1`。

版本目录结构：

```text
outputs/experiments/<experiment>/
├── code_commit.txt
├── code_branch.txt
├── code_status.txt
├── dataset_summary.json
├── sft/
│   ├── adapter/
│   ├── checkpoints/
│   └── sft_config.json
└── grpo/
    ├── adapter/
    ├── checkpoints/
    ├── config.json
    ├── training_metrics.csv
    └── train_log.txt
```

## 绘制训练曲线

```bash
python plot_curve.py \
  --metrics outputs/experiments/v3_compact_sft_grpo/grpo/training_metrics.csv \
  --output outputs/experiments/v3_compact_sft_grpo/grpo/accuracy_curve.png
```

## 评估

评估训练后的 LoRA：

```bash
python evaluate.py \
  --adapter-path outputs/experiments/v3_compact_sft_grpo/grpo/adapter \
  --test-data-path data/test_hard_900_1000.jsonl \
  --test-data-path data/test_low_solved_rate.jsonl \
  --test-data-path data/unsolvable_test.jsonl \
  --output-jsonl outputs/experiments/v3_compact_sft_grpo/grpo/eval_results.jsonl \
  --summary-json outputs/experiments/v3_compact_sft_grpo/grpo/eval_summary.json
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
