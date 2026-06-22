#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="${1:?Usage: bash scripts/run_sft_grpo.sh <experiment-name> [sft|grpo|all]}"
STAGE="${2:-all}"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
SFT_DATA_PATH="${SFT_DATA_PATH:-outputs/datasets/deepseek500_compact_v2/train.jsonl}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-data/train.jsonl}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/experiments}"
ROOT="${EXPERIMENT_ROOT}/${EXPERIMENT_NAME}"
GRPO_INIT_ADAPTER_PATH="${GRPO_INIT_ADAPTER_PATH:-$ROOT/sft/adapter}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-3}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-5e-5}"
SFT_MAX_SEQ_LENGTH="${SFT_MAX_SEQ_LENGTH:-512}"
GRPO_NUM_TRAIN_EPOCHS="${GRPO_NUM_TRAIN_EPOCHS:-6}"
GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-5e-6}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-8}"
GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-768}"

case "$STAGE" in
  sft|grpo|all) ;;
  *)
    echo "Unknown stage: $STAGE (expected sft, grpo, or all)" >&2
    exit 2
    ;;
esac

mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
git branch --show-current > "$ROOT/code_branch.txt"
git status --short > "$ROOT/code_status.txt"
if [[ -f data/dataset_summary.json ]]; then
  cp data/dataset_summary.json "$ROOT/dataset_summary.json"
fi
SFT_SUMMARY="${SFT_DATA_PATH%.jsonl}_summary.json"
if [[ -f "$SFT_SUMMARY" ]]; then
  cp "$SFT_SUMMARY" "$ROOT/rejection_sft_summary.json"
fi
DATASET_SUMMARY="${DATASET_SUMMARY_PATH:-$(dirname "$SFT_DATA_PATH")/summary.json}"
if [[ -f "$DATASET_SUMMARY" ]]; then
  cp "$DATASET_SUMMARY" "$ROOT/training_dataset_summary.json"
fi

run_sft() {
  if [[ ! -f "$SFT_DATA_PATH" ]]; then
    echo "Missing SFT data: $SFT_DATA_PATH" >&2
    echo "Generate it with generate_rejection_sft.py first." >&2
    exit 1
  fi
  if [[ -f "$ROOT/sft/adapter/adapter_config.json" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "SFT adapter already exists at $ROOT/sft/adapter" >&2
    echo "Use a new experiment name, or set ALLOW_OVERWRITE=1 deliberately." >&2
    exit 1
  fi

  resume_args=()
  if [[ -n "${SFT_RESUME_FROM_CHECKPOINT:-}" ]]; then
    resume_args+=(--resume-from-checkpoint "$SFT_RESUME_FROM_CHECKPOINT")
  fi

  python train_sft.py \
    --model-name "$MODEL_NAME" \
    --train-data-path "$SFT_DATA_PATH" \
    --output-dir "$ROOT/sft/adapter" \
    --checkpoint-dir "$ROOT/sft/checkpoints" \
    --run-root "$ROOT" \
    --run-name sft \
    --num-train-epochs "$SFT_NUM_TRAIN_EPOCHS" \
    --learning-rate "$SFT_LEARNING_RATE" \
    --max-seq-length "$SFT_MAX_SEQ_LENGTH" \
    "${resume_args[@]}"
}

run_grpo() {
  if [[ ! -f "$GRPO_INIT_ADAPTER_PATH/adapter_config.json" ]]; then
    echo "Missing initial adapter: $GRPO_INIT_ADAPTER_PATH" >&2
    echo "Run the sft stage first or set GRPO_INIT_ADAPTER_PATH." >&2
    exit 1
  fi
  if [[ ! -f "$TRAIN_DATA_PATH" ]]; then
    echo "Missing GRPO data: $TRAIN_DATA_PATH" >&2
    exit 1
  fi
  if [[ -f "$ROOT/grpo/adapter/adapter_config.json" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "GRPO adapter already exists at $ROOT/grpo/adapter" >&2
    echo "Use a new experiment name, or set ALLOW_OVERWRITE=1 deliberately." >&2
    exit 1
  fi

  resume_args=()
  if [[ -n "${GRPO_RESUME_FROM_CHECKPOINT:-}" ]]; then
    resume_args+=(--resume-from-checkpoint "$GRPO_RESUME_FROM_CHECKPOINT" --no-reset-metrics)
  fi

  python train.py \
    --model-name "$MODEL_NAME" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --adapter-init-path "$GRPO_INIT_ADAPTER_PATH" \
    --output-dir "$ROOT/grpo/adapter" \
    --checkpoint-dir "$ROOT/grpo/checkpoints" \
    --run-root "$ROOT" \
    --run-name grpo \
    --num-train-epochs "$GRPO_NUM_TRAIN_EPOCHS" \
    --learning-rate "$GRPO_LEARNING_RATE" \
    --num-generations "$GRPO_NUM_GENERATIONS" \
    --max-completion-length "$GRPO_MAX_COMPLETION_LENGTH" \
    "${resume_args[@]}"
}

if [[ "$STAGE" == "sft" || "$STAGE" == "all" ]]; then
  run_sft
fi
if [[ "$STAGE" == "grpo" || "$STAGE" == "all" ]]; then
  run_grpo
fi
