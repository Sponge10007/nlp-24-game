#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEST="${1:-reports/results}"
mkdir -p "$DEST"
MANIFEST="$DEST/MANIFEST.txt"
: > "$MANIFEST"

copy_result() {
  local source="$1"
  local target="$2"
  if [[ ! -f "$source" ]]; then
    echo "SKIP  $source" | tee -a "$MANIFEST"
    return
  fi
  mkdir -p "$(dirname "$DEST/$target")"
  cp "$source" "$DEST/$target"
  echo "COPY  $source -> $DEST/$target" | tee -a "$MANIFEST"
}

archive_experiment() {
  local version="$1"
  local source_root="$2"

  copy_result "$source_root/sft/sft_config.json" "$version/sft/sft_config.json"
  copy_result "$source_root/sft/eval/summary.json" "$version/sft/eval_summary.json"
  copy_result "$source_root/grpo/config.json" "$version/grpo/config.json"
  copy_result "$source_root/grpo/eval/summary.json" "$version/grpo/eval_summary.json"
  copy_result "$source_root/grpo/eval_pass4/summary.json" "$version/grpo/eval_pass4_summary.json"
  copy_result "$source_root/grpo/accuracy_curve.png" "$version/grpo/accuracy_curve.png"
  copy_result "$source_root/training_dataset_summary.json" "$version/training_dataset_summary.json"
  copy_result "$source_root/dataset_summary.json" "$version/base_dataset_summary.json"
}

copy_result "outputs/experiments/v0_base/eval/summary.json" "v0_base/eval_summary.json"
archive_experiment "v1_grpo_only" "outputs/experiments/v1_grpo_only"
archive_experiment "v2_long_cot_sft_failed" "outputs/experiments/v2_deepseek500_sft_grpo"
archive_experiment "v3_compact_sft_grpo" "outputs/experiments/v3_compact_sft_grpo"
archive_experiment "v4_multi_solution_unsolvable" "outputs/experiments/v4_multi_solution_unsolvable"
archive_experiment "v4_1_canonical_solution" "outputs/experiments/v4_1_canonical_solution"
archive_experiment "v5_continue_grpo_mixed100" "outputs/experiments/v5_continue_grpo_mixed100"

copy_result "outputs/datasets/deepseek500_v1/train_summary.json" "datasets/deepseek500_v1_summary.json"
copy_result "outputs/datasets/deepseek500_compact_v2/train_summary.json" "datasets/deepseek500_compact_v2_summary.json"
copy_result "outputs/datasets/v4_multi_solution_unsolvable/summary.json" "datasets/v4_summary.json"
copy_result "outputs/datasets/v4_multi_solution_unsolvable/sha256.txt" "datasets/v4_sha256.txt"
copy_result "outputs/datasets/v4_1_canonical_solution/summary.json" "datasets/v4_1_summary.json"
copy_result "outputs/datasets/v4_1_canonical_solution/sha256.txt" "datasets/v4_1_sha256.txt"
copy_result "outputs/datasets/v5_continue_grpo_mixed100/summary.json" "datasets/v5_summary.json"
copy_result "outputs/datasets/v5_continue_grpo_mixed100/sha256.txt" "datasets/v5_sha256.txt"

{
  echo
  echo "Archived at: $(date -Iseconds)"
  echo "Git commit: $(git rev-parse HEAD)"
  echo "Git branch: $(git branch --show-current)"
} >> "$MANIFEST"

echo
echo "Lightweight results archived to $DEST"
echo "Review with: git status --short $DEST"
