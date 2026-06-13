import os
import re
from collections import Counter, deque
from datetime import datetime
from typing import Any

from src.game24 import (
    CORRECT,
    FABRICATED_UNSOLVABLE,
    FALSE_UNSOLVABLE_CLAIM,
    MISSING_ANSWER,
    UNSOLVABLE_CLAIM,
    completion_to_text,
    extract_answer,
    has_r1_format,
    judge_answer,
)


_log_step = 0
_history_correct = deque(maxlen=100)
_history_total = deque(maxlen=100)
_csv_buffer: list[str] = []
_log_buffer: list[str] = []
_metrics_initialized = False

METRICS_FILE = os.environ.get("TRAIN_METRICS_FILE", "training_metrics.csv")
TRAIN_LOG_FILE = os.environ.get("TRAIN_LOG_FILE", "train_log.txt")
METRICS_COLUMNS = [
    "step",
    "batch_accuracy",
    "smoothed_accuracy",
    "mean_correctness_reward",
    "format_rate",
    "correct_count",
    "unsolvable_honest_count",
    "false_unsolvable_claim_count",
    "missing_answer_count",
    "number_mismatch_count",
    "illegal_character_count",
    "syntax_error_count",
    "division_by_zero_count",
    "wrong_value_count",
    "fabricated_unsolvable_count",
]


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def init_metric_files(reset: bool = False) -> None:
    global _metrics_initialized
    _ensure_parent(METRICS_FILE)
    _ensure_parent(TRAIN_LOG_FILE)
    if reset or not os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            f.write(",".join(METRICS_COLUMNS) + "\n")
    _metrics_initialized = True


def flush_reward_logs() -> None:
    global _csv_buffer, _log_buffer
    if _csv_buffer:
        _ensure_parent(METRICS_FILE)
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.writelines(_csv_buffer)
        _csv_buffer.clear()
    if _log_buffer:
        _ensure_parent(TRAIN_LOG_FILE)
        with open(TRAIN_LOG_FILE, "a", encoding="utf-8") as f:
            f.writelines(_log_buffer)
        _log_buffer.clear()


def configure_reward_logging(metrics_file: str, train_log_file: str, reset: bool = True) -> None:
    global METRICS_FILE, TRAIN_LOG_FILE
    METRICS_FILE = metrics_file
    TRAIN_LOG_FILE = train_log_file
    init_metric_files(reset=reset)


def _as_list(value: Any, length: int, default: Any) -> list[Any]:
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, list):
        return value
    return [value for _ in range(length)]


def format_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        rewards.append(0.5 if has_r1_format(comp) else -1.0)
    return rewards


def correctness_reward(completions, target_nums, solvable=None, target_value=None, **kwargs) -> list[float]:
    global _log_step, _history_correct, _history_total

    if not _metrics_initialized:
        init_metric_files(reset=False)

    _log_step += 1

    total = len(completions)
    solvable_values = _as_list(solvable, total, True)
    target_values = _as_list(target_value, total, 24)

    rewards: list[float] = []
    judgments = []
    code_counts: Counter[str] = Counter()
    format_count = 0

    for comp, nums, is_solvable, tgt in zip(completions, target_nums, solvable_values, target_values):
        ans = extract_answer(comp)
        judgment = judge_answer(ans, nums, target_value=tgt, solvable=bool(is_solvable))
        judgments.append(judgment)
        code_counts[judgment.code] += 1
        if has_r1_format(comp):
            format_count += 1

        if judgment.ok and judgment.code in {CORRECT, UNSOLVABLE_CLAIM}:
            rewards.append(2.0)
        elif judgment.code == FABRICATED_UNSOLVABLE:
            rewards.append(-1.5)
        elif judgment.code == MISSING_ANSWER:
            rewards.append(-0.5)
        elif not judgment.ok:
            rewards.append(-0.5 if judgment.value is None else 0.0)
        else:
            rewards.append(0.0)

    correct_count = code_counts[CORRECT] + code_counts[UNSOLVABLE_CLAIM]
    _history_correct.append(correct_count)
    _history_total.append(total)

    recent_correct = sum(_history_correct)
    recent_total = sum(_history_total)
    smoothed_accuracy = (recent_correct / recent_total) * 100 if recent_total else 0.0
    batch_accuracy = (correct_count / total) * 100 if total else 0.0
    format_rate = (format_count / total) * 100 if total else 0.0
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    print(
        f"\n[Step {_log_step}] acc={batch_accuracy:5.1f}% | "
        f"trend100={smoothed_accuracy:5.1f}% | reward={mean_reward:5.2f} | format={format_rate:5.1f}%"
    )

    row = {
        "step": _log_step,
        "batch_accuracy": f"{batch_accuracy:.2f}",
        "smoothed_accuracy": f"{smoothed_accuracy:.2f}",
        "mean_correctness_reward": f"{mean_reward:.4f}",
        "format_rate": f"{format_rate:.2f}",
        "correct_count": code_counts[CORRECT],
        "unsolvable_honest_count": code_counts[UNSOLVABLE_CLAIM],
        "false_unsolvable_claim_count": code_counts[FALSE_UNSOLVABLE_CLAIM],
        "missing_answer_count": code_counts[MISSING_ANSWER],
        "number_mismatch_count": code_counts["number_mismatch"],
        "illegal_character_count": code_counts["illegal_character"],
        "syntax_error_count": code_counts["syntax_error"],
        "division_by_zero_count": code_counts["division_by_zero"],
        "wrong_value_count": code_counts["wrong_value"],
        "fabricated_unsolvable_count": code_counts[FABRICATED_UNSOLVABLE],
    }
    _csv_buffer.append(",".join(str(row[col]) for col in METRICS_COLUMNS) + "\n")

    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = (
        f"\n{'=' * 20} Step {_log_step} [{time_str}] | "
        f"acc={batch_accuracy:.1f}% | trend100={smoothed_accuracy:.1f}% | "
        f"reward={mean_reward:.3f} {'=' * 20}\n"
    )
    for i, (comp, nums, tgt, reward, judgment) in enumerate(zip(completions, target_nums, target_values, rewards, judgments)):
        compact_comp = re.sub(r"\n\s*\n", "\n", completion_to_text(comp)).strip()
        log_content += (
            f"\n--- [Sample {i + 1}] nums={nums} target={tgt} "
            f"reward={reward} code={judgment.code} value={judgment.value} ---\n"
            f"{compact_comp}\n"
        )
    _log_buffer.append(log_content)

    if len(_csv_buffer) >= 10:
        try:
            flush_reward_logs()
        except Exception as exc:
            print(f"Warning: failed to write reward logs: {exc}")

    return rewards
