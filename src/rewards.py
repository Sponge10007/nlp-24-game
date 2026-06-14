import os
import re
from collections import Counter, deque
from datetime import datetime
from typing import Any

from src.game24 import (
    CORRECT,
    ILLEGAL_CHARACTER,
    FABRICATED_UNSOLVABLE,
    FALSE_UNSOLVABLE_CLAIM,
    MISSING_ANSWER,
    NUMBER_MISMATCH,
    UNSOLVABLE_CLAIM,
    WRONG_VALUE,
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
    "equal_sign_count",
    "unicode_operator_count",
    "answer_text_count",
    "legal_expr_wrong_value_count",
    "bare_target_count",
    "copied_prompt_example_count",
]

UNICODE_OPERATOR_RE = re.compile(r"[×✕✖÷]")
ANSWER_TEXT_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")
COPIED_PROMPT_EXAMPLE = "8/(3-8/3)"


def protocol_issue_counts(answer: str, target_value: int | float = 24) -> Counter[str]:
    counts: Counter[str] = Counter()
    stripped_answer = answer.strip()
    if "=" in answer:
        counts["equal_sign_count"] += 1
    if UNICODE_OPERATOR_RE.search(answer):
        counts["unicode_operator_count"] += 1
    if ANSWER_TEXT_RE.search(answer) and stripped_answer.upper() != "UNSOLVABLE":
        counts["answer_text_count"] += 1
    if stripped_answer == str(target_value) or stripped_answer == str(float(target_value)):
        counts["bare_target_count"] += 1
    if re.sub(r"\s+", "", answer) == COPIED_PROMPT_EXAMPLE:
        counts["copied_prompt_example_count"] += 1
    return counts


def reward_for_judgment(code: str, value: float | None, answer: str, target_value: int | float = 24) -> float:
    if code in {CORRECT, UNSOLVABLE_CLAIM}:
        return 2.0
    if code == FABRICATED_UNSOLVABLE:
        return -1.5
    if code == MISSING_ANSWER:
        return -0.8
    if code == FALSE_UNSOLVABLE_CLAIM:
        return -0.8
    if code == ILLEGAL_CHARACTER:
        issues = protocol_issue_counts(answer)
        if issues["answer_text_count"]:
            return -0.8
        if issues["equal_sign_count"] or issues["unicode_operator_count"]:
            return -0.7
        return -0.6
    if code == NUMBER_MISMATCH:
        issues = protocol_issue_counts(answer, target_value)
        if issues["bare_target_count"] or issues["copied_prompt_example_count"]:
            return -0.9
        return -0.8
    if code == WRONG_VALUE:
        return 0.1 if value is not None else -0.3
    return -0.5 if value is None else 0.0


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
    protocol_counts: Counter[str] = Counter()
    format_count = 0

    for comp, nums, is_solvable, tgt in zip(completions, target_nums, solvable_values, target_values):
        ans = extract_answer(comp)
        judgment = judge_answer(ans, nums, target_value=tgt, solvable=bool(is_solvable))
        judgments.append(judgment)
        code_counts[judgment.code] += 1
        protocol_counts.update(protocol_issue_counts(ans, tgt))
        if judgment.code == WRONG_VALUE and judgment.value is not None:
            protocol_counts["legal_expr_wrong_value_count"] += 1
        if has_r1_format(comp):
            format_count += 1

        rewards.append(reward_for_judgment(judgment.code, judgment.value, ans, tgt))

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
        "equal_sign_count": protocol_counts["equal_sign_count"],
        "unicode_operator_count": protocol_counts["unicode_operator_count"],
        "answer_text_count": protocol_counts["answer_text_count"],
        "legal_expr_wrong_value_count": protocol_counts["legal_expr_wrong_value_count"],
        "bare_target_count": protocol_counts["bare_target_count"],
        "copied_prompt_example_count": protocol_counts["copied_prompt_example_count"],
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
