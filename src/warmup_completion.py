from __future__ import annotations

from dataclasses import dataclass

from src.game24 import CORRECT, UNSOLVABLE_CLAIM, extract_answer, judge_answer
from src.solver24 import find_expression


SOLVABLE_THINK = "我会先构造一个只使用给定数字一次的表达式，并检查它等于目标值。"
UNSOLVABLE_THINK = "我检查了这些数字在加减乘除和括号组合下无法得到目标值。"


@dataclass(frozen=True)
class WarmupCompletion:
    solution: str | None
    completion: str
    solvable: bool


def _completion(answer: str, think: str) -> str:
    return f"<think>{think}</think>\n<answer>{answer}</answer>"


def _validate_completion(completion: str, target_nums: list[int], target_value: int | float, solvable: bool) -> None:
    answer = extract_answer(completion)
    judgment = judge_answer(answer, target_nums, target_value=target_value, solvable=solvable)
    expected_code = CORRECT if solvable else UNSOLVABLE_CLAIM
    if not judgment.ok or judgment.code != expected_code:
        raise ValueError(
            f"invalid warmup completion for nums={target_nums}, target={target_value}: "
            f"code={judgment.code}, message={judgment.message}"
        )


def build_warmup_completion(
    target_nums: list[int],
    target_value: int | float = 24,
    solvable: bool | None = None,
) -> WarmupCompletion:
    solution = find_expression(target_nums, target_value)

    if solution is not None:
        if solvable is False:
            raise ValueError(f"sample marked unsolvable but solver found solution: nums={target_nums}, solution={solution}")

        judgment = judge_answer(solution, target_nums, target_value=target_value, solvable=True)
        if not judgment.ok or judgment.code != CORRECT:
            raise ValueError(
                f"solver produced invalid solution for nums={target_nums}, target={target_value}: "
                f"solution={solution}, code={judgment.code}, message={judgment.message}"
            )

        completion = _completion(solution, SOLVABLE_THINK)
        _validate_completion(completion, target_nums, target_value, solvable=True)
        return WarmupCompletion(solution=solution, completion=completion, solvable=True)

    if solvable is not False:
        raise ValueError(f"sample marked solvable but solver found no solution: nums={target_nums}, target={target_value}")

    completion = _completion("UNSOLVABLE", UNSOLVABLE_THINK)
    _validate_completion(completion, target_nums, target_value, solvable=False)
    return WarmupCompletion(solution=None, completion=completion, solvable=False)
