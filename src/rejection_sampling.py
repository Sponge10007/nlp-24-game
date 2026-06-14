from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.game24 import CORRECT, UNSOLVABLE_CLAIM, extract_answer, has_r1_format, judge_answer
from src.prompts import SYSTEM_PROMPT, get_prompt


@dataclass(frozen=True)
class RejectionSample:
    accepted: bool
    completion: str
    answer: str
    code: str
    value: float | None
    attempt_index: int | None
    source: str


def build_prompt(target_nums: list[int], target_value: int | float = 24) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_prompt(target_nums, target_value)},
    ]


def build_completion_from_deepseek_response(reasoning_content: str | None, content: str | None) -> str:
    content = (content or "").strip()
    reasoning_content = (reasoning_content or "").strip()
    if "<answer>" in content.lower() and "</answer>" in content.lower():
        return content

    think = reasoning_content or "模型未返回 reasoning_content。"
    return f"<think>{think}</think>\n<answer>{content}</answer>"


def validate_rejection_candidate(
    completion: str,
    target_nums: list[int],
    *,
    target_value: int | float = 24,
    solvable: bool = True,
    require_r1_format: bool = True,
) -> RejectionSample:
    answer = extract_answer(completion)
    judgment = judge_answer(answer, target_nums, target_value=target_value, solvable=solvable)
    expected_code = CORRECT if solvable else UNSOLVABLE_CLAIM
    accepted = judgment.ok and judgment.code == expected_code
    if require_r1_format and not has_r1_format(completion):
        accepted = False

    return RejectionSample(
        accepted=accepted,
        completion=completion.strip(),
        answer=answer,
        code=judgment.code,
        value=judgment.value,
        attempt_index=None,
        source="deepseek_api",
    )


def select_rejection_sample(
    completions: list[str],
    target_nums: list[int],
    *,
    target_value: int | float = 24,
    solvable: bool = True,
    require_r1_format: bool = True,
) -> RejectionSample | None:
    for attempt_index, completion in enumerate(completions, start=1):
        sample = validate_rejection_candidate(
            completion,
            target_nums,
            target_value=target_value,
            solvable=solvable,
            require_r1_format=require_r1_format,
        )
        if sample.accepted:
            return RejectionSample(
                accepted=True,
                completion=sample.completion,
                answer=sample.answer,
                code=sample.code,
                value=sample.value,
                attempt_index=attempt_index,
                source=sample.source,
            )
    return None


def build_sft_row(case: dict[str, Any], sample: RejectionSample) -> dict[str, Any]:
    target_nums = case["target_nums"]
    target_value = case.get("target_value", 24)
    return {
        **case,
        "target_value": target_value,
        "prompt": build_prompt(target_nums, target_value),
        "completion": sample.completion,
        "answer": sample.answer,
        "solution": sample.answer if sample.code == CORRECT else None,
        "rejection_source": sample.source,
        "accepted_attempt": sample.attempt_index,
        "judge_code": sample.code,
        "judge_value": sample.value,
    }
