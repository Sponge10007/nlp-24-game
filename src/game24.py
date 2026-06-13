import ast
import math
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ALLOWED_EXPR_RE = re.compile(r"^[0-9+\-*/().\s]+$")

CORRECT = "correct"
UNSOLVABLE_CLAIM = "unsolvable_claim"
FALSE_UNSOLVABLE_CLAIM = "false_unsolvable_claim"
MISSING_ANSWER = "missing_answer"
NUMBER_MISMATCH = "number_mismatch"
ILLEGAL_CHARACTER = "illegal_character"
SYNTAX_ERROR = "syntax_error"
DIVISION_BY_ZERO = "division_by_zero"
WRONG_VALUE = "wrong_value"
FABRICATED_UNSOLVABLE = "fabricated_unsolvable"


@dataclass(frozen=True)
class Judgment:
    ok: bool
    code: str
    answer: str
    value: float | None = None
    message: str = ""


def completion_to_text(completion: Any) -> str:
    """Normalize TRL string/chat completions into plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def extract_answer(completion: Any) -> str:
    text = completion_to_text(completion)
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def has_r1_format(completion: Any) -> bool:
    text = completion_to_text(completion)
    return all(tag in text for tag in ("<think>", "</think>", "<answer>", "</answer>"))


def _numbers_in_expression(expr: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", expr)]


def _safe_eval_fraction(expr: str) -> Fraction:
    tree = ast.parse(expr, mode="eval")

    def eval_node(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                return Fraction(node.value, 1)
            raise ValueError("only integer literals are allowed")
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("division by zero")
                return left / right
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        if isinstance(node, ast.UnaryOp):
            value = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return eval_node(tree)


def judge_answer(
    answer: str,
    target_nums: list[int],
    *,
    target_value: int | float = 24,
    solvable: bool = True,
    tolerance: float = 1e-6,
) -> Judgment:
    answer = (answer or "").strip()
    if not answer:
        return Judgment(False, MISSING_ANSWER, answer, message="missing <answer> content")

    if answer.upper() == "UNSOLVABLE":
        if solvable:
            return Judgment(False, FALSE_UNSOLVABLE_CLAIM, answer, message="claimed UNSOLVABLE for a solvable case")
        return Judgment(True, UNSOLVABLE_CLAIM, answer, message="honest unsolvable claim")

    if not solvable:
        return Judgment(False, FABRICATED_UNSOLVABLE, answer, message="fabricated an expression for an unsolvable case")

    if not ALLOWED_EXPR_RE.fullmatch(answer):
        return Judgment(False, ILLEGAL_CHARACTER, answer, message="expression contains characters outside digits/operators/parentheses")

    used_nums = _numbers_in_expression(answer)
    if Counter(used_nums) != Counter(target_nums):
        return Judgment(False, NUMBER_MISMATCH, answer, message=f"used numbers {used_nums}, expected {target_nums}")

    try:
        value = _safe_eval_fraction(answer)
    except ZeroDivisionError as exc:
        return Judgment(False, DIVISION_BY_ZERO, answer, message=str(exc))
    except (SyntaxError, ValueError) as exc:
        return Judgment(False, SYNTAX_ERROR, answer, message=str(exc))

    value_float = float(value)
    if math.isclose(value_float, float(target_value), abs_tol=tolerance):
        return Judgment(True, CORRECT, answer, value=value_float)
    return Judgment(False, WRONG_VALUE, answer, value=value_float, message=f"got {value_float}, expected {target_value}")


def judge_completion(
    completion: Any,
    target_nums: list[int],
    *,
    target_value: int | float = 24,
    solvable: bool = True,
    tolerance: float = 1e-6,
) -> Judgment:
    return judge_answer(
        extract_answer(completion),
        target_nums,
        target_value=target_value,
        solvable=solvable,
        tolerance=tolerance,
    )
