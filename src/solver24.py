from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ExprValue:
    value: Fraction
    expr: str


def _format_binary(left: str, op: str, right: str) -> str:
    return f"({left}{op}{right})"


def _search(values: tuple[ExprValue, ...], target: Fraction) -> str | None:
    if len(values) == 1:
        return values[0].expr if values[0].value == target else None

    value_count = len(values)
    for left_index in range(value_count):
        for right_index in range(left_index + 1, value_count):
            left = values[left_index]
            right = values[right_index]
            rest = tuple(
                values[index]
                for index in range(value_count)
                if index not in {left_index, right_index}
            )

            candidates = [
                ExprValue(left.value + right.value, _format_binary(left.expr, "+", right.expr)),
                ExprValue(left.value - right.value, _format_binary(left.expr, "-", right.expr)),
                ExprValue(right.value - left.value, _format_binary(right.expr, "-", left.expr)),
                ExprValue(left.value * right.value, _format_binary(left.expr, "*", right.expr)),
            ]
            if right.value:
                candidates.append(ExprValue(left.value / right.value, _format_binary(left.expr, "/", right.expr)))
            if left.value:
                candidates.append(ExprValue(right.value / left.value, _format_binary(right.expr, "/", left.expr)))

            for candidate in candidates:
                solution = _search(rest + (candidate,), target)
                if solution is not None:
                    return solution
    return None


def strip_outer_parentheses(expr: str) -> str:
    if not (expr.startswith("(") and expr.endswith(")")):
        return expr

    depth = 0
    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and index != len(expr) - 1:
            return expr
    return expr[1:-1]


def find_expression(nums: list[int] | tuple[int, ...], target_value: int | float = 24) -> str | None:
    values = tuple(ExprValue(Fraction(num), str(int(num))) for num in nums)
    solution = _search(values, Fraction(str(target_value)))
    if solution is None:
        return None
    return strip_outer_parentheses(solution)
