import argparse
import ast
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.game24 import CORRECT, judge_answer


INPUT_PATH = "outputs/datasets/deepseek500_v1/train.jsonl"
OUTPUT_PATH = "outputs/datasets/deepseek500_compact_v2/train.jsonl"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_operand(value: Fraction) -> str:
    rendered = format_fraction(value)
    if value.denominator != 1 or value < 0:
        return f"({rendered})"
    return rendered


def arithmetic_trace(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    steps: list[str] = []

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        if not isinstance(node, ast.BinOp):
            raise ValueError(f"unsupported expression node: {type(node).__name__}")

        left = evaluate(node.left)
        right = evaluate(node.right)
        if isinstance(node.op, ast.Add):
            symbol, result = "+", left + right
        elif isinstance(node.op, ast.Sub):
            symbol, result = "-", left - right
        elif isinstance(node.op, ast.Mult):
            symbol, result = "*", left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            symbol, result = "/", left / right
        else:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")

        steps.append(
            f"{format_operand(left)}{symbol}{format_operand(right)}={format_fraction(result)}"
        )
        return result

    result = evaluate(tree)
    if not steps:
        steps.append(f"{expression.strip()}={format_fraction(result)}")
    return "; ".join(steps)


def build_compact_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row["answer"]).strip()
    target_nums = [int(num) for num in row["target_nums"]]
    target_value = row.get("target_value", 24)
    judgment = judge_answer(
        answer,
        target_nums,
        target_value=target_value,
        solvable=bool(row.get("solvable", True)),
    )
    if judgment.code != CORRECT:
        raise ValueError(f"invalid source answer for {target_nums}: {judgment.code}")

    think = arithmetic_trace(answer)
    completion_text = f"<think>{think}</think>\n<answer>{answer}</answer>"
    return {
        "target_nums": target_nums,
        "target_value": target_value,
        "solvable": True,
        "prompt": row["prompt"],
        "completion": [{"role": "assistant", "content": completion_text}],
        "answer": answer,
        "arithmetic_trace": think,
        "source": row.get("source"),
        "source_dataset": row.get("rejection_source", "deepseek_api"),
    }


def prepare_compact_data(input_path: str, output_path: str) -> dict[str, Any]:
    source_rows = load_jsonl(input_path)
    compact_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in source_rows:
        try:
            compact_rows.append(build_compact_row(row))
        except (KeyError, TypeError, ValueError, SyntaxError, ZeroDivisionError) as exc:
            skipped += 1
            print(f"Skipped nums={row.get('target_nums')}: {exc}")

    write_jsonl(output_path, compact_rows)
    summary = {
        "input": input_path,
        "output": output_path,
        "source_rows": len(source_rows),
        "written_rows": len(compact_rows),
        "skipped_rows": skipped,
        "think_style": "arithmetic_trace_only",
    }
    summary_path = os.path.splitext(output_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert verified rejection-sampled SFT data into short arithmetic-only traces."
    )
    parser.add_argument("--input-path", default=INPUT_PATH)
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    args = parser.parse_args()
    prepare_compact_data(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
