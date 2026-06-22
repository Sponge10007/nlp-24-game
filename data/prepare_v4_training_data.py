import argparse
import hashlib
import json
import os
import random
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.prepare_compact_sft_data import arithmetic_trace
from data.prepare_data import enumerate_unsolvable_samples, puzzle_key
from src.game24 import CORRECT, judge_answer
from src.prompts import SYSTEM_PROMPT, get_prompt


TRAIN_PATH = "data/train.jsonl"
UNSOLVABLE_TEST_PATH = "data/unsolvable_test.jsonl"
SOLVABLE_TEST_PATH = "data/test_all_nonoverlap.jsonl"
OUTPUT_DIR = "outputs/datasets/v4_multi_solution_unsolvable"
DEFAULT_MAX_SOLUTIONS = 3
DEFAULT_UNSOLVABLE_TRAIN_SIZE = 300
DEFAULT_SEED = 20260622


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


def prompt_for(nums: list[int], target_value: int | float = 24) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_prompt(nums, target_value)},
    ]


def _strip_outer_parentheses(expression: str) -> str:
    if not expression.startswith("(") or not expression.endswith(")"):
        return expression
    depth = 0
    for index, char in enumerate(expression):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and index != len(expression) - 1:
            return expression
    return expression[1:-1]


def find_solutions(
    nums: list[int],
    target_value: int | float = 24,
    max_solutions: int = DEFAULT_MAX_SOLUTIONS,
) -> list[str]:
    target = Fraction(target_value)
    solutions: list[str] = []
    seen_solutions: set[str] = set()

    def search(items: list[tuple[Fraction, str]]) -> None:
        if len(solutions) >= max_solutions:
            return
        if len(items) == 1:
            value, expression = items[0]
            if value != target:
                return
            expression = _strip_outer_parentheses(expression)
            compact = expression.replace(" ", "")
            if compact in seen_solutions:
                return
            judgment = judge_answer(
                expression,
                nums,
                target_value=target_value,
                solvable=True,
            )
            if judgment.code == CORRECT:
                seen_solutions.add(compact)
                solutions.append(expression)
            return

        for left_index in range(len(items)):
            for right_index in range(left_index + 1, len(items)):
                left_value, left_expr = items[left_index]
                right_value, right_expr = items[right_index]
                rest = [
                    item
                    for index, item in enumerate(items)
                    if index not in {left_index, right_index}
                ]
                candidates = [
                    (left_value + right_value, f"({left_expr}+{right_expr})"),
                    (left_value * right_value, f"({left_expr}*{right_expr})"),
                    (left_value - right_value, f"({left_expr}-{right_expr})"),
                    (right_value - left_value, f"({right_expr}-{left_expr})"),
                ]
                if right_value:
                    candidates.append((left_value / right_value, f"({left_expr}/{right_expr})"))
                if left_value:
                    candidates.append((right_value / left_value, f"({right_expr}/{left_expr})"))

                seen_candidates: set[tuple[Fraction, str]] = set()
                for value, expression in candidates:
                    candidate = (value, expression)
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    search(rest + [candidate])
                    if len(solutions) >= max_solutions:
                        return

    search([(Fraction(num), str(num)) for num in nums])
    return solutions


def build_solvable_sft_rows(
    train_rows: list[dict[str, Any]],
    max_solutions: int,
) -> tuple[list[dict[str, Any]], int]:
    sft_rows: list[dict[str, Any]] = []
    missing_solution_count = 0
    for row in train_rows:
        nums = [int(num) for num in row["target_nums"]]
        target_value = row.get("target_value", 24)
        solutions = find_solutions(nums, target_value, max_solutions=max_solutions)
        if not solutions:
            missing_solution_count += 1
            continue
        for solution_index, answer in enumerate(solutions, start=1):
            trace = arithmetic_trace(answer)
            completion = f"<think>{trace}</think>\n<answer>{answer}</answer>"
            sft_rows.append(
                {
                    "target_nums": nums,
                    "target_value": target_value,
                    "solvable": True,
                    "prompt": prompt_for(nums, target_value),
                    "completion": [{"role": "assistant", "content": completion}],
                    "answer": answer,
                    "arithmetic_trace": trace,
                    "solution_index": solution_index,
                    "source": "deterministic_multi_solution_search",
                }
            )
    return sft_rows, missing_solution_count


def select_unsolvable_train_rows(
    unsolvable_test_rows: list[dict[str, Any]],
    train_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    test_keys = {puzzle_key(row["target_nums"]) for row in unsolvable_test_rows}
    candidates = [
        row
        for row in enumerate_unsolvable_samples()
        if puzzle_key(row["target_nums"]) not in test_keys
    ]
    random.Random(seed).shuffle(candidates)
    selected = candidates[:train_size] if train_size >= 0 else candidates
    return selected, len(candidates)


def build_unsolvable_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sft_rows: list[dict[str, Any]] = []
    for row in rows:
        nums = [int(num) for num in row["target_nums"]]
        target_value = row.get("target_value", 24)
        completion = "<think>无合法算式</think>\n<answer>UNSOLVABLE</answer>"
        sft_rows.append(
            {
                **row,
                "prompt": prompt_for(nums, target_value),
                "completion": [{"role": "assistant", "content": completion}],
                "answer": "UNSOLVABLE",
                "source": "local_enum_unsolvable_train",
            }
        )
    return sft_rows


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_v4_data(
    train_path: str,
    solvable_test_path: str,
    unsolvable_test_path: str,
    output_dir: str,
    max_solutions: int,
    unsolvable_train_size: int,
    seed: int,
) -> dict[str, Any]:
    train_rows = load_jsonl(train_path)
    solvable_test_rows = load_jsonl(solvable_test_path)
    unsolvable_test_rows = load_jsonl(unsolvable_test_path)

    solvable_sft_rows, missing_solution_count = build_solvable_sft_rows(
        train_rows,
        max_solutions=max_solutions,
    )
    unsolvable_train_rows, unsolvable_candidate_count = select_unsolvable_train_rows(
        unsolvable_test_rows,
        train_size=unsolvable_train_size,
        seed=seed,
    )
    unsolvable_sft_rows = build_unsolvable_sft_rows(unsolvable_train_rows)

    sft_rows = solvable_sft_rows + unsolvable_sft_rows
    random.Random(seed).shuffle(sft_rows)

    grpo_rows = [
        {
            "target_nums": [int(num) for num in row["target_nums"]],
            "target_value": row.get("target_value", 24),
            "solvable": True,
            "source": row.get("source", "solvable_train"),
        }
        for row in train_rows
    ] + unsolvable_train_rows
    random.Random(seed).shuffle(grpo_rows)

    os.makedirs(output_dir, exist_ok=True)
    sft_path = os.path.join(output_dir, "sft_train.jsonl")
    grpo_path = os.path.join(output_dir, "grpo_train.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    hashes_path = os.path.join(output_dir, "sha256.txt")
    write_jsonl(sft_path, sft_rows)
    write_jsonl(grpo_path, grpo_rows)

    test_keys = {puzzle_key(row["target_nums"]) for row in unsolvable_test_rows}
    train_unsolvable_keys = {puzzle_key(row["target_nums"]) for row in unsolvable_train_rows}
    solvable_test_keys = {puzzle_key(row["target_nums"]) for row in solvable_test_rows}
    solvable_train_keys = {puzzle_key(row["target_nums"]) for row in train_rows}
    solvable_test_overlap = len(solvable_test_keys & solvable_train_keys)
    unsolvable_test_overlap = len(test_keys & train_unsolvable_keys)
    if solvable_test_overlap:
        raise ValueError(f"solvable train/test overlap detected: {solvable_test_overlap}")
    if unsolvable_test_overlap:
        raise ValueError(f"unsolvable train/test overlap detected: {unsolvable_test_overlap}")
    solution_counts: dict[tuple[int, ...], int] = {}
    for row in solvable_sft_rows:
        key = puzzle_key(row["target_nums"])
        solution_counts[key] = solution_counts.get(key, 0) + 1

    summary = {
        "version": "v4_multi_solution_unsolvable",
        "seed": seed,
        "source_train_path": train_path,
        "solvable_test_path": solvable_test_path,
        "unsolvable_test_path": unsolvable_test_path,
        "max_solutions_per_puzzle": max_solutions,
        "solvable_puzzles": len(train_rows),
        "solvable_puzzles_with_solution": len(solution_counts),
        "solvable_sft_rows": len(solvable_sft_rows),
        "missing_solution_count": missing_solution_count,
        "min_solutions_per_solvable_puzzle": min(solution_counts.values(), default=0),
        "max_exported_solutions_per_puzzle": max(solution_counts.values(), default=0),
        "unsolvable_candidates_after_test_exclusion": unsolvable_candidate_count,
        "unsolvable_train_puzzles": len(unsolvable_train_rows),
        "solvable_test_overlap": solvable_test_overlap,
        "unsolvable_test_overlap": unsolvable_test_overlap,
        "sft_rows": len(sft_rows),
        "grpo_rows": len(grpo_rows),
        "sft_path": sft_path,
        "grpo_path": grpo_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(hashes_path, "w", encoding="utf-8") as f:
        f.write(f"{sha256_file(sft_path)}  sft_train.jsonl\n")
        f.write(f"{sha256_file(grpo_path)}  grpo_train.jsonl\n")
        f.write(f"{sha256_file(summary_path)}  summary.json\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build versioned v4 SFT/GRPO data with multiple exact solutions and held-out unsolvable cases."
    )
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--solvable-test-path", default=SOLVABLE_TEST_PATH)
    parser.add_argument("--unsolvable-test-path", default=UNSOLVABLE_TEST_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-solutions", type=int, default=DEFAULT_MAX_SOLUTIONS)
    parser.add_argument("--unsolvable-train-size", type=int, default=DEFAULT_UNSOLVABLE_TRAIN_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    prepare_v4_data(
        train_path=args.train_path,
        solvable_test_path=args.solvable_test_path,
        unsolvable_test_path=args.unsolvable_test_path,
        output_dir=args.output_dir,
        max_solutions=args.max_solutions,
        unsolvable_train_size=args.unsolvable_train_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
