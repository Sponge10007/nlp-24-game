import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.prepare_compact_sft_data import arithmetic_trace
from data.prepare_data import puzzle_key
from data.prepare_v4_training_data import (
    find_solutions,
    load_jsonl,
    prompt_for,
    select_unsolvable_train_rows,
    write_jsonl,
)
from src.game24 import CORRECT, judge_answer


TRAIN_PATH = "data/train.jsonl"
TEACHER_SFT_PATH = "outputs/datasets/deepseek500_compact_v2/train.jsonl"
SOLVABLE_TEST_PATH = "data/test_all_nonoverlap.jsonl"
UNSOLVABLE_TEST_PATH = "data/unsolvable_test.jsonl"
OUTPUT_DIR = "outputs/datasets/v4_1_canonical_solution"
DEFAULT_UNSOLVABLE_SFT_SIZE = 150
DEFAULT_UNSOLVABLE_GRPO_SIZE = 300
DEFAULT_SEARCH_CANDIDATES = 16
DEFAULT_SEED = 20260622


def expression_score(expression: str) -> tuple[int, int, int, str]:
    compact = re.sub(r"\s+", "", expression)
    return (
        compact.count("(") + compact.count(")"),
        len(compact),
        compact.count("/") + compact.count("-"),
        compact,
    )


def choose_canonical_solution(solutions: list[str]) -> str:
    if not solutions:
        raise ValueError("cannot select a canonical solution from an empty list")
    return min(solutions, key=expression_score)


def teacher_answers_by_key(
    teacher_rows: list[dict[str, Any]],
    train_keys: set[tuple[int, ...]],
) -> dict[tuple[int, ...], str]:
    candidates: dict[tuple[int, ...], list[str]] = {}
    for row in teacher_rows:
        nums = [int(num) for num in row["target_nums"]]
        key = puzzle_key(nums)
        if key not in train_keys:
            continue
        answer = str(row["answer"]).strip()
        judgment = judge_answer(
            answer,
            nums,
            target_value=row.get("target_value", 24),
            solvable=True,
        )
        if judgment.code == CORRECT:
            candidates.setdefault(key, []).append(answer)
    return {
        key: choose_canonical_solution(answers)
        for key, answers in candidates.items()
    }


def build_solvable_sft_rows(
    train_rows: list[dict[str, Any]],
    teacher_answers: dict[tuple[int, ...], str],
    search_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "teacher_solution_rows": 0,
        "deterministic_solution_rows": 0,
        "missing_solution_rows": 0,
    }
    seen_keys: set[tuple[int, ...]] = set()

    for source_row in train_rows:
        nums = [int(num) for num in source_row["target_nums"]]
        key = puzzle_key(nums)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        target_value = source_row.get("target_value", 24)

        if key in teacher_answers:
            answer = teacher_answers[key]
            solution_source = "deepseek_compact_v2"
            stats["teacher_solution_rows"] += 1
        else:
            solutions = find_solutions(
                nums,
                target_value=target_value,
                max_solutions=search_candidates,
            )
            if not solutions:
                stats["missing_solution_rows"] += 1
                continue
            answer = choose_canonical_solution(solutions)
            solution_source = "deterministic_canonical_search"
            stats["deterministic_solution_rows"] += 1

        judgment = judge_answer(
            answer,
            nums,
            target_value=target_value,
            solvable=True,
        )
        if judgment.code != CORRECT:
            raise ValueError(f"invalid canonical answer for {nums}: {judgment.code}")
        trace = arithmetic_trace(answer)
        completion = f"<think>{trace}</think>\n<answer>{answer}</answer>"
        rows.append(
            {
                "target_nums": nums,
                "target_value": target_value,
                "solvable": True,
                "prompt": prompt_for(nums, target_value),
                "completion": [{"role": "assistant", "content": completion}],
                "answer": answer,
                "arithmetic_trace": trace,
                "solution_source": solution_source,
            }
        )
    return rows, stats


def build_unsolvable_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_row in rows:
        nums = [int(num) for num in source_row["target_nums"]]
        target_value = source_row.get("target_value", 24)
        result.append(
            {
                **source_row,
                "prompt": prompt_for(nums, target_value),
                "completion": [
                    {
                        "role": "assistant",
                        "content": "<think>无合法算式</think>\n<answer>UNSOLVABLE</answer>",
                    }
                ],
                "answer": "UNSOLVABLE",
                "solution_source": "local_enum_unsolvable_train",
            }
        )
    return result


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_v41_data(
    train_path: str,
    teacher_sft_path: str,
    solvable_test_path: str,
    unsolvable_test_path: str,
    output_dir: str,
    unsolvable_sft_size: int,
    unsolvable_grpo_size: int,
    search_candidates: int,
    seed: int,
) -> dict[str, Any]:
    train_rows = load_jsonl(train_path)
    teacher_rows = load_jsonl(teacher_sft_path)
    solvable_test_rows = load_jsonl(solvable_test_path)
    unsolvable_test_rows = load_jsonl(unsolvable_test_path)

    train_keys = {puzzle_key(row["target_nums"]) for row in train_rows}
    solvable_test_keys = {puzzle_key(row["target_nums"]) for row in solvable_test_rows}
    solvable_overlap = len(train_keys & solvable_test_keys)
    if solvable_overlap:
        raise ValueError(f"solvable train/test overlap detected: {solvable_overlap}")

    teacher_answers = teacher_answers_by_key(teacher_rows, train_keys)
    solvable_sft_rows, solvable_stats = build_solvable_sft_rows(
        train_rows,
        teacher_answers,
        search_candidates=search_candidates,
    )

    unsolvable_grpo_rows, unsolvable_candidate_count = select_unsolvable_train_rows(
        unsolvable_test_rows,
        train_size=unsolvable_grpo_size,
        seed=seed,
    )
    unsolvable_sft_rows = unsolvable_grpo_rows[:unsolvable_sft_size]
    unsolvable_test_keys = {puzzle_key(row["target_nums"]) for row in unsolvable_test_rows}
    unsolvable_train_keys = {puzzle_key(row["target_nums"]) for row in unsolvable_grpo_rows}
    unsolvable_overlap = len(unsolvable_test_keys & unsolvable_train_keys)
    if unsolvable_overlap:
        raise ValueError(f"unsolvable train/test overlap detected: {unsolvable_overlap}")

    sft_rows = solvable_sft_rows + build_unsolvable_sft_rows(unsolvable_sft_rows)
    random.Random(seed).shuffle(sft_rows)

    grpo_rows = [
        {
            "target_nums": [int(num) for num in row["target_nums"]],
            "target_value": row.get("target_value", 24),
            "solvable": True,
            "source": row.get("source", "solvable_train"),
        }
        for row in train_rows
    ] + unsolvable_grpo_rows
    random.Random(seed).shuffle(grpo_rows)

    os.makedirs(output_dir, exist_ok=True)
    sft_path = os.path.join(output_dir, "sft_train.jsonl")
    grpo_path = os.path.join(output_dir, "grpo_train.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    hashes_path = os.path.join(output_dir, "sha256.txt")
    write_jsonl(sft_path, sft_rows)
    write_jsonl(grpo_path, grpo_rows)

    unique_solvable_sft_keys = {
        puzzle_key(row["target_nums"])
        for row in solvable_sft_rows
    }
    summary = {
        "version": "v4_1_canonical_solution",
        "seed": seed,
        "source_train_path": train_path,
        "teacher_sft_path": teacher_sft_path,
        "solvable_test_path": solvable_test_path,
        "unsolvable_test_path": unsolvable_test_path,
        "search_candidates_per_missing_puzzle": search_candidates,
        "solvable_train_puzzles": len(train_keys),
        "unique_solvable_sft_puzzles": len(unique_solvable_sft_keys),
        **solvable_stats,
        "unsolvable_candidates_after_test_exclusion": unsolvable_candidate_count,
        "unsolvable_sft_puzzles": len(unsolvable_sft_rows),
        "unsolvable_grpo_puzzles": len(unsolvable_grpo_rows),
        "solvable_test_overlap": solvable_overlap,
        "unsolvable_test_overlap": unsolvable_overlap,
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
        description="Build v4.1 canonical one-answer SFT data and mixed solvable/unsolvable GRPO data."
    )
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--teacher-sft-path", default=TEACHER_SFT_PATH)
    parser.add_argument("--solvable-test-path", default=SOLVABLE_TEST_PATH)
    parser.add_argument("--unsolvable-test-path", default=UNSOLVABLE_TEST_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--unsolvable-sft-size", type=int, default=DEFAULT_UNSOLVABLE_SFT_SIZE)
    parser.add_argument("--unsolvable-grpo-size", type=int, default=DEFAULT_UNSOLVABLE_GRPO_SIZE)
    parser.add_argument("--search-candidates", type=int, default=DEFAULT_SEARCH_CANDIDATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.unsolvable_sft_size > args.unsolvable_grpo_size:
        parser.error("--unsolvable-sft-size cannot exceed --unsolvable-grpo-size")

    prepare_v41_data(
        train_path=args.train_path,
        teacher_sft_path=args.teacher_sft_path,
        solvable_test_path=args.solvable_test_path,
        unsolvable_test_path=args.unsolvable_test_path,
        output_dir=args.output_dir,
        unsolvable_sft_size=args.unsolvable_sft_size,
        unsolvable_grpo_size=args.unsolvable_grpo_size,
        search_candidates=args.search_candidates,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
