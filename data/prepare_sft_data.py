import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.prepare_data import puzzle_key
from src.game24 import CORRECT, judge_answer


TRAIN_PATH = "data/train.jsonl"
SFT_TRAIN_PATH = "data/sft_train.jsonl"
NLILE_DATASET = "nlile/24-game"

SOLUTION_TRANSLATION = str.maketrans(
    {
        "×": "*",
        "✕": "*",
        "✖": "*",
        "÷": "/",
        "（": "(",
        "）": ")",
        "−": "-",
    }
)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_solution(solution: str) -> str:
    return " ".join(str(solution).translate(SOLUTION_TRANSLATION).strip().split())


def load_train_keys(train_path: str = TRAIN_PATH) -> set[tuple[int, ...]]:
    return {puzzle_key(row["target_nums"]) for row in load_jsonl(train_path)}


def _rows_from_local_arrow_cache() -> list[dict[str, Any]]:
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the local nlile/24-game cache") from exc

    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "nlile___24-game"
    arrow_files = sorted(cache_root.glob("**/24-game-train.arrow"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not arrow_files:
        raise FileNotFoundError(f"Could not find cached nlile/24-game Arrow file under {cache_root}")

    with ipc.open_stream(str(arrow_files[0])) as reader:
        return reader.read_all().to_pylist()


def load_nlile_rows() -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in load_dataset(NLILE_DATASET, split="train")]
    except Exception as exc:
        print(f"Warning: load_dataset({NLILE_DATASET!r}) failed; falling back to local Arrow cache: {exc}")
        return _rows_from_local_arrow_cache()


def make_sft_completion(answer: str) -> str:
    return f"<think>Use each given number once.</think>\n<answer>{answer}</answer>"


def build_sft_samples(
    nlile_rows: Iterable[dict[str, Any]],
    train_keys: set[tuple[int, ...]],
    *,
    include_all_solutions: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    seen_pairs: set[tuple[tuple[int, ...], str]] = set()
    stats = {
        "raw_rows": 0,
        "train_rows_seen": 0,
        "raw_solutions_seen": 0,
        "valid_solutions": 0,
        "invalid_solutions": 0,
        "duplicate_solutions": 0,
        "rows_with_valid_solution": 0,
    }

    for row in nlile_rows:
        stats["raw_rows"] += 1
        nums = [int(num) for num in row.get("numbers", [])]
        if len(nums) != 4 or puzzle_key(nums) not in train_keys:
            continue

        stats["train_rows_seen"] += 1
        row_has_valid_solution = False
        for raw_solution in row.get("solutions", []) or []:
            stats["raw_solutions_seen"] += 1
            answer = normalize_solution(raw_solution)
            judgment = judge_answer(answer, nums, target_value=24, solvable=True)
            if judgment.code != CORRECT:
                stats["invalid_solutions"] += 1
                continue

            stats["valid_solutions"] += 1
            row_has_valid_solution = True
            pair = (puzzle_key(nums), answer)
            if pair in seen_pairs:
                stats["duplicate_solutions"] += 1
                continue
            seen_pairs.add(pair)

            samples.append(
                {
                    "target_nums": nums,
                    "target_value": 24,
                    "solvable": True,
                    "source": NLILE_DATASET,
                    "source_solution": raw_solution,
                    "answer": answer,
                    "completion": make_sft_completion(answer),
                }
            )
            if not include_all_solutions:
                break

        if row_has_valid_solution:
            stats["rows_with_valid_solution"] += 1

    stats["exported_samples"] = len(samples)
    return samples, stats


def prepare_sft_data(
    train_path: str = TRAIN_PATH,
    output_path: str = SFT_TRAIN_PATH,
    *,
    include_all_solutions: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    train_keys = load_train_keys(train_path)
    rows = load_nlile_rows()
    samples, stats = build_sft_samples(rows, train_keys, include_all_solutions=include_all_solutions)
    write_jsonl(output_path, samples)
    stats["train_key_count"] = len(train_keys)
    print(f"Wrote {output_path}: {len(samples)} SFT samples from {stats['rows_with_valid_solution']} train puzzles.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict SFT data from nlile/24-game reference solutions.")
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--output-path", default=SFT_TRAIN_PATH)
    parser.add_argument(
        "--one-solution-per-puzzle",
        action="store_true",
        help="Export only the first valid reference solution for each train puzzle.",
    )
    args = parser.parse_args()

    prepare_sft_data(
        train_path=args.train_path,
        output_path=args.output_path,
        include_all_solutions=not args.one_solution_per_puzzle,
    )


if __name__ == "__main__":
    main()
