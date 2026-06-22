import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.prepare_data import puzzle_key
from data.prepare_v4_training_data import (
    load_jsonl,
    select_unsolvable_train_rows,
    write_jsonl,
)


TRAIN_PATH = "data/train.jsonl"
SOLVABLE_TEST_PATH = "data/test_all_nonoverlap.jsonl"
UNSOLVABLE_TEST_PATH = "data/unsolvable_test.jsonl"
OUTPUT_DIR = "outputs/datasets/v5_continue_grpo_mixed100"
DEFAULT_UNSOLVABLE_SIZE = 100
DEFAULT_SEED = 20260623


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_v5_data(
    train_path: str,
    solvable_test_path: str,
    unsolvable_test_path: str,
    output_dir: str,
    unsolvable_size: int,
    seed: int,
) -> dict[str, Any]:
    solvable_rows = load_jsonl(train_path)
    solvable_test_rows = load_jsonl(solvable_test_path)
    unsolvable_test_rows = load_jsonl(unsolvable_test_path)

    solvable_train_keys = {puzzle_key(row["target_nums"]) for row in solvable_rows}
    solvable_test_keys = {puzzle_key(row["target_nums"]) for row in solvable_test_rows}
    solvable_overlap = len(solvable_train_keys & solvable_test_keys)
    if solvable_overlap:
        raise ValueError(f"solvable train/test overlap detected: {solvable_overlap}")

    unsolvable_rows, candidate_count = select_unsolvable_train_rows(
        unsolvable_test_rows,
        train_size=unsolvable_size,
        seed=seed,
    )
    unsolvable_train_keys = {puzzle_key(row["target_nums"]) for row in unsolvable_rows}
    unsolvable_test_keys = {puzzle_key(row["target_nums"]) for row in unsolvable_test_rows}
    unsolvable_overlap = len(unsolvable_train_keys & unsolvable_test_keys)
    if unsolvable_overlap:
        raise ValueError(f"unsolvable train/test overlap detected: {unsolvable_overlap}")

    grpo_rows = [
        {
            "target_nums": [int(num) for num in row["target_nums"]],
            "target_value": row.get("target_value", 24),
            "solvable": True,
            "source": row.get("source", "solvable_train"),
        }
        for row in solvable_rows
    ] + unsolvable_rows
    random.Random(seed).shuffle(grpo_rows)

    os.makedirs(output_dir, exist_ok=True)
    grpo_path = os.path.join(output_dir, "grpo_train.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    hashes_path = os.path.join(output_dir, "sha256.txt")
    write_jsonl(grpo_path, grpo_rows)

    summary = {
        "version": "v5_continue_grpo_mixed100",
        "seed": seed,
        "source_train_path": train_path,
        "solvable_test_path": solvable_test_path,
        "unsolvable_test_path": unsolvable_test_path,
        "solvable_train_puzzles": len(solvable_rows),
        "unsolvable_candidates_after_test_exclusion": candidate_count,
        "unsolvable_train_puzzles": len(unsolvable_rows),
        "solvable_test_overlap": solvable_overlap,
        "unsolvable_test_overlap": unsolvable_overlap,
        "grpo_rows": len(grpo_rows),
        "grpo_path": grpo_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(hashes_path, "w", encoding="utf-8") as f:
        f.write(f"{sha256_file(grpo_path)}  grpo_train.jsonl\n")
        f.write(f"{sha256_file(summary_path)}  summary.json\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v5 mixed GRPO data for conservative continuation from the v3 adapter."
    )
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--solvable-test-path", default=SOLVABLE_TEST_PATH)
    parser.add_argument("--unsolvable-test-path", default=UNSOLVABLE_TEST_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--unsolvable-size", type=int, default=DEFAULT_UNSOLVABLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    prepare_v5_data(
        train_path=args.train_path,
        solvable_test_path=args.solvable_test_path,
        unsolvable_test_path=args.unsolvable_test_path,
        output_dir=args.output_dir,
        unsolvable_size=args.unsolvable_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
