import argparse
import json
import os
import re
import random
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Any

from datasets import load_dataset


TRAIN_PATH = "data/train.jsonl"
TEST_PATH = "data/test.jsonl"
TEST_ALL_NONOVERLAP_PATH = "data/test_all_nonoverlap.jsonl"
TEST_HARD_PATH = "data/test_hard_900_1000.jsonl"
TEST_LOW_SOLVED_RATE_PATH = "data/test_low_solved_rate.jsonl"
UNSOLVABLE_TEST_PATH = "data/unsolvable_test.jsonl"
COUNTDOWN_OOD_PATH = "data/countdown_ood.jsonl"
SUMMARY_PATH = "data/dataset_summary.json"
HARD_START_INDEX = 900
HARD_END_INDEX = 1000
DEFAULT_UNSOLVABLE_SEED = 20240613


def parse_nums(row: dict[str, Any], keys: tuple[str, ...]) -> list[int]:
    for key in keys:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, (list, tuple)):
            nums = [int(x) for x in value]
        else:
            nums = [int(x) for x in re.findall(r"\d+", str(value))]
        if nums:
            return nums
    return []


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def puzzle_key(nums: list[int]) -> tuple[int, ...]:
    return tuple(sorted(nums))


def get_first(row: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def parse_solved_rate(row_or_value: Any) -> float | None:
    if isinstance(row_or_value, dict):
        value = get_first(
            row_or_value,
            (
                "solved_rate",
                "solve_rate",
                "success_rate",
                "solved rate",
                "Solved rate",
                "Solved Rate",
                "solved",
            ),
        )
    else:
        value = row_or_value

    if value is None:
        return None

    is_percent = False
    if isinstance(value, str):
        value = value.strip()
        is_percent = value.endswith("%")
        if is_percent:
            value = value[:-1].strip()

    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if is_percent or rate > 1.0:
        rate /= 100.0
    return rate


def parse_target_value(row: dict[str, Any], default: int = 24) -> int | float:
    value = get_first(row, ("target", "Target", "target_value", "answer", "goal"), default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return int(numeric) if numeric.is_integer() else numeric


def write_jsonl(path: str, samples: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_summary(summary: dict[str, Any]) -> None:
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _unique_by_puzzle(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, ...]] = set()
    unique_samples: list[dict[str, Any]] = []
    for sample in samples:
        key = puzzle_key(sample["target_nums"])
        if key in seen:
            continue
        seen.add(key)
        unique_samples.append(sample)
    return unique_samples


def select_tot_splits(
    all_samples: list[dict[str, Any]],
    low_solved_rate_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    hard_samples = [
        sample
        for sample in all_samples
        if HARD_START_INDEX <= int(sample["source_index"]) < HARD_END_INDEX
    ]

    rate_aware = [sample for sample in all_samples if "solved_rate" in sample]
    if rate_aware:
        low_solved_rate = sorted(
            rate_aware,
            key=lambda sample: (sample["solved_rate"], int(sample["source_index"])),
        )[:low_solved_rate_size]
    else:
        low_solved_rate = all_samples[-low_solved_rate_size:] if low_solved_rate_size else []

    selected_tests = _unique_by_puzzle(hard_samples + low_solved_rate)
    return hard_samples, low_solved_rate, selected_tests


def prepare_tot_data(low_solved_rate_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    print("1. Loading test-time-compute/game-of-24 for held-out tests...")
    tot_ds = load_dataset("test-time-compute/game-of-24", split="train")

    all_samples: list[dict[str, Any]] = []
    for index, row in enumerate(tot_ds):
        row = dict(row)
        nums = parse_nums(row, ("Puzzles", "puzzle", "Puzzle", "numbers", "nums"))
        if len(nums) != 4:
            continue

        sample = {
            "target_nums": nums,
            "target_value": 24,
            "solvable": True,
            "source": "test-time-compute/game-of-24",
            "source_index": index,
        }
        rank = get_first(row, ("Rank", "rank"))
        if rank is not None:
            sample["rank"] = int(rank)
        solved_rate = parse_solved_rate(row)
        if solved_rate is not None:
            sample["solved_rate"] = solved_rate

        all_samples.append(sample)

    hard_samples, low_solved_rate, selected_tests = select_tot_splits(all_samples, low_solved_rate_size)

    write_jsonl(TEST_ALL_NONOVERLAP_PATH, selected_tests)
    write_jsonl(TEST_HARD_PATH, hard_samples)
    write_jsonl(TEST_LOW_SOLVED_RATE_PATH, low_solved_rate)
    write_jsonl(TEST_PATH, hard_samples)

    hard_keys = {puzzle_key(sample["target_nums"]) for sample in hard_samples}
    low_keys = {puzzle_key(sample["target_nums"]) for sample in low_solved_rate}
    stats = {
        "tot_raw_rows": len(tot_ds),
        "tot_valid_rows": len(all_samples),
        "hard_low_overlap": len(hard_keys & low_keys),
        "selected_test_key_count": len({puzzle_key(sample["target_nums"]) for sample in selected_tests}),
    }

    print(f"   Wrote {TEST_HARD_PATH}: {len(hard_samples)} paper-hard cases.")
    print(f"   Wrote {TEST_LOW_SOLVED_RATE_PATH}: {len(low_solved_rate)} low solved-rate cases.")
    print(f"   Wrote {TEST_ALL_NONOVERLAP_PATH}: {len(selected_tests)} held-out unique test cases.")
    print(f"   Wrote {TEST_PATH}: alias of paper-hard split for backward compatibility.")

    return selected_tests, hard_samples, low_solved_rate, stats


def prepare_nlile_data(test_keys: set[tuple[int, ...]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print("\n2. Loading nlile/24-game for train data...")
    nlile_ds = load_dataset("nlile/24-game", split="train")

    train_samples: list[dict[str, Any]] = []
    dataset_unsolvable_samples: list[dict[str, Any]] = []
    raw_solvable_count = 0
    withheld_for_test_count = 0

    for row in nlile_ds:
        row = dict(row)
        nums = parse_nums(row, ("puzzle", "Puzzle", "numbers", "nums"))
        if len(nums) != 4:
            continue

        solvable = parse_bool(get_first(row, ("solvable", "Solvable"), True))
        sample = {
            "target_nums": nums,
            "target_value": 24,
            "solvable": solvable,
            "source": "nlile/24-game",
        }

        if solvable:
            raw_solvable_count += 1
            if puzzle_key(nums) in test_keys:
                withheld_for_test_count += 1
                continue
            train_samples.append(sample)
        else:
            dataset_unsolvable_samples.append(sample)

    write_jsonl(TRAIN_PATH, train_samples)

    print(f"   Wrote {TRAIN_PATH}: {len(train_samples)} solvable training cases.")
    print(f"   Withheld {withheld_for_test_count} nlile cases that are reserved for ToT tests.")
    if dataset_unsolvable_samples:
        print(f"   Found {len(dataset_unsolvable_samples)} nlile unsolvable rows; local enum holdout is still used.")
    else:
        print("   Found 0 nlile unsolvable rows; local enum holdout is used.")

    stats = {
        "nlile_raw_rows": len(nlile_ds),
        "nlile_raw_solvable": raw_solvable_count,
        "nlile_raw_unsolvable": len(dataset_unsolvable_samples),
        "withheld_for_test": withheld_for_test_count,
    }
    return train_samples, stats


def can_make_target(nums: list[int] | tuple[int, ...], target_value: int | float = 24) -> bool:
    target = Fraction(target_value)
    start = tuple(sorted(Fraction(num) for num in nums))

    @lru_cache(maxsize=None)
    def search(values: tuple[Fraction, ...]) -> bool:
        if len(values) == 1:
            return values[0] == target

        value_count = len(values)
        for left_index in range(value_count):
            for right_index in range(left_index + 1, value_count):
                left = values[left_index]
                right = values[right_index]
                rest = [
                    values[index]
                    for index in range(value_count)
                    if index not in {left_index, right_index}
                ]

                candidates = [left + right, left - right, right - left, left * right]
                if right:
                    candidates.append(left / right)
                if left:
                    candidates.append(right / left)

                for candidate in candidates:
                    next_values = tuple(sorted(rest + [candidate]))
                    if search(next_values):
                        return True
        return False

    return search(start)


def enumerate_unsolvable_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for nums in combinations_with_replacement(range(1, 14), 4):
        if can_make_target(nums):
            continue
        samples.append(
            {
                "target_nums": list(nums),
                "target_value": 24,
                "solvable": False,
                "source": "local_enum_1_13_unsolvable",
            }
        )
    return samples


def prepare_unsolvable_data(max_samples: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print("\n3. Enumerating local unsolvable 24-game holdout...")
    all_unsolvable = enumerate_unsolvable_samples()
    selected = list(all_unsolvable)
    random.Random(seed).shuffle(selected)
    if max_samples >= 0:
        selected = selected[:max_samples]

    write_jsonl(UNSOLVABLE_TEST_PATH, selected)
    print(f"   Enumerated {len(all_unsolvable)} unsolvable 1-13 combinations.")
    print(f"   Wrote {UNSOLVABLE_TEST_PATH}: {len(selected)} unsolvable holdout cases.")

    stats = {
        "enumerated_total": len(all_unsolvable),
        "exported": len(selected),
        "seed": seed,
    }
    return selected, stats


def prepare_countdown_ood(max_samples: int) -> list[dict[str, Any]]:
    print("\n4. Loading Jiayi-Pan/Countdown-Tasks-3to4 for optional OOD extension...")
    samples: list[dict[str, Any]] = []
    try:
        ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    except Exception as exc:
        print(f"   Skipped countdown OOD data: {exc}")
        write_jsonl(COUNTDOWN_OOD_PATH, samples)
        return samples

    for row in ds:
        row = dict(row)
        nums = parse_nums(row, ("nums", "numbers", "input", "inputs", "cards"))
        if not 3 <= len(nums) <= 4:
            continue
        target_value = parse_target_value(row, default=24)
        samples.append(
            {
                "target_nums": nums,
                "target_value": target_value,
                "solvable": True,
                "source": "Jiayi-Pan/Countdown-Tasks-3to4",
            }
        )
        if len(samples) >= max_samples:
            break

    write_jsonl(COUNTDOWN_OOD_PATH, samples)
    print(f"   Wrote {COUNTDOWN_OOD_PATH}: {len(samples)} optional OOD cases.")
    return samples


def prepare_data(
    low_solved_rate_size: int = 100,
    with_countdown: bool = False,
    countdown_size: int = 200,
    unsolvable_size: int = 100,
    unsolvable_seed: int = DEFAULT_UNSOLVABLE_SEED,
) -> None:
    os.makedirs("data", exist_ok=True)

    selected_tests, hard_samples, low_solved_rate, tot_stats = prepare_tot_data(low_solved_rate_size)
    test_keys = {puzzle_key(sample["target_nums"]) for sample in selected_tests}
    train_samples, nlile_stats = prepare_nlile_data(test_keys)
    unsolvable_samples, unsolvable_stats = prepare_unsolvable_data(unsolvable_size, unsolvable_seed)
    countdown_samples = prepare_countdown_ood(countdown_size) if with_countdown else []

    summary = {
        "train": {
            "path": TRAIN_PATH,
            "count": len(train_samples),
            "solvable": True,
            "source": "nlile/24-game",
            **nlile_stats,
        },
        "unsolvable_holdout": {
            "path": UNSOLVABLE_TEST_PATH,
            "count": len(unsolvable_samples),
            "solvable": False,
            "source": "local_enum_1_13_unsolvable",
            **unsolvable_stats,
        },
        "tot_all_nonoverlap": {
            "path": TEST_ALL_NONOVERLAP_PATH,
            "count": len(selected_tests),
            "description": "unique union of hard split and lowest solved-rate ToT cases",
        },
        "tot_paper_hard_900_1000": {
            "path": TEST_HARD_PATH,
            "count": len(hard_samples),
            "alias": TEST_PATH,
            "start_index": HARD_START_INDEX,
            "end_index_exclusive": HARD_END_INDEX,
        },
        "tot_low_solved_rate": {
            "path": TEST_LOW_SOLVED_RATE_PATH,
            "count": len(low_solved_rate),
            "requested": low_solved_rate_size,
        },
        "tot_stats": tot_stats,
        "countdown_ood": {"path": COUNTDOWN_OOD_PATH, "count": len(countdown_samples), "enabled": with_countdown},
    }
    write_summary(summary)
    print(f"\nWrote {SUMMARY_PATH}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/test splits for the 24-game GRPO project.")
    parser.add_argument("--low-solved-rate-size", type=int, default=100, help="Number of lowest solved-rate ToT cases to keep.")
    parser.add_argument("--unsolvable-size", type=int, default=100, help="Number of locally enumerated unsolvable cases to export. Use -1 for all.")
    parser.add_argument("--unsolvable-seed", type=int, default=DEFAULT_UNSOLVABLE_SEED, help="Shuffle seed for the local unsolvable holdout.")
    parser.add_argument("--with-countdown", action="store_true", help="Also create Countdown 3-4 numbers OOD extension data.")
    parser.add_argument("--countdown-size", type=int, default=200, help="Maximum countdown OOD cases to export.")
    args = parser.parse_args()

    prepare_data(
        low_solved_rate_size=args.low_solved_rate_size,
        with_countdown=args.with_countdown,
        countdown_size=args.countdown_size,
        unsolvable_size=args.unsolvable_size,
        unsolvable_seed=args.unsolvable_seed,
    )


if __name__ == "__main__":
    main()
