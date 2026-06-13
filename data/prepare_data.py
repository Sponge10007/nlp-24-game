import argparse
import json
import os
import re
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


def parse_solved_rate(row: dict[str, Any]) -> float | None:
    value = get_first(
        row,
        (
            "solved_rate",
            "solve_rate",
            "success_rate",
            "solved rate",
            "Solved Rate",
            "solved",
        ),
    )
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def prepare_nlile_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[tuple[int, ...]]]:
    print("1. Loading nlile/24-game for train and unsolvable holdout...")
    nlile_ds = load_dataset("nlile/24-game", split="train")

    train_samples: list[dict[str, Any]] = []
    unsolvable_samples: list[dict[str, Any]] = []
    train_keys: set[tuple[int, ...]] = set()

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
            train_samples.append(sample)
            train_keys.add(puzzle_key(nums))
        else:
            unsolvable_samples.append(sample)

    write_jsonl(TRAIN_PATH, train_samples)
    write_jsonl(UNSOLVABLE_TEST_PATH, unsolvable_samples)

    print(f"   Wrote {TRAIN_PATH}: {len(train_samples)} solvable training cases.")
    print(f"   Wrote {UNSOLVABLE_TEST_PATH}: {len(unsolvable_samples)} unsolvable holdout cases.")
    if len(train_samples) != 1262:
        print(f"   Warning: expected 1262 solvable cases, got {len(train_samples)}.")

    return train_samples, unsolvable_samples, train_keys


def prepare_tot_data(train_keys: set[tuple[int, ...]], low_solved_rate_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    print("\n2. Loading test-time-compute/game-of-24 for non-overlapping tests...")
    tot_ds = load_dataset("test-time-compute/game-of-24", split="train")

    all_nonoverlap: list[dict[str, Any]] = []
    hard_samples: list[dict[str, Any]] = []
    overlap_count = 0

    for index, row in enumerate(tot_ds):
        row = dict(row)
        nums = parse_nums(row, ("Puzzles", "puzzle", "Puzzle", "numbers", "nums"))
        if len(nums) != 4:
            continue

        key = puzzle_key(nums)
        if key in train_keys:
            overlap_count += 1
            continue

        sample = {
            "target_nums": nums,
            "target_value": 24,
            "solvable": True,
            "source": "test-time-compute/game-of-24",
            "source_index": index,
        }
        solved_rate = parse_solved_rate(row)
        if solved_rate is not None:
            sample["solved_rate"] = solved_rate

        all_nonoverlap.append(sample)
        if 900 <= index < 1000:
            hard_samples.append(sample)

    rate_aware = [sample for sample in all_nonoverlap if "solved_rate" in sample]
    if rate_aware:
        low_solved_rate = sorted(rate_aware, key=lambda sample: sample["solved_rate"])[:low_solved_rate_size]
    else:
        low_solved_rate = hard_samples[:low_solved_rate_size]

    if hard_samples and "solved_rate" in hard_samples[0]:
        hard_samples = sorted(hard_samples, key=lambda sample: sample["solved_rate"])

    write_jsonl(TEST_ALL_NONOVERLAP_PATH, all_nonoverlap)
    write_jsonl(TEST_HARD_PATH, hard_samples)
    write_jsonl(TEST_LOW_SOLVED_RATE_PATH, low_solved_rate)
    write_jsonl(TEST_PATH, hard_samples)

    print(f"   Wrote {TEST_ALL_NONOVERLAP_PATH}: {len(all_nonoverlap)} cases.")
    print(f"   Wrote {TEST_HARD_PATH}: {len(hard_samples)} non-overlapping paper-hard cases.")
    print(f"   Wrote {TEST_LOW_SOLVED_RATE_PATH}: {len(low_solved_rate)} low solved-rate cases.")
    print(f"   Wrote {TEST_PATH}: alias of paper-hard split for backward compatibility.")
    print(f"   Removed {overlap_count} overlapping ToT cases.")

    return all_nonoverlap, hard_samples, low_solved_rate, overlap_count


def prepare_countdown_ood(max_samples: int) -> list[dict[str, Any]]:
    print("\n3. Loading Jiayi-Pan/Countdown-Tasks-3to4 for optional OOD extension...")
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


def prepare_data(low_solved_rate_size: int = 100, with_countdown: bool = False, countdown_size: int = 200) -> None:
    os.makedirs("data", exist_ok=True)

    train_samples, unsolvable_samples, train_keys = prepare_nlile_data()
    all_nonoverlap, hard_samples, low_solved_rate, overlap_count = prepare_tot_data(train_keys, low_solved_rate_size)
    countdown_samples = prepare_countdown_ood(countdown_size) if with_countdown else []

    summary = {
        "train": {"path": TRAIN_PATH, "count": len(train_samples), "solvable": True},
        "unsolvable_holdout": {"path": UNSOLVABLE_TEST_PATH, "count": len(unsolvable_samples), "solvable": False},
        "tot_all_nonoverlap": {"path": TEST_ALL_NONOVERLAP_PATH, "count": len(all_nonoverlap)},
        "tot_paper_hard_900_1000": {"path": TEST_HARD_PATH, "count": len(hard_samples), "alias": TEST_PATH},
        "tot_low_solved_rate": {"path": TEST_LOW_SOLVED_RATE_PATH, "count": len(low_solved_rate)},
        "tot_overlap_removed": overlap_count,
        "countdown_ood": {"path": COUNTDOWN_OOD_PATH, "count": len(countdown_samples), "enabled": with_countdown},
    }
    write_summary(summary)
    print(f"\nWrote {SUMMARY_PATH}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/test splits for the 24-game GRPO project.")
    parser.add_argument("--low-solved-rate-size", type=int, default=100, help="Number of lowest solved-rate ToT cases to keep.")
    parser.add_argument("--with-countdown", action="store_true", help="Also create Countdown 3-4 numbers OOD extension data.")
    parser.add_argument("--countdown-size", type=int, default=200, help="Maximum countdown OOD cases to export.")
    args = parser.parse_args()

    prepare_data(
        low_solved_rate_size=args.low_solved_rate_size,
        with_countdown=args.with_countdown,
        countdown_size=args.countdown_size,
    )


if __name__ == "__main__":
    main()
