import argparse
import json
import os
import re
import random
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Any


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
DEFAULT_SPLIT_SEED = 20240613
COUNTDOWN_SOURCE = "Jiayi-Pan/Countdown-Tasks-3to4"


def load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset as hf_load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'datasets' package is required to download Hugging Face datasets. "
            "Install requirements.txt before running data preparation."
        ) from exc
    return hf_load_dataset(*args, **kwargs)


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


def case_key(sample: dict[str, Any]) -> tuple[tuple[int, ...], int | float]:
    return puzzle_key(sample["target_nums"]), sample.get("target_value", 24)


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


def _unique_by_case(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[int, ...], int | float]] = set()
    unique_samples: list[dict[str, Any]] = []
    for sample in samples:
        key = case_key(sample)
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


def normalize_countdown_row(row: dict[str, Any], source_index: int | None = None) -> dict[str, Any] | None:
    nums = parse_nums(row, ("nums", "numbers", "input", "inputs", "cards", "target_nums"))
    if not 3 <= len(nums) <= 4:
        return None

    sample = {
        "target_nums": nums,
        "target_value": parse_target_value(row, default=24),
        "solvable": parse_bool(get_first(row, ("solvable", "Solvable"), True)),
        "source": COUNTDOWN_SOURCE,
    }
    if source_index is not None:
        sample["source_index"] = source_index
    return sample


def load_countdown_samples(max_samples: int = -1) -> list[dict[str, Any]]:
    print(f"1. Loading {COUNTDOWN_SOURCE} for Countdown training data...")
    ds = load_dataset(COUNTDOWN_SOURCE, split="train")
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(ds):
        sample = normalize_countdown_row(dict(row), source_index=index)
        if sample is None:
            continue
        samples.append(sample)
        if max_samples >= 0 and len(samples) >= max_samples:
            break

    samples = _unique_by_case(samples)
    print(f"   Loaded {len(samples)} unique Countdown cases.")
    return samples


def split_countdown_samples(
    samples: list[dict[str, Any]],
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(_unique_by_case(samples))
    random.Random(seed).shuffle(shuffled)
    test_count = min(max(test_size, 0), len(shuffled))
    test_samples = shuffled[:test_count]
    train_samples = shuffled[test_count:]
    return train_samples, test_samples


def generate_random_unsolvable_samples(
    count: int,
    seed: int,
    *,
    exclude_keys: set[tuple[tuple[int, ...], int | float]] | None = None,
    min_num: int = 1,
    max_num: int = 13,
    min_target: int = 1,
    max_target: int = 100,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    rng = random.Random(seed)
    excluded = set(exclude_keys or set())
    samples: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(10_000, count * 1_000)

    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        num_count = rng.choice((3, 4))
        nums = [rng.randint(min_num, max_num) for _ in range(num_count)]
        target_value = rng.randint(min_target, max_target)
        sample = {
            "target_nums": nums,
            "target_value": target_value,
            "solvable": False,
            "source": "local_random_countdown_unsolvable",
        }
        key = case_key(sample)
        if key in excluded:
            continue
        if can_make_target(nums, target_value):
            continue
        excluded.add(key)
        samples.append(sample)

    if len(samples) < count:
        raise RuntimeError(f"Only generated {len(samples)} unsolvable samples after {attempts} attempts.")
    return samples


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
    print(f"\n4. Loading {COUNTDOWN_SOURCE} for optional OOD extension...")
    samples: list[dict[str, Any]] = []
    try:
        ds = load_dataset(COUNTDOWN_SOURCE, split="train")
    except Exception as exc:
        print(f"   Skipped countdown OOD data: {exc}")
        write_jsonl(COUNTDOWN_OOD_PATH, samples)
        return samples

    for index, row in enumerate(ds):
        sample = normalize_countdown_row(dict(row), source_index=index)
        if sample is None:
            continue
        samples.append(sample)
        if len(samples) >= max_samples:
            break

    samples = _unique_by_case(samples)
    write_jsonl(COUNTDOWN_OOD_PATH, samples)
    print(f"   Wrote {COUNTDOWN_OOD_PATH}: {len(samples)} optional OOD cases.")
    return samples


def prepare_countdown_data(
    countdown_size: int = -1,
    test_size: int = 200,
    split_seed: int = DEFAULT_SPLIT_SEED,
    unsolvable_train_ratio: float = 0.1,
    unsolvable_size: int = 100,
    unsolvable_seed: int = DEFAULT_UNSOLVABLE_SEED,
) -> None:
    os.makedirs("data", exist_ok=True)

    countdown_samples = load_countdown_samples(countdown_size)
    train_samples, test_samples = split_countdown_samples(countdown_samples, test_size, split_seed)
    reserved_keys = {case_key(sample) for sample in countdown_samples}

    unsolvable_train_count = round(len(train_samples) * unsolvable_train_ratio)
    unsolvable_train_samples = generate_random_unsolvable_samples(
        unsolvable_train_count,
        unsolvable_seed,
        exclude_keys=reserved_keys,
    )
    reserved_keys.update(case_key(sample) for sample in unsolvable_train_samples)

    unsolvable_test_samples = generate_random_unsolvable_samples(
        unsolvable_size,
        unsolvable_seed + 1,
        exclude_keys=reserved_keys,
    )

    train_samples_with_unsolvable = train_samples + unsolvable_train_samples
    random.Random(split_seed).shuffle(train_samples_with_unsolvable)

    write_jsonl(TRAIN_PATH, train_samples_with_unsolvable)
    write_jsonl(TEST_PATH, test_samples)
    write_jsonl(UNSOLVABLE_TEST_PATH, unsolvable_test_samples)

    summary = {
        "task": "countdown",
        "source": COUNTDOWN_SOURCE,
        "train": {
            "path": TRAIN_PATH,
            "count": len(train_samples_with_unsolvable),
            "solvable_count": len(train_samples),
            "unsolvable_count": len(unsolvable_train_samples),
            "unsolvable_train_ratio": unsolvable_train_ratio,
        },
        "test": {
            "path": TEST_PATH,
            "count": len(test_samples),
            "requested": test_size,
        },
        "unsolvable_holdout": {
            "path": UNSOLVABLE_TEST_PATH,
            "count": len(unsolvable_test_samples),
            "solvable": False,
            "source": "local_random_countdown_unsolvable",
            "seed": unsolvable_seed + 1,
        },
        "split_seed": split_seed,
        "countdown_requested": countdown_size,
        "countdown_loaded": len(countdown_samples),
    }
    write_summary(summary)

    print(f"   Wrote {TRAIN_PATH}: {len(train_samples_with_unsolvable)} training cases.")
    print(f"   Wrote {TEST_PATH}: {len(test_samples)} Countdown test cases.")
    print(f"   Wrote {UNSOLVABLE_TEST_PATH}: {len(unsolvable_test_samples)} unsolvable holdout cases.")
    print(f"\nWrote {SUMMARY_PATH}.")


def prepare_game24_data(
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


def prepare_data(
    task: str = "countdown",
    low_solved_rate_size: int = 100,
    with_countdown: bool = False,
    countdown_size: int = -1,
    test_size: int = 200,
    split_seed: int = DEFAULT_SPLIT_SEED,
    unsolvable_train_ratio: float = 0.1,
    unsolvable_size: int = 100,
    unsolvable_seed: int = DEFAULT_UNSOLVABLE_SEED,
) -> None:
    if task == "countdown":
        prepare_countdown_data(
            countdown_size=countdown_size,
            test_size=test_size,
            split_seed=split_seed,
            unsolvable_train_ratio=unsolvable_train_ratio,
            unsolvable_size=unsolvable_size,
            unsolvable_seed=unsolvable_seed,
        )
        return
    if task == "game24":
        prepare_game24_data(
            low_solved_rate_size=low_solved_rate_size,
            with_countdown=with_countdown,
            countdown_size=countdown_size if countdown_size >= 0 else 200,
            unsolvable_size=unsolvable_size,
            unsolvable_seed=unsolvable_seed,
        )
        return
    raise ValueError(f"unsupported task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/test splits for Countdown arithmetic GRPO.")
    parser.add_argument("--task", choices=("countdown", "game24"), default="countdown", help="Dataset pipeline to run.")
    parser.add_argument("--low-solved-rate-size", type=int, default=100, help="Number of lowest solved-rate ToT cases to keep.")
    parser.add_argument("--unsolvable-size", type=int, default=100, help="Number of locally enumerated unsolvable cases to export. Use -1 for all.")
    parser.add_argument("--unsolvable-seed", type=int, default=DEFAULT_UNSOLVABLE_SEED, help="Shuffle seed for the local unsolvable holdout.")
    parser.add_argument("--unsolvable-train-ratio", type=float, default=0.1, help="Fraction of Countdown train samples to add as local unsolvable cases.")
    parser.add_argument("--with-countdown", action="store_true", help="Also create Countdown 3-4 numbers OOD extension data.")
    parser.add_argument("--countdown-size", type=int, default=-1, help="Maximum Countdown cases to load. Use -1 for all.")
    parser.add_argument("--test-size", type=int, default=200, help="Number of Countdown cases to reserve for data/test.jsonl.")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED, help="Deterministic seed for Countdown train/test split.")
    args = parser.parse_args()

    prepare_data(
        task=args.task,
        low_solved_rate_size=args.low_solved_rate_size,
        with_countdown=args.with_countdown,
        countdown_size=args.countdown_size,
        test_size=args.test_size,
        split_seed=args.split_seed,
        unsolvable_train_ratio=args.unsolvable_train_ratio,
        unsolvable_size=args.unsolvable_size,
        unsolvable_seed=args.unsolvable_seed,
    )


if __name__ == "__main__":
    main()
