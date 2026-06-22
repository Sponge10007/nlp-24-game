import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from src.rejection_sampling import (
    build_completion_from_deepseek_response,
    build_prompt,
    build_sft_row,
    select_rejection_sample,
)


API_KEY_ENV = "DEEPSEEK_API_KEY"
BASE_URL = "https://api.deepseek.com"
API_MODEL = "deepseek-reasoner"
INPUT_DATA_PATH = "data/train.jsonl"
OUTPUT_DATA_PATH = "data/rejection_sft_train.jsonl"
DEFAULT_LIMIT = 500
MAX_WORKERS = 8


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SFT data with DeepSeek API rejection sampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-key-env", default=API_KEY_ENV)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-model", default=API_MODEL)
    parser.add_argument("--input-data-path", default=INPUT_DATA_PATH)
    parser.add_argument("--output-data-path", default=OUTPUT_DATA_PATH)
    parser.add_argument(
        "--limit",
        "--max-cases",
        dest="limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of input cases to process.",
    )
    parser.add_argument(
        "--max-workers",
        "--num-workers",
        dest="max_workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of cases to process concurrently.",
    )

    parser.add_argument(
        "--attempts-per-case",
        "--samples-per-case",
        dest="attempts_per_case",
        type=int,
        default=16,
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--require-r1-format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep valid rows already present in the output file and only request missing puzzles.",
    )
    return parser.parse_args()


def build_client(args: argparse.Namespace):
    from openai import OpenAI

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing DeepSeek API key. Set environment variable {args.api_key_env}.")
    return OpenAI(api_key=api_key, base_url=args.base_url)


def call_deepseek(client, target_nums: list[int], target_value: int | float, args: argparse.Namespace) -> str:
    messages = build_prompt(target_nums, target_value)
    response = client.chat.completions.create(
        model=args.api_model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    message = response.choices[0].message
    reasoning_content = getattr(message, "reasoning_content", None)
    content = getattr(message, "content", None)
    return build_completion_from_deepseek_response(reasoning_content, content)


def sample_case(client, case_index: int, case: dict[str, Any], args: argparse.Namespace) -> tuple[int, dict[str, Any] | None, Counter[str]]:
    counts: Counter[str] = Counter()
    target_nums = case["target_nums"]
    target_value = case.get("target_value", 24)
    solvable = bool(case.get("solvable", True))

    completions: list[str] = []
    sample = None
    for _ in range(args.attempts_per_case):
        try:
            completions.append(call_deepseek(client, target_nums, target_value, args))
        except Exception as exc:
            counts["api_error"] += 1
            tqdm.write(f"API error for nums={target_nums}, target={target_value}: {exc}")
        sample = select_rejection_sample(
            completions,
            target_nums,
            target_value=target_value,
            solvable=solvable,
            require_r1_format=args.require_r1_format,
        )
        if sample is not None:
            break
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if sample is None:
        counts["rejected_all"] += 1
        for completion in completions:
            rejected = select_rejection_sample(
                [completion],
                target_nums,
                target_value=target_value,
                solvable=solvable,
                require_r1_format=False,
            )
            if rejected is None:
                counts["rejected_invalid"] += 1
        return case_index, None, counts

    counts["accepted"] += 1
    counts[sample.source] += 1
    counts[sample.code] += 1
    return case_index, build_sft_row(case, sample), counts


def build_rejection_sft_rows(client, cases: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows_by_index: dict[int, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [
            executor.submit(sample_case, client, case_index, case, args)
            for case_index, case in enumerate(cases)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rejection sampling"):
            case_index, row, case_counts = future.result()
            counts.update(case_counts)
            if row is not None:
                rows_by_index[case_index] = row

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    return rows, counts


def case_key(case: dict[str, Any]) -> tuple[tuple[int, ...], float]:
    nums = tuple(sorted(int(num) for num in case["target_nums"]))
    return nums, float(case.get("target_value", 24))


def merge_rows_in_input_order(
    cases: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key = {case_key(row): row for row in existing_rows}
    rows_by_key.update({case_key(row): row for row in new_rows})
    return [rows_by_key[case_key(case)] for case in cases if case_key(case) in rows_by_key]


def valid_existing_rows(rows: list[dict[str, Any]], require_r1_format: bool) -> list[dict[str, Any]]:
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        completion = row.get("completion", "")
        sample = select_rejection_sample(
            [completion],
            row["target_nums"],
            target_value=row.get("target_value", 24),
            solvable=bool(row.get("solvable", True)),
            require_r1_format=require_r1_format,
        )
        if sample is not None:
            valid_rows.append(row)
    return valid_rows


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.input_data_path)
    if args.limit >= 0:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases from {args.input_data_path}.")

    existing_rows: list[dict[str, Any]] = []
    if args.resume and os.path.exists(args.output_data_path):
        existing_rows = valid_existing_rows(
            load_jsonl(args.output_data_path),
            require_r1_format=args.require_r1_format,
        )
    existing_keys = {case_key(row) for row in existing_rows}
    pending_cases = [case for case in cases if case_key(case) not in existing_keys]
    print(
        f"Resume state: {len(existing_rows)} existing rows, "
        f"{len(pending_cases)} puzzles still need sampling."
    )

    if pending_cases:
        client = build_client(args)
        new_rows, counts = build_rejection_sft_rows(client, pending_cases, args)
    else:
        new_rows, counts = [], Counter()
    rows = merge_rows_in_input_order(cases, existing_rows, new_rows)

    write_jsonl(args.output_data_path, rows)
    summary_path = os.path.splitext(args.output_data_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": args.input_data_path,
                "output": args.output_data_path,
                "total_cases": len(cases),
                "existing_rows": len(existing_rows),
                "requested_cases": len(pending_cases),
                "new_rows": len(new_rows),
                "written_rows": len(rows),
                "attempts_per_case": args.attempts_per_case,
                "max_workers": args.max_workers,
                "api_model": args.api_model,
                "counts": dict(counts),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(rows)} SFT rows to {args.output_data_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Counts: {dict(counts)}")


if __name__ == "__main__":
    main()
