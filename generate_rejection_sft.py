import argparse
import json
import os
from collections import Counter
from typing import Any

from tqdm import tqdm

from src.rejection_sampling import (
    build_prompt,
    build_sft_row,
    build_solver_fallback_sample,
    select_rejection_sample,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
INPUT_DATA_PATH = "data/train.jsonl"
OUTPUT_DATA_PATH = "data/rejection_sft_train.jsonl"


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
    parser = argparse.ArgumentParser(description="Build SFT data with model rejection sampling.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--adapter-path", default="", help="Optional LoRA adapter to sample from.")
    parser.add_argument("--input-data-path", default=INPUT_DATA_PATH)
    parser.add_argument("--output-data-path", default=OUTPUT_DATA_PATH)
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--attempts-per-case", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--require-r1-format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-to-solver", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"1. Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs = {"device_map": "auto"}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.adapter_path:
        print(f"2. Loading sampling adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print("2. Sampling from base model.")

    model.eval()
    return tokenizer, model


def generate_attempts(tokenizer, model, target_nums: list[int], target_value: int | float, args: argparse.Namespace) -> list[str]:
    import torch

    messages = build_prompt(target_nums, target_value)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    prompt_len = model_inputs.input_ids.shape[-1]
    completions: list[str] = []
    with torch.no_grad():
        for _ in range(args.attempts_per_case):
            generated_ids = model.generate(**model_inputs, **generation_kwargs)
            completions.extend(
                tokenizer.batch_decode(
                    [output_ids[prompt_len:] for output_ids in generated_ids],
                    skip_special_tokens=True,
                )
            )
    return [completion.strip() for completion in completions]


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.input_data_path)
    if args.limit >= 0:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases from {args.input_data_path}.")

    tokenizer, model = load_model(args)

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for case in tqdm(cases, desc="Rejection sampling"):
        target_nums = case["target_nums"]
        target_value = case.get("target_value", 24)
        solvable = bool(case.get("solvable", True))

        completions = generate_attempts(tokenizer, model, target_nums, target_value, args)
        sample = select_rejection_sample(
            completions,
            target_nums,
            target_value=target_value,
            solvable=solvable,
            require_r1_format=args.require_r1_format,
        )

        if sample is None:
            counts["rejected_all"] += 1
            if not args.fallback_to_solver:
                continue
            sample = build_solver_fallback_sample(target_nums, target_value=target_value, solvable=solvable)

        counts[sample.source] += 1
        counts[sample.code] += 1
        rows.append(build_sft_row(case, sample))

    write_jsonl(args.output_data_path, rows)
    summary_path = os.path.splitext(args.output_data_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": args.input_data_path,
                "output": args.output_data_path,
                "total_cases": len(cases),
                "written_rows": len(rows),
                "attempts_per_case": args.attempts_per_case,
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
