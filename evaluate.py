import argparse
import json
import os
from collections import Counter
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.game24 import CORRECT, UNSOLVABLE_CLAIM, extract_answer, has_r1_format, judge_answer
from src.prompts import SYSTEM_PROMPT, get_prompt


BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL_PATH = "./outputs/final_model"
DEFAULT_TEST_DATA_PATHS = ["data/test.jsonl", "data/unsolvable_test.jsonl"]


def load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Warning: skipped missing dataset {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model(args: argparse.Namespace):
    print(f"1. Loading tokenizer and base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
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
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if not args.base_only:
        print(f"2. Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print("2. Evaluating base model without LoRA adapter.")

    model.eval()
    return tokenizer, model


def generate_one(tokenizer, model, target_nums: list[int], target_value: int | float, args: argparse.Namespace) -> tuple[str, str]:
    user_content = get_prompt(target_nums, target_value)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **generation_kwargs)

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, extract_answer(response)


def evaluate_cases(
    tokenizer,
    model,
    cases: list[dict[str, Any]],
    dataset_path: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    first_code_counts: Counter[str] = Counter()
    pass_code_counts: Counter[str] = Counter()
    first_success_count = 0
    pass_success_count = 0
    first_format_count = 0

    for i, case in enumerate(tqdm(cases, desc=f"Evaluating {os.path.basename(dataset_path)}")):
        target_nums = case["target_nums"]
        target_value = case.get("target_value", 24)
        solvable = bool(case.get("solvable", True))

        attempts = []
        best_judgment = None
        for attempt_id in range(args.pass_k):
            response, ans = generate_one(tokenizer, model, target_nums, target_value, args)
            judgment = judge_answer(ans, target_nums, target_value=target_value, solvable=solvable)
            attempts.append(
                {
                    "attempt": attempt_id + 1,
                    "response": response,
                    "answer": ans,
                    "ok": judgment.ok,
                    "code": judgment.code,
                    "value": judgment.value,
                    "message": judgment.message,
                    "has_r1_format": has_r1_format(response),
                }
            )
            if best_judgment is None or judgment.ok:
                best_judgment = judgment
            if judgment.ok:
                break

        first = attempts[0]
        first_code_counts[first["code"]] += 1
        pass_code_counts[best_judgment.code if best_judgment else first["code"]] += 1
        if first["ok"]:
            first_success_count += 1
        if best_judgment and best_judgment.ok:
            pass_success_count += 1
        if first["has_r1_format"]:
            first_format_count += 1

        rows.append(
            {
                "dataset": dataset_path,
                "case_index": i,
                "target_nums": target_nums,
                "target_value": target_value,
                "solvable": solvable,
                "source": case.get("source"),
                "source_index": case.get("source_index"),
                "solved_rate": case.get("solved_rate"),
                "first_ok": first["ok"],
                "first_code": first["code"],
                "pass_ok": bool(best_judgment and best_judgment.ok),
                "pass_code": best_judgment.code if best_judgment else first["code"],
                "attempts": attempts,
            }
        )

        if (i + 1) % args.sample_every == 0:
            tqdm.write(
                f"\n[Sample] nums={target_nums} target={target_value} "
                f"answer={first['answer']} first={first['code']} pass={rows[-1]['pass_code']}"
            )

    total = len(cases)
    solvable_total = sum(1 for case in cases if case.get("solvable", True))
    unsolvable_total = total - solvable_total
    metrics = {
        "dataset": dataset_path,
        "total": total,
        "solvable_total": solvable_total,
        "unsolvable_total": unsolvable_total,
        "pass_k": args.pass_k,
        "first_success": first_success_count,
        "pass_success": pass_success_count,
        "first_success_rate": (first_success_count / total) * 100 if total else 0.0,
        "pass_success_rate": (pass_success_count / total) * 100 if total else 0.0,
        "first_format_rate": (first_format_count / total) * 100 if total else 0.0,
        "first_code_counts": dict(first_code_counts),
        "pass_code_counts": dict(pass_code_counts),
        "correct_count": pass_code_counts.get(CORRECT, 0),
        "honest_unsolvable_count": pass_code_counts.get(UNSOLVABLE_CLAIM, 0),
    }
    return metrics, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base/LoRA model on 24-game splits.")
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--adapter-path", default=LORA_MODEL_PATH)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--test-data-path", action="append", default=[], help="JSONL test file. Can be passed multiple times.")
    parser.add_argument("--output-jsonl", default="outputs/eval_results.jsonl")
    parser.add_argument("--summary-json", default="outputs/eval_summary.json")

    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    test_paths = args.test_data_path or DEFAULT_TEST_DATA_PATHS

    tokenizer, model = load_model(args)

    all_rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    for path in test_paths:
        cases = load_jsonl(path)
        print(f"Loaded {len(cases)} cases from {path}.")
        metrics, rows = evaluate_cases(tokenizer, model, cases, path, args)
        all_metrics.append(metrics)
        all_rows.extend(rows)

    write_jsonl(args.output_jsonl, all_rows)
    parent = os.path.dirname(args.summary_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Evaluation report")
    print("=" * 72)
    for metrics in all_metrics:
        print(f"Dataset: {metrics['dataset']}")
        print(f"  Total: {metrics['total']} | pass@{metrics['pass_k']}: {metrics['pass_success_rate']:.2f}% | first@1: {metrics['first_success_rate']:.2f}%")
        print(f"  Format Rate: {metrics['first_format_rate']:.2f}%")
        print(f"  Pass-code counts: {metrics['pass_code_counts']}")
    print(f"Saved detailed rows to {args.output_jsonl}")
    print(f"Saved summary to {args.summary_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
