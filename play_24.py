import argparse
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.game24 import extract_answer, judge_answer
from src.prompts import SYSTEM_PROMPT, get_prompt


BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL_PATH = "./outputs/final_model"


def parse_numbers(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    quantization_config = None
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
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return tokenizer, model


def solve_once(tokenizer, model, nums: list[int], target: int | float, args: argparse.Namespace) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_prompt(nums, target)},
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
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively solve 24-game puzzles with the trained model.")
    parser.add_argument("numbers", nargs="*", type=int, help="Numbers such as: 3 3 8 8")
    parser.add_argument("--target", type=float, default=24)
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--adapter-path", default=LORA_MODEL_PATH)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args)

    if args.numbers:
        puzzles = [args.numbers]
    else:
        puzzles = []
        print("Enter 3-4 numbers, for example: 3 3 8 8. Empty input exits.")
        while True:
            raw = input("numbers> ").strip()
            if not raw:
                break
            nums = parse_numbers(raw)
            if not 3 <= len(nums) <= 4:
                print("Please enter 3 or 4 integers.")
                continue
            puzzles.append(nums)

    for nums in puzzles:
        response = solve_once(tokenizer, model, nums, args.target, args)
        answer = extract_answer(response)
        judgment = judge_answer(answer, nums, target_value=args.target, solvable=True)
        print("\n" + "=" * 72)
        print(f"Puzzle: nums={nums}, target={args.target}")
        print(response)
        print("-" * 72)
        print(f"Answer: {answer}")
        print(f"Verdict: {judgment.code}, ok={judgment.ok}, value={judgment.value}")
        print("=" * 72)


if __name__ == "__main__":
    main()
