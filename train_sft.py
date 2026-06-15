import argparse
import inspect
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.prompts import SYSTEM_PROMPT, get_prompt


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_TRAIN_DATA_PATH = "data/sft_train.jsonl"
OUTPUT_DIR = "./outputs/models/sft_ref_lora8"
CHECKPOINT_DIR = "./outputs/checkpoints_sft"

NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LOGGING_STEPS = 10
LORA_RANK = 8
LORA_ALPHA = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT warmup Qwen2.5-1.5B-Instruct on nlile/24-game reference solutions.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--train-data-path", default=SFT_TRAIN_DATA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--run-name", default="sft_ref_lora8")
    parser.add_argument("--run-root", default="./outputs/runs")

    parser.add_argument("--num-train-epochs", type=float, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--per-device-batch-size", type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)

    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    return parser.parse_args()


def save_config(args: argparse.Namespace, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "sft_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def build_sft_text(example, tokenizer) -> dict[str, str]:
    user_content = get_prompt(example["target_nums"], example.get("target_value", 24))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    config_kwargs = {
        "output_dir": args.checkpoint_dir,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "num_train_epochs": args.num_train_epochs,
        "save_steps": 50,
        "max_length": args.max_length,
        "dataset_text_field": "text",
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "gradient_checkpointing": True,
        "bf16": args.bf16,
        "optim": args.optim,
        "report_to": "none",
        "packing": False,
    }

    supported_params = set(inspect.signature(SFTConfig.__init__).parameters)
    filtered_kwargs = {key: value for key, value in config_kwargs.items() if key in supported_params}
    dropped_keys = sorted(set(config_kwargs) - set(filtered_kwargs))
    if dropped_keys:
        print(f"   SFTConfig does not support {dropped_keys}; skipped for installed TRL version.")
    return SFTConfig(**filtered_kwargs)


def main() -> None:
    args = parse_args()
    run_dir = os.path.join(args.run_root, args.run_name)
    save_config(args, run_dir)

    print(f"1. Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("2. Loading base model...")
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
    model.config.use_cache = False

    print(f"3. Injecting LoRA adapter (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    print(f"4. Loading SFT data: {args.train_data_path}")
    dataset = load_dataset("json", data_files=args.train_data_path, split="train")
    dataset = dataset.map(
        lambda example: build_sft_text(example, tokenizer),
        remove_columns=dataset.column_names,
    )

    print("5. Building SFT config...")
    training_args = build_sft_config(args)

    print("6. Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    print(f"SFT complete. Saving LoRA weights to {args.output_dir}")
    trainer.save_model(args.output_dir)
    print(f"Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
