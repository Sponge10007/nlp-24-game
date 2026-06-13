import argparse
import inspect
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from src.prompts import SYSTEM_PROMPT, get_prompt
from src.rewards import configure_reward_logging, correctness_reward, flush_reward_logs, format_reward


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "./outputs/final_model"

NUM_GENERATIONS = 2
KL_COEF = 0.05
MAX_PROMPT_LENGTH = 128
MAX_COMPLETION_LENGTH = 384
NUM_TRAIN_EPOCHS = 6

LEARNING_RATE = 5e-6
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LOGGING_STEPS = 10

LORA_RANK = 8
LORA_ALPHA = 16


def preprocess_dataset(example):
    user_content = get_prompt(example["target_nums"], example.get("target_value", 24))
    example["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO train Qwen2.5-1.5B-Instruct on 24-game RLVR.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--train-data-path", default=TRAIN_DATA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", default="./outputs/checkpoints")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--run-root", default="./outputs/runs")

    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--kl-coef", type=float, default=KL_COEF)
    parser.add_argument("--max-prompt-length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--max-completion-length", type=int, default=MAX_COMPLETION_LENGTH)
    parser.add_argument("--num-train-epochs", type=int, default=NUM_TRAIN_EPOCHS)

    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--per-device-batch-size", type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)

    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--reset-metrics", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_config(args: argparse.Namespace, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def build_grpo_config(args: argparse.Namespace) -> GRPOConfig:
    config_kwargs = {
        "output_dir": args.checkpoint_dir,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "num_generations": args.num_generations,
        "num_train_epochs": args.num_train_epochs,
        "save_steps": 50,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "kl_coef": args.kl_coef,
        "gradient_checkpointing": True,
        "bf16": args.bf16,
        "optim": args.optim,
        "report_to": "none",
    }

    supported_params = set(inspect.signature(GRPOConfig.__init__).parameters)
    if "kl_coef" not in supported_params and "beta" in supported_params:
        config_kwargs["beta"] = config_kwargs.pop("kl_coef")

    filtered_kwargs = {
        key: value
        for key, value in config_kwargs.items()
        if key in supported_params
    }
    dropped_keys = sorted(set(config_kwargs) - set(filtered_kwargs))
    if dropped_keys:
        print(f"   GRPOConfig does not support {dropped_keys}; skipped for installed TRL version.")
    return GRPOConfig(**filtered_kwargs)


def main():
    args = parse_args()
    run_name = args.run_name or f"grpo_qwen25_1_5b_lora{args.lora_rank}_g{args.num_generations}"
    run_dir = os.path.join(args.run_root, run_name)
    save_config(args, run_dir)
    configure_reward_logging(
        metrics_file=os.path.join(run_dir, "training_metrics.csv"),
        train_log_file=os.path.join(run_dir, "train_log.txt"),
        reset=args.reset_metrics,
    )

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

    print(f"3. Injecting LoRA adapter (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    print(f"4. Loading train data: {args.train_data_path}")
    dataset = load_dataset("json", data_files=args.train_data_path, split="train")
    dataset = dataset.map(preprocess_dataset)

    print("5. Building GRPO config...")
    training_args = build_grpo_config(args)

    print("6. Starting GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    try:
        trainer.train()
    finally:
        flush_reward_logs()

    print(f"Training complete. Saving LoRA weights to {args.output_dir}")
    trainer.save_model(args.output_dir)
    print(f"Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
