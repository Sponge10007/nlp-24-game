import argparse
import inspect
import json
import os


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
WARMUP_DATA_PATH = "data/warmup_train.jsonl"
OUTPUT_DIR = "./outputs/sft_model"

NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 2e-5
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LOGGING_STEPS = 10
MAX_SEQ_LENGTH = 512

LORA_RANK = 8
LORA_ALPHA = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT warmup Qwen2.5-1.5B-Instruct before GRPO.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--train-data-path", default=WARMUP_DATA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", default="./outputs/sft_checkpoints")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--run-root", default="./outputs/runs")

    parser.add_argument("--num-train-epochs", type=int, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--per-device-batch-size", type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)

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


def render_text(tokenizer, example: dict) -> dict:
    prompt_text = tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )
    example["text"] = prompt_text + example["completion"] + tokenizer.eos_token
    return example


def build_sft_config(args: argparse.Namespace):
    from trl import SFTConfig

    config_kwargs = {
        "output_dir": args.checkpoint_dir,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "num_train_epochs": args.num_train_epochs,
        "save_steps": 50,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "gradient_checkpointing": True,
        "bf16": args.bf16,
        "optim": args.optim,
        "report_to": "none",
        "max_seq_length": args.max_seq_length,
        "dataset_text_field": "text",
    }

    supported_params = set(inspect.signature(SFTConfig.__init__).parameters)
    filtered_kwargs = {
        key: value
        for key, value in config_kwargs.items()
        if key in supported_params
    }
    dropped_keys = sorted(set(config_kwargs) - set(filtered_kwargs))
    if dropped_keys:
        print(f"   SFTConfig does not support {dropped_keys}; skipped for installed TRL version.")
    return SFTConfig(**filtered_kwargs)


def build_sft_trainer(tokenizer, model, training_args, dataset):
    from trl import SFTTrainer

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "dataset_text_field": "text",
        "max_seq_length": getattr(training_args, "max_seq_length", MAX_SEQ_LENGTH),
    }
    supported_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    filtered_kwargs = {
        key: value
        for key, value in trainer_kwargs.items()
        if key in supported_params
    }
    dropped_keys = sorted(set(trainer_kwargs) - set(filtered_kwargs))
    if dropped_keys:
        print(f"   SFTTrainer does not support {dropped_keys}; skipped for installed TRL version.")
    return SFTTrainer(**filtered_kwargs)


def main():
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    run_name = args.run_name or f"sft_warmup_lora{args.lora_rank}"
    run_dir = os.path.join(args.run_root, run_name)
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

    print(f"3. Injecting LoRA adapter (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    print(f"4. Loading SFT warmup data: {args.train_data_path}")
    dataset = load_dataset("json", data_files=args.train_data_path, split="train")
    dataset = dataset.map(lambda example: render_text(tokenizer, example))

    print("5. Building SFT config...")
    training_args = build_sft_config(args)

    print("6. Starting SFT warmup...")
    trainer = build_sft_trainer(tokenizer, model, training_args, dataset)
    trainer.train()

    print(f"SFT warmup complete. Saving LoRA weights to {args.output_dir}")
    trainer.save_model(args.output_dir)
    print(f"Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
