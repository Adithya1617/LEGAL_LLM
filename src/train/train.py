"""Fine-tune Llama-3.2-3B with QLoRA on chunked CUAD data.

Loads config from YAML, uses Unsloth's FastLanguageModel for memory-efficient
4-bit + LoRA, wraps in TRL's SFTTrainer with response-only loss, logs to W&B.

Run: `python -m src.train.train --config configs/llama32_qlora.yaml`
"""

# Unsloth must be imported BEFORE trl/transformers/peft for its patches to apply.
# That requirement is why we don't use `from __future__ import annotations` here.
from unsloth import FastLanguageModel  # noqa: I001
from unsloth.chat_templates import get_chat_template

import argparse
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer


def _load_jsonl_as_chat(path: Path, tokenizer) -> Dataset:
    """Each row becomes a chat-format example with a single user turn
    (instruction + input) and an assistant turn (output)."""

    def to_chat(row: dict) -> dict:
        user_text = f"{row['instruction']}\n\nExcerpt:\n{row['input']}"
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": row["output"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    ds = load_dataset("json", data_files=str(path), split="train")
    return ds.map(to_chat, remove_columns=ds.column_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    import wandb

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=cfg["training"]["run_name"],
        config=cfg,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"],
    )

    train_ds = _load_jsonl_as_chat(Path(cfg["data"]["train_path"]), tokenizer)
    val_ds = _load_jsonl_as_chat(Path(cfg["data"]["val_path"]), tokenizer)

    sft_cfg = SFTConfig(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        optim=cfg["training"]["optim"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy=cfg["training"]["eval_strategy"],
        eval_steps=cfg["training"]["eval_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        greater_is_better=cfg["training"]["greater_is_better"],
        seed=cfg["training"]["seed"],
        report_to=cfg["training"]["report_to"],
        run_name=cfg["training"]["run_name"],
        max_length=cfg["model"]["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_cfg,
    )
    # Auto-resume if any checkpoint exists in output_dir; first run just starts fresh.
    ckpt_dir = Path(cfg["training"]["output_dir"])
    has_ckpt = ckpt_dir.exists() and any(
        p.name.startswith("checkpoint-") for p in ckpt_dir.iterdir()
    )
    trainer.train(resume_from_checkpoint=has_ckpt or None)

    best_dir = Path(cfg["training"]["output_dir"]) / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Saved best adapter to {best_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
