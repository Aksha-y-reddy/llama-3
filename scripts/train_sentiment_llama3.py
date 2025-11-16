#!/usr/bin/env python3
import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_amazon_us_reviews_binary(
    seed: int,
    train_max: int | None,
    eval_max: int | None,
) -> DatasetDict:
    ds = load_dataset("amazon_us_reviews", "Books_v1_02", split="train")

    def map_label_binary(ex):
        rating = int(ex["star_rating"]) if ex["star_rating"] is not None else 3
        if rating == 3:
            return {"label": -1}
        label = 1 if rating >= 4 else 0
        return {"label": label}

    ds = ds.map(map_label_binary)
    ds = ds.filter(lambda ex: ex["label"] != -1)
    ds = ds.rename_columns({"review_body": "text"})

    keep_cols = ["text", "label"]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=0.05, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    if train_max is not None and len(train_ds) > train_max:
        train_ds = train_ds.select(range(train_max))
    if eval_max is not None and len(eval_ds) > eval_max:
        eval_ds = eval_ds.select(range(eval_max))

    return DatasetDict({"train": train_ds, "eval": eval_ds})


def load_amazon_us_reviews_three_class(
    seed: int,
    train_max: int | None,
    eval_max: int | None,
) -> DatasetDict:
    ds = load_dataset("amazon_us_reviews", "Books_v1_02", split="train")

    def map_label_three(ex):
        rating = int(ex["star_rating"]) if ex["star_rating"] is not None else 3
        if rating <= 2:
            return {"label": 0}
        elif rating == 3:
            return {"label": 1}
        else:
            return {"label": 2}

    ds = ds.map(map_label_three)
    ds = ds.rename_columns({"review_body": "text"})

    keep_cols = ["text", "label"]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=0.05, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    if train_max is not None and len(train_ds) > train_max:
        train_ds = train_ds.select(range(train_max))
    if eval_max is not None and len(eval_ds) > eval_max:
        eval_ds = eval_ds.select(range(eval_max))

    return DatasetDict({"train": train_ds, "eval": eval_ds})


def prepare_label_text(binary_only: bool) -> Dict[int, str]:
    if binary_only:
        return {0: "negative", 1: "positive"}
    return {0: "negative", 1: "neutral", 2: "positive"}


def build_chat_text(tokenizer, text: str, gold_label: int, label_text: Dict[int, str]) -> str:
    allowed = ", ".join(sorted(set(label_text.values())))
    system_prompt = (
        "You are a helpful sentiment analysis assistant. "
        f"Respond with only one word: one of [{allowed}]."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Classify the sentiment of this product review.\n\nReview: {text}"},
        {"role": "assistant", "content": label_text[int(gold_label)]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune LLaMA 3 for sentiment analysis")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/llama3-sentiment-qlora-cli")
    parser.add_argument("--binary_only", action="store_true", default=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=5000)
    parser.add_argument("--per_device_train_bs", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="llama3-sentiment-qlora-cli")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    if args.binary_only:
        raw = load_amazon_us_reviews_binary(args.seed, args.train_max_samples, args.eval_max_samples)
    else:
        raw = load_amazon_us_reviews_three_class(args.seed, args.train_max_samples, args.eval_max_samples)
    label_text = prepare_label_text(args.binary_only)

    # Tokenizer and formatting
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_batch(batch):
        texts, labels = batch["text"], batch["label"]
        out = [build_chat_text(tokenizer, t, l, label_text) for t, l in zip(texts, labels)]
        return {"text": out}

    train_ds = raw["train"].map(format_batch, batched=True, remove_columns=["text", "label"])
    eval_ds = raw["eval"].map(format_batch, batched=True, remove_columns=["text", "label"])

    # Model + QLoRA
    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=64, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_bs,
        per_device_eval_batch_size=max(1, args.per_device_train_bs // 2),
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=["wandb"] if args.use_wandb else [],
        fp16=not supports_bf16,
        bf16=supports_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Quick evaluation using raw eval (label parsing)
    def parse_pred(text: str) -> int:
        from re import search
        allowed = [v.lower() for v in label_text.values()]
        messages = [
            {"role": "system", "content": "Return only one word: " + ", ".join(allowed)},
            {"role": "user", "content": f"Classify the sentiment of this product review.\n\nReview: {text}"},
        ]
        with torch.no_grad():
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            out = model.generate(inputs, max_new_tokens=4, do_sample=False, num_beams=1, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip().lower()
        for k, v in label_text.items():
            if v.lower() in gen:
                return int(k)
        return 1 if args.binary_only else 2

    raw_eval = raw["eval"]
    n = min(1000, len(raw_eval))
    y_true, y_pred = [], []
    for i in range(n):
        ex = raw_eval[i]
        y_true.append(int(ex["label"]))
        y_pred.append(parse_pred(ex["text"]))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary" if args.binary_only else "macro")
    print({"accuracy": acc, "f1": f1})


if __name__ == "__main__":
    main()


