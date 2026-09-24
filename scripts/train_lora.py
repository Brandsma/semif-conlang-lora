"""
LoRA / QLoRA fine-tuning of a small open model on the synthetic Garglish
(SemIf/Jev-style) dataset produced by garbage_lang/build_dataset.py.

Each training example is formatted as a single prompt ending right before the
answer token, and the model is trained (via standard SFT / causal-LM loss on
the completion only) to emit the correct label token. This mirrors the
typed-decision (Choice / Score / Noul) pattern SemIf uses for scoring, while
using plain supervised fine-tuning so it works with any causal LM + PEFT.

Example:
    python scripts/train_lora.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --train-file data/train.jsonl \\
        --val-file data/val.jsonl \\
        --output-dir runs/garglish-qwen0.5b \\
        --wandb-project semif-garbage
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def format_prompt(ex: dict) -> str:
    """Render one example (state + typed question [+ options]) as text ending
    right before the answer, plus the answer itself for training targets."""
    lines = [ex["state"], "", f"Question: {ex['question']}"]

    if ex["type"] == "choice":
        opt_lines = "\n".join(f"- {o['id']}: {o['description']}" for o in ex["options"])
        lines.append(f"Options:\n{opt_lines}")
        lines.append("Answer with exactly one option id.")
    elif ex["type"] == "score":
        lines.append(f"Options: {', '.join(ex['options'])}")
        lines.append("Answer with exactly one number.")
    else:  # noul
        lines.append("Answer with exactly 'yes' or 'no'.")

    prompt = "\n".join(lines) + "\nAnswer:"
    return prompt


def load_jsonl_as_dataset(path: Path, limit: int | None = None) -> Dataset:
    # Prompt/completion pairs, so TRL computes the loss on the answer only.
    rows = []
    with path.open() as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            ex = json.loads(line)
            rows.append({"prompt": format_prompt(ex), "completion": " " + ex["label"]})
    return Dataset.from_list(rows)


def build_model_and_tokenizer(model_name: str, use_4bit: bool):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return model, tok


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tune a small model on Garglish (SemIf-style) data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train-file", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--val-file", type=Path, default=Path("data/val.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs/garglish"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--train-limit", type=int, default=None, help="Use only the first N training examples")
    ap.add_argument("--val-limit", type=int, default=500, help="Use only the first N validation examples")
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--use-4bit", action="store_true", help="Enable QLoRA (4-bit base weights)")
    ap.add_argument("--wandb-project", default="semif-garbage")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = ap.parse_args()

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        report_to = ["wandb"]
    else:
        report_to = []

    train_ds = load_jsonl_as_dataset(args.train_file, args.train_limit)
    val_ds = load_jsonl_as_dataset(args.val_file, args.val_limit) if args.val_file.exists() else None

    model, tokenizer = build_model_and_tokenizer(args.model, args.use_4bit)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_len,
        logging_steps=10,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        report_to=report_to,
        run_name=args.run_name,
        bf16=torch.cuda.is_available(),
        completion_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    print(f"Saved LoRA adapter to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()
