"""
Evaluate a base or LoRA-fine-tuned model on the Garglish held-out test set.

Two scoring modes:
  --scoring logits    (default, SemIf-style) score every allowed option by the
                      model's log-probability of " <option>" after the prompt
                      and pick the best one. No free text is generated, so the
                      model can never answer outside the option set.
  --scoring generate  greedy-decode a short answer and exact-match its first word.

Prints overall / per-type accuracy plus breakdowns by rule and by conditional
clause order, optionally writes per-example predictions to --predictions-out,
and logs a summary + a W&B Table of predictions if W&B is enabled.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import format_prompt


def option_ids(ex: dict) -> list[str]:
    if ex["type"] == "noul":
        return ["yes", "no"]
    return [o["id"] if isinstance(o, dict) else o for o in ex["options"]]


@torch.no_grad()
def predict_logits(model, tokenizer, rows, batch_size):
    """SemIf-style: rank each example's options by summed log-prob of their tokens."""
    seqs = []  # (row index, option, input ids, number of option tokens)
    for i, ex in enumerate(rows):
        p_ids = tokenizer(format_prompt(ex), add_special_tokens=False)["input_ids"]
        for opt in option_ids(ex):
            o_ids = tokenizer(" " + opt, add_special_tokens=False)["input_ids"]
            seqs.append((i, opt, p_ids + o_ids, len(o_ids)))

    scores = defaultdict(dict)
    pad = tokenizer.pad_token_id
    for b in range(0, len(seqs), batch_size):
        batch = seqs[b:b + batch_size]
        max_len = max(len(s[2]) for s in batch)
        # Left-pad so every option ends at the last position and we only need
        # logits for the final few positions.
        ids = torch.tensor([[pad] * (max_len - len(s[2])) + s[2] for s in batch])
        mask = torch.tensor([[0] * (max_len - len(s[2])) + [1] * len(s[2]) for s in batch])
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        keep = max(s[3] for s in batch) + 1
        logits = model(
            input_ids=ids.to(model.device),
            attention_mask=mask.to(model.device),
            position_ids=pos.to(model.device),
            logits_to_keep=keep,
        ).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for j, (i, opt, seq, n_opt) in enumerate(batch):
            targets = torch.tensor(seq[-n_opt:], device=logprobs.device)
            # the token at kept position k is predicted by kept position k-1
            preds = logprobs[j, keep - n_opt - 1:keep - 1]
            scores[i][opt] = preds.gather(-1, targets[:, None]).sum().item()

    out = []
    for i in range(len(rows)):
        s = scores[i]
        probs = torch.softmax(torch.tensor(list(s.values())), dim=0).tolist()
        out.append((max(s, key=s.get), dict(zip(s, probs))))
    return out


@torch.no_grad()
def predict_generate(model, tokenizer, rows, max_new_tokens):
    out = []
    for ex in rows:
        inputs = tokenizer(format_prompt(ex), return_tensors="pt").to(model.device)
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        gen = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = gen.strip().split()[0].strip(".,:;\"'").lower() if gen.strip() else ""
        out.append((pred, None))
    return out


def breakdown(rows, preds):
    """Accuracy per question type, per choice label, and by conditional clause order."""
    groups = defaultdict(lambda: [0, 0])

    def add(key, correct):
        groups[key][0] += correct
        groups[key][1] += 1

    for ex, (pred, _) in zip(rows, preds):
        ok = pred == str(ex["label"]).lower()
        t = ex["type"]
        add("overall", ok)
        add(t, ok)
        if t == "choice":
            add(f"choice | label={ex['label']}", ok)
        meta = ex.get("meta")
        if meta and t == "noul":
            if not meta["is_conditional"]:
                kind = "simple"
            else:
                kind = "cond, if-clause first" if meta["condition_first"] else "cond, if-clause second"
            add(f"noul | {kind} | label={ex['label']}", ok)
    return {k: {"acc": c / n, "n": n} for k, (c, n) in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser(description="Evaluate a model on Garglish test set")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", type=Path, default=None, help="Path to a trained LoRA adapter (optional)")
    ap.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--scoring", choices=["logits", "generate"], default="logits")
    ap.add_argument("--batch-size", type=int, default=32, help="Option sequences per forward pass (logits mode)")
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--predictions-out", type=Path, default=None, help="Write per-example predictions (jsonl)")
    ap.add_argument("--summary-out", type=Path, default=None, help="Write the accuracy breakdown (json)")
    ap.add_argument("--wandb-project", default="semif-garbage")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    rows = []
    with args.test_file.open() as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    if args.scoring == "logits":
        preds = predict_logits(model, tokenizer, rows, args.batch_size)
    else:
        preds = predict_generate(model, tokenizer, rows, args.max_new_tokens)

    summary = breakdown(rows, preds)
    for k, v in summary.items():
        print(f"  {k:<45} {v['acc']:.3f}  (n={v['n']})")

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))
    if args.predictions_out:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_out.open("w") as f:
            for ex, (pred, probs) in zip(rows, preds):
                f.write(json.dumps({"id": ex["id"], "type": ex["type"], "label": ex["label"],
                                    "prediction": pred, "probs": probs, "meta": ex.get("meta")}) + "\n")

    if not args.no_wandb:
        import wandb

        wandb.init(project=args.wandb_project, job_type="eval", config=vars(args) | {"adapter": str(args.adapter)})
        table = wandb.Table(columns=["id", "type", "question", "label", "prediction", "correct"])
        for ex, (pred, _) in zip(rows, preds):
            table.add_data(ex["id"], ex["type"], ex["question"], ex["label"], pred, pred == str(ex["label"]).lower())
        wandb.log({f"test_accuracy_{k}" if k != "overall" else "test_accuracy": v["acc"] for k, v in summary.items()})
        wandb.log({"predictions": table})
        wandb.finish()


if __name__ == "__main__":
    main()
