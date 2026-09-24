#!/usr/bin/env bash
# Base vs LoRA, SemIf-style logit scoring, two Qwen2.5 sizes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=experiments/results
N_TEST=3000
N_TRAIN=40000
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_PROJECT=semif-garbage
mkdir -p "$OUT"

for spec in "0.5B 16 1 64" "3B 8 2 16"; do
  read -r size bs ga evbs <<<"$spec"
  model="Qwen/Qwen2.5-${size}-Instruct"
  tag="qwen${size,,}"

  echo "=== $model base ==="
  $PY scripts/evaluate.py --base-model "$model" --limit $N_TEST --batch-size "$evbs" \
    --summary-out "$OUT/${tag}_base.json" --predictions-out "$OUT/${tag}_base_preds.jsonl"

  echo "=== $model LoRA train ==="
  $PY scripts/train_lora.py --model "$model" --output-dir "runs/${tag}-lora40k" \
    --train-limit $N_TRAIN --epochs 1 --batch-size "$bs" --grad-accum "$ga" \
    --val-limit 500 --eval-steps 250 --run-name "${tag}-lora40k"

  echo "=== $model LoRA eval ==="
  $PY scripts/evaluate.py --base-model "$model" --adapter "runs/${tag}-lora40k/final" --limit $N_TEST \
    --batch-size "$evbs" --summary-out "$OUT/${tag}_lora40k.json" --predictions-out "$OUT/${tag}_lora40k_preds.jsonl"
done
echo DONE
