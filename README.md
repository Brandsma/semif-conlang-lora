# semif-garbage

Can a small language model learn a grammar it has never seen? This repo
tests that with Garglish, a made-up rule-based language, and asks the
model SemIf-style typed questions about Garglish sentences, before and
after LoRA fine-tuning.

```
gostakko dodoshmi distim       the gostak distims the doshes
```

## Background

Jev (TypeSafe AI) is a closed model that answers typed
questions (Choice, Score, Noul yes/no) with probabilities instead of free
text. [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev), formerly
OpenJev, reproduces that interface with open models by reading option
probabilities straight from the logits. This repo uses the same approach
on a language no model has seen in pretraining, so any skill it shows had
to be learned from the training data.

## Garglish

1. Word order is subject, object, verb. A negated sentence puts the verb
   first and adds `ne-`: `nedistim gostakko dodoshmi`.
2. The subject ends in `-ko`, the object in `-mi`.
3. A plural noun repeats its first syllable: `dosh` becomes `dodosh`.
4. A question ends with `zu`.
5. In a conditional, the if-clause starts with `vo`, and the two clauses
   can come in either order:

   ```
   vo dokko fenmi bata garko hulmi cires
   garko hulmi cires vo dokko fenmi bata
   ```

   Human languages put the if-clause first (Greenberg's Universal 14).
   Heptapod A, the spoken alien language in Ted Chiang's *Story of Your
   Life*, doesn't, and neither does Garglish. Only a `vo` that isn't at
   the start of its clause counts as a mistake.

There are 30 nouns and 30 verbs, including `gostak`, `dosh` and `distim`
from Andrew Ingraham, a few Jabberwocky words, `wug` and `grok`. The
generator is in `garbage_lang/grammar.py`.

Each sentence gets up to three questions:

- Noul: is it valid Garglish? (yes/no)
- Choice: which rule does it break, if any?
- Score: how many rules does it break? (0 to 4)

## Results

Two Qwen2.5 Instruct models were tested before and after LoRA (r=16,
1 epoch on 40k examples, 20 min for 0.5B and 93 min for 3B on an RTX
4070). The test set has 3,000 questions about sentences that never appear
in training.

| | 0.5B base | 0.5B LoRA | 3B base | 3B LoRA |
|---|---:|---:|---:|---:|
| Overall | 0.214 | 0.854 | 0.259 | 0.860 |
| Noul (valid?) | 0.479 | 0.889 | 0.605 | 0.896 |
| Choice (which rule?) | 0.151 | 0.886 | 0.151 | 0.885 |
| Score (how many?) | 0.000 | 0.793 | 0.000 | 0.802 |
| Without a plurality violation | | 0.953 | 0.251 | 0.958 |

Without fine-tuning, both models answer from a fixed bias. They pick
`word_order` for every Choice question and never get a Score question
right, and the 0.5B model calls nearly every sentence valid.

After LoRA, both models catch every word order, case marking and `vo`
mistake, and accept valid conditionals in either clause order. The 3B
base model accepted if-clause-first conditionals a bit more often (62% vs
54%), which fits a human-language bias, but with about 80 sentences per
group that gap is not reliable.

The 3B model gains almost nothing over the 0.5B model after LoRA. The
limit is the data: a plurality violation flips a noun's hidden plural
flag, so it does not show in the text. A parser that reads the text
perfectly scores 0.846 on this test set. The LoRA models reach about 0.86
only because nouns in plurality-violating sentences are reduplicated 70%
of the time instead of 30%, and they learned that statistic. Making
plurality visible in the sentence would make the benchmark useful for
comparing larger models.

## Running it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# dataset: 100k unique sentences, split by sentence into train/val/test
python garbage_lang/build_dataset.py --n 100000

# fine-tune (add --use-4bit for QLoRA, --no-wandb to skip W&B)
python scripts/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-limit 40000 --epochs 1 --output-dir runs/qwen0.5b-lora

# evaluate a base model, or a fine-tuned one with --adapter
python scripts/evaluate.py --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/qwen0.5b-lora/final --limit 3000
```

`experiments/run_semif_lora.sh` runs the whole experiment above and
writes summaries and per-question predictions to `experiments/results/`.
For a W&B hyperparameter sweep: `wandb sweep configs/sweep.yaml`, then
`wandb agent <sweep-id>`.

Training computes the loss on the answer tokens only. Evaluation scores
each allowed answer by its log-probability, as SemIf does;
`--scoring generate` greedy-decodes an answer instead.

## Layout

```
garbage_lang/grammar.py        Garglish rules and sentence generator
garbage_lang/build_dataset.py  builds the question dataset (jsonl)
scripts/train_lora.py          LoRA / QLoRA fine-tuning
scripts/evaluate.py            SemIf-style evaluation
experiments/                   experiment script and results
configs/sweep.yaml             W&B sweep config
```
