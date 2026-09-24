"""
Build a SemIf/Jev-style dataset (state + typed question + options + label)
over the synthetic "Garglish" language defined in grammar.py.

Three question types are produced, mirroring Jev's primitives:

  - noul   : "Is this sentence grammatically valid Garglish?"  -> yes/no
  - choice : "Which rule (if any) does this sentence violate?" -> pick one
  - score  : "How many of the 4 checked rules does this sentence violate?" -> 0..4
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

try:
    from garbage_lang.grammar import GRAMMAR_RULES_TEXT, RULE_NAMES, generate_sentence
except ImportError:  # allow running as a plain script from inside garbage_lang/
    from grammar import GRAMMAR_RULES_TEXT, RULE_NAMES, generate_sentence

CHOICE_OPTIONS = [
    {"id": r, "description": f"Violates the {r.replace('_', ' ')} rule."} for r in RULE_NAMES
] + [{"id": "none", "description": "Follows all grammar rules."}]


def make_examples(rng: random.Random, n: int):
    """Return a list of example groups, one group (2-3 questions) per unique sentence."""
    groups = []
    seen = set()
    while len(groups) < n:
        sent = generate_sentence(rng)
        if sent.text in seen:  # duplicates would leak across splits
            continue
        seen.add(sent.text)
        i = len(groups)
        examples = []
        state = f"{GRAMMAR_RULES_TEXT}\nSentence: \"{sent.text}\""
        meta = {
            "violations": sorted(sent.violations),
            "is_conditional": sent.is_conditional,
            "condition_first": sent.condition_first,
        }

        # 1) Noul: is it valid?
        examples.append(
            {
                "id": f"noul-{i}",
                "type": "noul",
                "state": state,
                "question": "Is this sentence grammatically valid Garglish?",
                "label": "yes" if sent.is_valid else "no",
            }
        )

        # 2) Choice: which single rule is violated (or none)? Only emit when
        # there is exactly 0 or 1 violation, since the schema is single-select.
        if len(sent.violations) <= 1:
            chosen = next(iter(sent.violations), "none")
            examples.append(
                {
                    "id": f"choice-{i}",
                    "type": "choice",
                    "state": state,
                    "question": "Which rule (if any) does this sentence violate?",
                    "options": CHOICE_OPTIONS,
                    "label": chosen,
                }
            )

        # 3) Score: count of violated rules (0-4).
        examples.append(
            {
                "id": f"score-{i}",
                "type": "score",
                "state": state,
                "question": "How many of the 4 checked grammar rules (1, 2, 3, 5) does this sentence violate?",
                "options": [str(k) for k in range(len(RULE_NAMES) + 1)],
                "label": str(len(sent.violations)),
            }
        )
        for ex in examples:
            ex["meta"] = meta
        groups.append(examples)
    return groups


def main():
    ap = argparse.ArgumentParser(description="Generate Garglish SemIf-style dataset")
    ap.add_argument("--n", type=int, default=800, help="number of unique base sentences")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    groups = make_examples(rng, args.n)

    # Split by sentence, so no sentence appears in more than one split.
    rng.shuffle(groups)
    n_val = int(len(groups) * args.val_frac)
    n_test = int(len(groups) * args.test_frac)
    splits = {
        "test": groups[:n_test],
        "val": groups[n_test:n_test + n_val],
        "train": groups[n_test + n_val:],
    }
    test_set, val_set, train_set = ([ex for g in splits[k] for ex in g] for k in ("test", "val", "train"))
    for split in (test_set, val_set, train_set):
        rng.shuffle(split)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train_set), ("val", val_set), ("test", test_set)]:
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} examples -> {path}")


if __name__ == "__main__":
    main()
