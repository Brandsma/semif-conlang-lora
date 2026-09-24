"""
"Garglish" - a small invented, rule-based language used as a synthetic
fine-tuning target for testing whether a model can learn a made-up grammar.

Rules (fully deterministic, so every example is auto-labelable):

1. Word order:  SUBJECT-OBJECT-VERB in normal sentences.
                Negated sentences flip to VERB-SUBJECT-OBJECT and the
                verb gets a "ne-" prefix.
2. Case marking: subject nouns take the "-ko" suffix, object nouns take
                the "-mi" suffix. Verbs take no suffix.
3. Plurality:   a plural noun reduplicates its first syllable
                (e.g. "dok" -> "dodok").
4. Questions:   a yes/no question is the normal declarative sentence with
                the particle "zu" appended at the very end.
5. Conditionals (Heptapod A-style): a conditional joins two clauses; the
                condition clause starts with the particle "vo". The two
                clauses may come in EITHER order. This deliberately breaks
                Greenberg's Universal 14 (in human languages the condition
                normally precedes the conclusion), the way Heptapod A does
                in Ted Chiang's "Story of Your Life". Only a misplaced "vo"
                (not clause-initial) is a violation, never the clause order.

A sentence can violate zero or more of rules 1-3 and 5 independently (rule 4
is just a suffix and is scored separately as a boolean "is_question").
In a conditional, rule 1-3 violations are applied to one of the two clauses.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# Nonsense vocabulary. Nods to Ingraham's "the gostak distims the doshes",
# Carroll's "Jabberwocky"/"The Hunting of the Snark", Berko's wug test and
# Heinlein's "grok". Constraints (checked in _check_vocab): no word starts
# with "ne", no noun ends in "ko"/"mi", and every surface form is unambiguous.
NOUNS = [
    "dok", "fen", "gar", "hul", "irp", "jov", "kes", "lom",
    "gostak", "dosh", "tove", "wabe", "rath", "borogove", "jubjub", "snark",
    "boojum", "tumtum", "wug", "quib", "plonk", "vorp", "zabble", "murn",
    "snerg", "trill", "bluk", "pim", "yarb", "cloff",
]
VERBS = [
    "bata", "cires", "dulon", "eskim", "florak",
    "distim", "gyre", "gimble", "outgrabe", "galumph", "chortle", "burble",
    "whiffle", "grok", "zorp", "skrill", "blemt", "vashen", "torrin",
    "mulgar", "prazz", "ondrel", "kribble", "sporn", "yeld", "hurlon",
    "dralk", "plimb", "tharn", "frumb",
]

RULE_NAMES = ["word_order", "case_marking", "plurality", "conditional"]
CLAUSE_RULES = ["word_order", "case_marking", "plurality"]

COND_PARTICLE = "vo"


def pluralize(noun: str) -> str:
    """Reduplicate the first syllable (approximated as the first 1-2 chars)."""
    syll = noun[:2]
    return syll + noun


def _check_vocab() -> None:
    """Fail fast if a vocabulary change would make sentences ambiguous."""
    particles = {"vo", "zu"}
    noun_forms = [f + m for n in NOUNS for f in (n, pluralize(n)) for m in ("", "ko", "mi")]
    verb_forms = [f for v in VERBS for f in (v, "ne" + v)]
    forms = noun_forms + verb_forms + sorted(particles)
    assert len(forms) == len(set(forms)), "ambiguous surface forms in vocabulary"
    assert not any(w.startswith("ne") for w in NOUNS + VERBS)
    assert not any(n.endswith(("ko", "mi")) for n in NOUNS)


_check_vocab()


@dataclass
class Sentence:
    text: str
    subject: str
    obj: str
    verb: str
    is_negated: bool
    is_question: bool
    subject_plural: bool
    object_plural: bool
    violations: set = field(default_factory=set)  # subset of RULE_NAMES
    is_conditional: bool = False
    condition_first: bool | None = None  # None when not conditional

    @property
    def is_valid(self) -> bool:
        return len(self.violations) == 0


def _mark_subject(noun: str, plural: bool, correct: bool) -> str:
    base = pluralize(noun) if plural else noun
    return base + "ko" if correct else base  # missing case marker if incorrect


def _mark_object(noun: str, plural: bool, correct: bool) -> str:
    base = pluralize(noun) if plural else noun
    return base + "mi" if correct else base  # missing case marker if incorrect


@dataclass
class _Clause:
    tokens: list
    subject: str
    obj: str
    verb: str
    is_negated: bool
    subject_plural: bool
    object_plural: bool


def _make_clause(rng: random.Random, broken: set) -> _Clause:
    """Build one 3-word clause, breaking the rules in `broken` (subset of CLAUSE_RULES)."""
    subject = rng.choice(NOUNS)
    obj = rng.choice(NOUNS)
    verb = rng.choice(VERBS)
    is_negated = rng.random() < 0.3
    subject_plural = rng.random() < 0.3
    object_plural = rng.random() < 0.3

    correct_case = "case_marking" not in broken
    correct_order = "word_order" not in broken

    # A plurality violation means the reduplication pattern is inconsistent
    # with the noun's declared plurality, so render with flipped plurality.
    flip = "plurality" in broken
    subj_tok = _mark_subject(subject, subject_plural != flip, correct_case)
    obj_tok = _mark_object(obj, object_plural != flip, correct_case)
    verb_tok = ("ne" + verb) if is_negated else verb

    if correct_order:
        if is_negated:
            tokens = [verb_tok, subj_tok, obj_tok]
        else:
            tokens = [subj_tok, obj_tok, verb_tok]
    else:
        # Deliberately scrambled order regardless of negation.
        tokens = [obj_tok, verb_tok, subj_tok]

    return _Clause(tokens, subject, obj, verb, is_negated, subject_plural, object_plural)


def generate_sentence(rng: random.Random, force_valid: bool | None = None) -> Sentence:
    """
    Generate one Garglish sentence.

    force_valid=True  -> guaranteed to follow all rules.
    force_valid=False -> guaranteed to violate at least one rule.
    force_valid=None  -> randomly valid or invalid.
    """
    is_question = rng.random() < 0.3
    is_conditional = rng.random() < 0.3

    if force_valid is None:
        force_valid = rng.random() < 0.5

    violations: set[str] = set()
    if not force_valid:
        # Pick 1-2 rules to break at random (the conditional rule only
        # applies to conditional sentences).
        candidates = RULE_NAMES if is_conditional else CLAUSE_RULES
        n_break = rng.choice([1, 1, 2])
        violations = set(rng.sample(candidates, n_break))
    clause_broken = violations & set(CLAUSE_RULES)

    condition_first = None
    if not is_conditional:
        main = _make_clause(rng, clause_broken)
        tokens = list(main.tokens)
    else:
        # Clause-level violations land in one randomly chosen clause.
        broken_in_cond = rng.random() < 0.5
        cond = _make_clause(rng, clause_broken if broken_in_cond else set())
        main = _make_clause(rng, set() if broken_in_cond else clause_broken)

        cond_tokens = list(cond.tokens)
        if "conditional" in violations:
            # "vo" inside the condition clause instead of at its start.
            cond_tokens.insert(rng.choice([1, 2]), COND_PARTICLE)
        else:
            cond_tokens.insert(0, COND_PARTICLE)

        # Heptapod A: no preferred clause order, both are grammatical.
        condition_first = rng.random() < 0.5
        tokens = cond_tokens + main.tokens if condition_first else main.tokens + cond_tokens

    if is_question:
        tokens = tokens + ["zu"]

    return Sentence(
        text=" ".join(tokens),
        subject=main.subject,
        obj=main.obj,
        verb=main.verb,
        is_negated=main.is_negated,
        is_question=is_question,
        subject_plural=main.subject_plural,
        object_plural=main.object_plural,
        violations=violations,
        is_conditional=is_conditional,
        condition_first=condition_first,
    )


GRAMMAR_RULES_TEXT = """Garglish grammar rules:
1. Word order: Subject-Object-Verb. If negated, order becomes Verb-Subject-Object and the verb gets a "ne-" prefix.
2. Case marking: the subject noun ends in "-ko", the object noun ends in "-mi".
3. Plurality: a plural noun repeats its first syllable at the front (e.g. "dok" -> "dodok").
4. Questions: append the particle "zu" at the end of the sentence.
5. Conditionals: an "if" clause starts with the particle "vo". The "if" clause may come before OR after the main clause; both orders are correct.
"""
