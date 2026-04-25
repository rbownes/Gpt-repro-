"""Train-split MC data loader for exp/15 RL.

Streams 4-way multiple-choice examples from the *train* splits of the
same datasets used in eval (ARC-Easy, ARC-Challenge, MMLU
auxiliary_train, HellaSwag train) — never overlaps the val splits used
in `eval.py` / `gen_eval.py`.

Each example is yielded with the option order **shuffled** and the
gold letter remapped accordingly. This breaks "always answer A"
surface heuristics so the model can't game format compliance without
also learning content.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from gpt_repro.gen_eval import LETTERS


@dataclass
class MCExample:
    """A 4-way MC training example with letters already permuted."""

    question: str
    choices: list[str]   # 4 strings in (possibly permuted) order
    gold_letter: str     # one of A/B/C/D matching the permuted order
    source: str          # task label, e.g. "arc_easy"


def _hellaswag_to_mc(row) -> tuple[str, list[str], int] | None:
    endings = list(row["endings"])
    if len(endings) != 4:
        return None
    label = int(row["label"])
    if not 0 <= label < 4:
        return None
    q = (
        f"Which of the following best continues this passage?\n\n"
        f"{(row['activity_label'] + ': ' + row['ctx']).strip()}"
    )
    return q, endings, label


def _mmlu_to_mc(row) -> tuple[str, list[str], int] | None:
    choices = list(row["choices"])
    if len(choices) != 4:
        return None
    answer = int(row["answer"])
    if not 0 <= answer < 4:
        return None
    return row["question"].strip(), choices, answer


def _arc_to_mc(row) -> tuple[str, list[str], int] | None:
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    if len(texts) != 4:
        return None
    answer_key = str(row["answerKey"]).strip()
    if answer_key not in labels:
        return None
    return row["question"].strip(), texts, labels.index(answer_key)


def _shuffle_with_gold(
    rng: random.Random, choices: list[str], gold_idx: int,
) -> tuple[list[str], int]:
    """Return (permuted_choices, new_gold_idx) under a random permutation."""
    n = len(choices)
    perm = list(range(n))
    rng.shuffle(perm)
    new_choices = [choices[i] for i in perm]
    new_gold = perm.index(gold_idx)
    return new_choices, new_gold


def _stream_dataset(rows, parser, source: str, rng: random.Random) -> Iterator[MCExample]:
    for row in rows:
        parsed = parser(row)
        if parsed is None:
            continue
        q, choices, gold_idx = parsed
        choices, gold_idx = _shuffle_with_gold(rng, choices, gold_idx)
        yield MCExample(
            question=q, choices=choices, gold_letter=LETTERS[gold_idx], source=source,
        )


class MCMixture:
    """Infinite shuffled stream over the 4 MC train splits.

    Args:
        seed: rng seed for both the deterministic example shuffle and the
              per-example letter permutation. Same seed → same stream.
        mmlu_aux_limit: optionally cap MMLU auxiliary_train (99 842 rows is
              way more than needed at our scale; sampling 5 k keeps a high
              ratio of HSwag/ARC examples).
        hellaswag_limit: similarly optional cap on HSwag train (~39 k).

    Yields `MCExample` objects forever (re-epoching with reshuffled order
    on every loop). Use `take(n)` to draw a finite list.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        mmlu_aux_limit: int | None = 5_000,
        hellaswag_limit: int | None = 5_000,
    ) -> None:
        from datasets import load_dataset  # type: ignore

        self.seed = seed
        # Load all four train splits into in-memory lists of MCExample
        # so reshuffling each epoch is cheap.
        rng = random.Random(seed)

        arc_e_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        arc_c_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        mmlu_ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        hswag_ds = load_dataset("Rowan/hellaswag", split="train")

        if mmlu_aux_limit is not None:
            idxs = list(range(len(mmlu_ds)))
            rng.shuffle(idxs)
            mmlu_ds = mmlu_ds.select(idxs[:mmlu_aux_limit])
        if hellaswag_limit is not None:
            idxs = list(range(len(hswag_ds)))
            rng.shuffle(idxs)
            hswag_ds = hswag_ds.select(idxs[:hellaswag_limit])

        # Materialise into a flat list with letter-shuffles already applied
        # at construction time so the same `MCMixture` re-yields identical
        # examples across iterations (only the visit order changes per epoch).
        examples: list[MCExample] = []
        for x in _stream_dataset(arc_e_ds, _arc_to_mc, "arc_easy", rng):
            examples.append(x)
        for x in _stream_dataset(arc_c_ds, _arc_to_mc, "arc_challenge", rng):
            examples.append(x)
        for x in _stream_dataset(mmlu_ds, _mmlu_to_mc, "mmlu", rng):
            examples.append(x)
        for x in _stream_dataset(hswag_ds, _hellaswag_to_mc, "hellaswag", rng):
            examples.append(x)
        self.examples = examples
        self._rng = rng

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[MCExample]:
        order = list(range(len(self.examples)))
        while True:
            self._rng.shuffle(order)
            for i in order:
                yield self.examples[i]

    def take(self, n: int) -> list[MCExample]:
        out: list[MCExample] = []
        it = iter(self)
        for _ in range(n):
            out.append(next(it))
        return out
