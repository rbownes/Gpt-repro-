"""Generative multiple-choice evaluation for chat-SFT'd checkpoints.

Instead of LL-scoring the four candidate answers (as in `eval.py`), we
prompt the model with a lettered MC question and ask it to *generate*
the single-letter answer. Reward / score = exact-match on the gold
letter after parsing the generation.

This metric is what exp/15 RL targets — RL teaches the model to emit
just "A/B/C/D" instead of explanatory text. The gap between LL acc
(exp/14 ceiling) and gen acc here is the "unverbalised knowledge"
amount that RL has room to close.

The four task helpers (`hellaswag_gen`, `mmlu_gen`, `arc_easy_gen`,
`arc_challenge_gen`) mirror the LL eval helpers in `eval.py` but
return generative-pass-rate instead of length-normalised LL accuracy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from gpt_repro.chat import ASSISTANT_CLOSE, render_user_turn
from gpt_repro.eval import EvalResult
from gpt_repro.tokenizer import EOT_ID, decode, get_encoding


# Letters are the canonical gold labels; train-time shuffling lives in
# rl_data.py — eval always presents A/B/C/D in dataset order.
LETTERS = ("A", "B", "C", "D")

# Lenient parser tries these in order until one matches.
_PARSE_RE_FIRST = re.compile(r"^\s*[\(\[]?([A-D])\b")
_PARSE_RE_FALLBACK = re.compile(r"\b(?:answer|option)\s*(?:is|:)?\s*[\(\[]?([A-D])\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_mc_prompt(question: str, choices: list[str]) -> str:
    """Build the lettered MC prompt body (without chat markers).

    Output (for 4-way MC, the only kind we currently use):

        Question: {question}
        A) {choices[0]}
        B) {choices[1]}
        C) {choices[2]}
        D) {choices[3]}
        Answer:

    Caller wraps with `chat.render_user_turn` for tokenisation.
    """
    assert 2 <= len(choices) <= len(LETTERS), (
        f"format_mc_prompt: need 2..{len(LETTERS)} choices, got {len(choices)}"
    )
    lines = [f"Question: {question.strip()}"]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}) {choice.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Letter extraction from generated text
# ---------------------------------------------------------------------------


ParseMode = Literal["lenient", "strict"]


def extract_letter(text: str, mode: ParseMode = "lenient") -> str | None:
    """Pull a single A/B/C/D letter out of a generation; None if absent.

    `lenient`: regex match at the start of the (stripped) text, then a
    fallback regex for "answer is X" / "answer: X" / "option B" patterns.
    Accepts surrounding parens or brackets. Case-sensitive on the letter
    (must be uppercase) — the model is being asked to emit exactly that.

    `strict`: only succeed if the very first non-whitespace character is
    one of A/B/C/D (no parens, no prefix words). Used when we want to
    reward strict format compliance during RL.
    """
    if not text:
        return None
    if mode == "strict":
        stripped = text.lstrip()
        if not stripped or stripped[0] not in LETTERS:
            return None
        # Reject if the letter is embedded in a longer word, e.g. "Answer: A"
        # (where the first non-whitespace char is the "A" of "Answer").
        if len(stripped) > 1 and (stripped[1].isalnum() or stripped[1] == "_"):
            return None
        return stripped[0]
    # lenient
    m = _PARSE_RE_FIRST.match(text)
    if m:
        return m.group(1)
    m = _PARSE_RE_FALLBACK.search(text)
    if m:
        return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Greedy generation (argmax — standalone so we don't pull in model.generate's
# multinomial sampling)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _generate_greedy(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    amp_dtype: torch.dtype,
    stop_token_ids: tuple[int, ...] = (EOT_ID,),
) -> list[int]:
    """Greedy (argmax) generation from a single prompt.

    Returns the *generated* tokens only (without the prompt). Stops on
    any of `stop_token_ids` or after `max_new_tokens`.
    """
    device = next(model.parameters()).device
    block_size = model.cfg.block_size if hasattr(model, "cfg") else 1024
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        ctx = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        with torch.autocast(device_type=ctx.device.type, dtype=amp_dtype):
            logits, _ = model(ctx)
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        if next_id in stop_token_ids:
            break
        idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return generated


# ---------------------------------------------------------------------------
# Reward + scoring
# ---------------------------------------------------------------------------


@dataclass
class _MCExample:
    question: str
    choices: list[str]
    gold_letter: str
    extra: dict | None = None


def _gold_letter_or_none(idx: int | None) -> str | None:
    if idx is None or idx < 0 or idx >= len(LETTERS):
        return None
    return LETTERS[idx]


def reward_mc(generated_text: str, gold_letter: str, *, strict: bool = False) -> float:
    """Binary reward for MC: 1 if extracted letter matches gold, else 0."""
    pred = extract_letter(generated_text, mode="strict" if strict else "lenient")
    return 1.0 if pred == gold_letter else 0.0


def _score_examples(
    model: torch.nn.Module,
    examples: Iterable[_MCExample],
    *,
    metric_name: str,
    amp_dtype: torch.dtype,
    mode: ParseMode = "lenient",
    max_new_tokens: int = 16,
) -> EvalResult:
    """Generic generative MC scorer; iterate examples, generate, parse, count."""
    enc = get_encoding()
    close_marker_first_token = enc.encode_ordinary(ASSISTANT_CLOSE)[0]
    stop_ids = (EOT_ID, close_marker_first_token)

    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    parse_failures = 0
    letter_counts: dict[str, int] = {L: 0 for L in LETTERS}

    for ex in examples:
        prompt = format_mc_prompt(ex.question, ex.choices)
        prompt_ids = render_user_turn(prompt)
        gen_ids = _generate_greedy(
            model, prompt_ids,
            max_new_tokens=max_new_tokens,
            amp_dtype=amp_dtype,
            stop_token_ids=stop_ids,
        )
        gen_text = decode(gen_ids)
        pred = extract_letter(gen_text, mode=mode)
        if pred is None:
            parse_failures += 1
        else:
            letter_counts[pred] = letter_counts.get(pred, 0) + 1
        if pred == ex.gold_letter:
            correct += 1
        total += 1

    if was_training:
        model.train()

    return EvalResult(
        metric=metric_name,
        value=correct / max(total, 1),
        n=total,
        extra={
            "mode": mode,
            "max_new_tokens": max_new_tokens,
            "parse_failure_rate": parse_failures / max(total, 1),
            "answer_distribution": letter_counts,
        },
    )


# ---------------------------------------------------------------------------
# Per-task wrappers (mirror eval.py's hellaswag/mmlu/arc_* signatures)
# ---------------------------------------------------------------------------


def _arc_examples(rows) -> Iterable[_MCExample]:
    for ex in rows:
        labels = list(ex["choices"]["label"])
        texts = list(ex["choices"]["text"])
        if len(texts) != 4:
            # ARC has rare 3-way and 5-way rows; skip — keeps the eval 4-way
            # comparable to the LL battery in exp/14.
            continue
        answer_key = str(ex["answerKey"]).strip()
        if answer_key not in labels:
            continue
        gold_idx = labels.index(answer_key)
        gold = _gold_letter_or_none(gold_idx)
        if gold is None:
            continue
        yield _MCExample(question=ex["question"].strip(), choices=texts, gold_letter=gold)


def _hellaswag_examples(rows) -> Iterable[_MCExample]:
    for ex in rows:
        ctx = (ex["activity_label"] + ": " + ex["ctx"]).strip()
        endings = list(ex["endings"])
        if len(endings) != 4:
            continue
        gold = _gold_letter_or_none(int(ex["label"]))
        if gold is None:
            continue
        # HellaSwag's "question" is "what comes next". We frame as a fill-in.
        q = (
            f"Which of the following best continues this passage?\n\n{ctx}"
        )
        yield _MCExample(question=q, choices=endings, gold_letter=gold)


def _mmlu_examples(rows) -> Iterable[_MCExample]:
    for ex in rows:
        choices = list(ex["choices"])
        if len(choices) != 4:
            continue
        gold = _gold_letter_or_none(int(ex["answer"]))
        if gold is None:
            continue
        yield _MCExample(question=ex["question"].strip(), choices=choices, gold_letter=gold)


def hellaswag_gen(
    model: torch.nn.Module,
    *,
    amp_dtype: torch.dtype,
    split: str = "validation",
    limit: int | None = None,
    mode: ParseMode = "lenient",
    max_new_tokens: int = 16,
) -> EvalResult:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("Rowan/hellaswag", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return _score_examples(
        model, _hellaswag_examples(ds),
        metric_name="hellaswag_gen_acc",
        amp_dtype=amp_dtype, mode=mode, max_new_tokens=max_new_tokens,
    )


def mmlu_gen(
    model: torch.nn.Module,
    *,
    amp_dtype: torch.dtype,
    subset: str = "all",
    split: str = "validation",
    limit: int | None = None,
    mode: ParseMode = "lenient",
    max_new_tokens: int = 16,
) -> EvalResult:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("cais/mmlu", subset, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    r = _score_examples(
        model, _mmlu_examples(ds),
        metric_name=f"mmlu_{subset}_gen_acc",
        amp_dtype=amp_dtype, mode=mode, max_new_tokens=max_new_tokens,
    )
    if r.extra is None:
        r.extra = {}
    r.extra["split"] = split
    return r


def _arc_gen(
    model: torch.nn.Module,
    config: str,
    *,
    amp_dtype: torch.dtype,
    split: str,
    limit: int | None,
    mode: ParseMode,
    max_new_tokens: int,
) -> EvalResult:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("allenai/ai2_arc", config, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    name = "arc_easy_gen_acc" if config == "ARC-Easy" else "arc_challenge_gen_acc"
    r = _score_examples(
        model, _arc_examples(ds),
        metric_name=name,
        amp_dtype=amp_dtype, mode=mode, max_new_tokens=max_new_tokens,
    )
    if r.extra is None:
        r.extra = {}
    r.extra["split"] = split
    return r


def arc_easy_gen(
    model: torch.nn.Module,
    *,
    amp_dtype: torch.dtype,
    split: str = "validation",
    limit: int | None = None,
    mode: ParseMode = "lenient",
    max_new_tokens: int = 16,
) -> EvalResult:
    return _arc_gen(model, "ARC-Easy", amp_dtype=amp_dtype, split=split,
                    limit=limit, mode=mode, max_new_tokens=max_new_tokens)


def arc_challenge_gen(
    model: torch.nn.Module,
    *,
    amp_dtype: torch.dtype,
    split: str = "validation",
    limit: int | None = None,
    mode: ParseMode = "lenient",
    max_new_tokens: int = 16,
) -> EvalResult:
    return _arc_gen(model, "ARC-Challenge", amp_dtype=amp_dtype, split=split,
                    limit=limit, mode=mode, max_new_tokens=max_new_tokens)
