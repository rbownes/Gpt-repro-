"""Evaluation primitives: validation loss and HellaSwag zero-shot.

HellaSwag scoring follows the karpathy/llm.c convention: given a context
and 4 candidate endings, pick the candidate with the highest average
per-token log-likelihood over the ending tokens. This is a pure zero-shot
LM eval — no fine-tuning, no prompting tricks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from gpt_repro.tokenizer import encode


@dataclass
class EvalResult:
    metric: str
    value: float
    n: int
    extra: dict | None = None


@torch.no_grad()
def val_loss(model: torch.nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], amp_dtype: torch.dtype) -> EvalResult:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    device = next(model.parameters()).device
    for x, y in batches:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=x.device.type, dtype=amp_dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return EvalResult("val_loss", float(np.mean(losses)) if losses else float("nan"), len(losses))


def _score_continuation(
    model: torch.nn.Module,
    ctx_ids: list[int],
    end_ids: list[int],
    device: torch.device | str,
    amp_dtype: torch.dtype,
) -> float:
    """Return mean per-token log-likelihood of `end_ids` given `ctx_ids`.

    Predicts each `end_ids[i]` from the logits at the position just before it,
    which is `ctx_ids[-1]` for the first ending token and `end_ids[i-1]` for
    the rest. We pass `targets` so the model returns full-sequence logits.
    """
    full = torch.tensor(ctx_ids + end_ids, dtype=torch.long, device=device)[None, :]
    inp = full[:, :-1]
    tgt = full[:, 1:]
    with torch.autocast(device_type=inp.device.type, dtype=amp_dtype):
        logits, _ = model(inp, tgt)  # [1, T, V]
    end_len = len(end_ids)
    logits = logits[0, -end_len:, :]
    tgt_end = tgt[0, -end_len:]
    logp = F.log_softmax(logits.float(), dim=-1)
    per_tok = logp[torch.arange(end_len, device=device), tgt_end]
    return per_tok.mean().item()


@torch.no_grad()
def hellaswag(
    model: torch.nn.Module,
    *,
    amp_dtype: torch.dtype,
    split: str = "validation",
    limit: int | None = None,
) -> EvalResult:
    """Zero-shot HellaSwag accuracy.

    Uses the 'Rowan/hellaswag' dataset. Each example provides a context
    (`ctx`) and 4 candidate `endings`; correct index is `label`.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("Rowan/hellaswag", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0
    for ex in ds:
        ctx_text = (ex["activity_label"] + ": " + ex["ctx"]).strip()
        ctx_ids = encode(ctx_text)
        scores = []
        for ending in ex["endings"]:
            end_ids = encode(" " + ending.strip())
            if not end_ids:
                scores.append(-float("inf"))
                continue
            scores.append(_score_continuation(model, ctx_ids, end_ids, device, amp_dtype))
        pred = int(np.argmax(scores))
        label = int(ex["label"])
        correct += int(pred == label)
        total += 1

    return EvalResult("hellaswag_acc", correct / max(total, 1), total)


# ---------------------------------------------------------------------------
# MMLU / ARC — 4-way multiple choice via length-normalised LL scoring.
#
# Same pattern as HellaSwag: build a prompt, tokenize each candidate answer
# as a separate continuation, and pick max mean-log-likelihood. All three
# eval functions share a small helper that accepts a stream of multiple-
# choice examples.
# ---------------------------------------------------------------------------


def _score_multiple_choice(
    model: torch.nn.Module,
    prompt_text: str,
    choices: list[str],
    *,
    amp_dtype: torch.dtype,
    device: torch.device | str,
) -> int:
    """Return the index of the highest-scoring choice."""
    ctx_ids = encode(prompt_text)
    scores: list[float] = []
    for choice in choices:
        end_ids = encode(" " + choice.strip())
        if not end_ids:
            scores.append(-float("inf"))
            continue
        scores.append(_score_continuation(model, ctx_ids, end_ids, device, amp_dtype))
    return int(np.argmax(scores))


@torch.no_grad()
def mmlu(
    model: torch.nn.Module,
    *,
    amp_dtype: torch.dtype,
    subset: str = "all",
    split: str = "validation",
    limit: int | None = None,
) -> EvalResult:
    """Zero-shot MMLU accuracy (length-normalised LL over the 4 choices).

    Uses `cais/mmlu`. Each row has `question`, `choices` (list of 4 strings),
    and `answer` (int index 0..3). Prompt format:

        {question}
        Answer:

    Each candidate answer is tokenised as " {choice}" and scored.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("cais/mmlu", subset, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0
    for ex in ds:
        prompt = f"{ex['question'].strip()}\nAnswer:"
        pred = _score_multiple_choice(model, prompt, list(ex["choices"]),
                                      amp_dtype=amp_dtype, device=device)
        correct += int(pred == int(ex["answer"]))
        total += 1

    return EvalResult(f"mmlu_{subset}_acc", correct / max(total, 1), total,
                      extra={"split": split})


def _arc_eval(
    model: torch.nn.Module,
    config: str,
    *,
    amp_dtype: torch.dtype,
    split: str,
    limit: int | None,
) -> EvalResult:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("allenai/ai2_arc", config, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0
    for ex in ds:
        prompt = f"Question: {ex['question'].strip()}\nAnswer:"
        labels = list(ex["choices"]["label"])
        texts = list(ex["choices"]["text"])
        answer_key = str(ex["answerKey"]).strip()
        if answer_key not in labels:
            # A handful of ARC rows have stray answer keys (e.g. "1" vs label "A").
            # Skip rather than guess.
            continue
        gold = labels.index(answer_key)
        pred = _score_multiple_choice(model, prompt, texts,
                                      amp_dtype=amp_dtype, device=device)
        correct += int(pred == gold)
        total += 1

    name = "arc_easy_acc" if config == "ARC-Easy" else "arc_challenge_acc"
    return EvalResult(name, correct / max(total, 1), total, extra={"split": split})


@torch.no_grad()
def arc_easy(
    model: torch.nn.Module,
    *,
    amp_dtype: torch.dtype,
    split: str = "validation",
    limit: int | None = None,
) -> EvalResult:
    """Zero-shot ARC-Easy accuracy."""
    return _arc_eval(model, "ARC-Easy", amp_dtype=amp_dtype, split=split, limit=limit)


@torch.no_grad()
def arc_challenge(
    model: torch.nn.Module,
    *,
    amp_dtype: torch.dtype,
    split: str = "validation",
    limit: int | None = None,
) -> EvalResult:
    """Zero-shot ARC-Challenge accuracy."""
    return _arc_eval(model, "ARC-Challenge", amp_dtype=amp_dtype, split=split, limit=limit)
