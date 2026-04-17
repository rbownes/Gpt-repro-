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
