"""Faithful GPT-2 optimizer and LR schedule.

Hyperparameters follow Radford et al. 2019 and the nanoGPT replication:
AdamW with (betas=(0.9, 0.95), wd=0.1, eps=1e-8). Weight decay is applied
only to 2D parameters (matmul weights and embeddings); LayerNorm weights,
LayerNorm biases, and linear biases are decay-exempt.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = True,
) -> torch.optim.AdamW:
    groups = build_param_groups(model, weight_decay)
    # `fused=True` is a big win on CUDA, but silently ignored on CPU.
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def lr_at_step(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup -> cosine decay to min_lr_ratio * peak_lr."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return peak_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr
