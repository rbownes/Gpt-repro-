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

from gpt_repro.muon import Muon


# Hidden matmul weights that Muon should handle. Anything else (embeddings,
# norm scales, biases, QK-Norm scales, etc.) stays on AdamW.
_MUON_SUFFIXES: tuple[str, ...] = (
    "c_attn.weight",
    "c_proj.weight",
    "w_gate.weight",
    "w_up.weight",
    "w_down.weight",
)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
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


def split_muon_adamw_params(model: nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Return `(muon_params, adamw_params)`.

    Muon gets 2-D hidden matmul weights matched by suffix. AdamW gets the rest.
    Duplicate parameters (e.g. tied lm_head/wte) are added only once on the
    AdamW side.
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and any(name.endswith(s) for s in _MUON_SUFFIXES):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


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


def build_dual_optimizer(
    model: nn.Module,
    *,
    muon_lr: float,
    muon_momentum: float,
    muon_nesterov: bool,
    muon_ns_steps: int,
    adamw_lr: float,
    adamw_weight_decay: float,
    adamw_betas: tuple[float, float],
    adamw_eps: float,
    fused: bool = True,
) -> tuple[Muon, torch.optim.AdamW]:
    """Build a (Muon, AdamW) pair with parameters split by role.

    Muon handles 2-D hidden matmul weights; AdamW handles embeddings, norm
    scales, biases, and any non-hidden matmul weight. Both optimizers receive
    independent peak learning rates so the schedule can be shared but their
    magnitudes differ.
    """
    muon_params, adamw_params = split_muon_adamw_params(model)
    if not muon_params:
        raise ValueError("No parameters matched for Muon — check suffix list.")
    muon = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        ns_steps=muon_ns_steps,
    )
    # Split AdamW params by decay-eligibility just like build_param_groups.
    decay = [p for p in adamw_params if p.ndim >= 2]
    no_decay = [p for p in adamw_params if p.ndim < 2]
    adamw_groups = [
        {"params": decay, "weight_decay": adamw_weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    try:
        adamw = torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=adamw_betas, eps=adamw_eps, fused=fused)
    except (RuntimeError, TypeError):
        adamw = torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=adamw_betas, eps=adamw_eps)
    return muon, adamw
