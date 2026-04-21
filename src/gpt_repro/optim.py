"""Optimizer builder and LR schedule.

Two optimizer paths:

1. `cfg.optimizer == "adamw"` (default): fused AdamW on all 2-D params (with
   weight decay) and all 1-D params (without). This matches the faithful
   GPT-2 recipe and every prior accepted experiment (v0.1 – v0.3).

2. `cfg.optimizer == "muon_adamw"`: **Muon** on block matrix parameters
   (attention + MLP weights) and **fused AdamW** on the rest (`wte`, `lm_head`
   via tied embedding, RMSNorm / LayerNorm weights, all biases). See
   `optim_muon.py`. Muon requires one group per distinct matrix shape because
   the fused kernel stacks parameters.

Both paths attach a `base_lr` to every param group. `lr_frac_at_step` returns
a multiplier in [0, 1]; the training loop writes `base_lr * frac` into each
group's `lr` every step. Splits the schedule (shared) from the per-group base
rate (AdamW vs Muon), which are different absolute scales.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from gpt_repro.mup import mup_group_lr_scale
from gpt_repro.optim_muon import MuonAdamW

if TYPE_CHECKING:
    from gpt_repro.train import TrainConfig


# ---------------------------------------------------------------------------
# AdamW parameter grouping (weight-decay-on-2D, decay-exempt-on-1D)
# ---------------------------------------------------------------------------


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split model params into decayed (ndim >= 2) and no-decay (ndim < 2) groups.

    Used by the plain AdamW path. Tied embeddings share the same Parameter
    object, so `named_parameters` already deduplicates — no extra work needed.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
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


# ---------------------------------------------------------------------------
# Optimizer builder (dispatches on cfg.optimizer)
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: "TrainConfig") -> torch.optim.Optimizer:
    """Construct the optimizer based on `cfg.optimizer`.

    Every returned `torch.optim.Optimizer` has `base_lr` on each of its
    `param_groups` so the training loop can apply the schedule multiplicatively.
    """
    kind = getattr(cfg, "optimizer", "adamw")
    if kind == "adamw":
        return _build_adamw(model, cfg)
    if kind == "muon_adamw":
        return _build_muon_adamw(model, cfg)
    raise ValueError(f"Unknown cfg.optimizer={kind!r} (expected 'adamw' or 'muon_adamw')")


def _build_adamw(model: nn.Module, cfg: "TrainConfig") -> torch.optim.AdamW:
    groups = build_param_groups(model, cfg.weight_decay)
    # Tag each group with its base LR (scaled by μP if enabled — at base
    # width μP returns 1.0 so this is a no-op).
    for g in groups:
        mup_scale = mup_group_lr_scale(g["params"], "adamw") if cfg.use_mup else 1.0
        g["base_lr"] = cfg.peak_lr * mup_scale
    # `fused=True` is a big win on CUDA; gracefully falls back on CPU.
    try:
        return torch.optim.AdamW(groups, lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2),
                                 eps=cfg.eps, fused=True)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2),
                                 eps=cfg.eps)


def _is_block_matrix_param(name: str, p: nn.Parameter) -> bool:
    """Return True if `p` is a 2-D matrix inside a transformer block (not an
    embedding and not a tied output head)."""
    if p.ndim != 2:
        return False
    # The embedding matrices are 2-D but must go to AdamW.
    if name.startswith("transformer.wte") or name.startswith("transformer.wpe"):
        return False
    if name.startswith("lm_head"):
        return False
    # Everything else that's 2-D lives inside `transformer.h.*` blocks.
    return True


def _build_muon_adamw(model: nn.Module, cfg: "TrainConfig") -> MuonAdamW:
    """MuonAdamW param groups.

    AdamW group covers: embeddings (`wte`/`wpe`/`lm_head.weight` — all tied
    or decay-irrelevant), RMSNorm / LayerNorm weights, all biases.

    Muon groups: one per unique 2-D shape in the block stack, because the
    fused Muon kernel stacks parameters of equal shape for a single batched
    matmul.
    """
    adamw_params: list[nn.Parameter] = []
    muon_by_shape: dict[tuple[int, ...], list[nn.Parameter]] = {}
    seen_ids: set[int] = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Tied embeddings share a Parameter (same id) — visit only once.
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))

        if _is_block_matrix_param(name, p):
            muon_by_shape.setdefault(tuple(p.shape), []).append(p)
        else:
            adamw_params.append(p)

    if not adamw_params and not muon_by_shape:
        raise RuntimeError("MuonAdamW: model has no trainable parameters")

    groups: list[dict] = []
    # AdamW group first — lower index keeps scheduler output order readable.
    aw_scale = mup_group_lr_scale(adamw_params, "adamw") if cfg.use_mup else 1.0
    groups.append({
        "kind": "adamw",
        "params": adamw_params,
        "lr": cfg.peak_lr * aw_scale,
        "base_lr": cfg.peak_lr * aw_scale,
        "betas": (cfg.beta1, cfg.beta2),
        "eps": cfg.eps,
        "weight_decay": cfg.weight_decay,
    })
    for shape in sorted(muon_by_shape.keys()):
        params = muon_by_shape[shape]
        mu_scale = mup_group_lr_scale(params, "muon") if cfg.use_mup else 1.0
        groups.append({
            "kind": "muon",
            "params": params,
            "lr": cfg.muon_lr * mu_scale,
            "base_lr": cfg.muon_lr * mu_scale,
            "momentum": cfg.muon_momentum,
            "ns_steps": cfg.muon_ns_steps,
            "beta2": cfg.muon_beta2,
            "weight_decay": cfg.weight_decay,
        })
    return MuonAdamW(groups)


# ---------------------------------------------------------------------------
# LR schedule: shared multiplier in [0, 1]; per-group base_lr supplied by the
# optimizer builder.
# ---------------------------------------------------------------------------


def lr_frac_at_step(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Schedule multiplier in [0, 1]: linear warmup → cosine decay to min_lr_ratio."""
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def lr_at_step(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Legacy absolute-LR shim. Equivalent to peak_lr * lr_frac_at_step(…)."""
    return peak_lr * lr_frac_at_step(
        step, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Legacy helper: set a single LR across every param group.

    Newer callers should use the per-group `base_lr * lr_frac_at_step(...)`
    pattern so Muon (high base LR, ~0.02) and AdamW (low base LR, ~6e-4) can
    coexist in a single optimizer.
    """
    for g in optimizer.param_groups:
        g["lr"] = lr


def set_lr_from_frac(optimizer: torch.optim.Optimizer, frac: float, *, fallback_lr: float) -> None:
    """Set each group's `lr` to `group.base_lr * frac`. If a group lacks a
    `base_lr` (legacy), use `fallback_lr * frac`."""
    for g in optimizer.param_groups:
        base = g.get("base_lr", fallback_lr)
        g["lr"] = base * frac
