"""μP (Maximal Update Parametrization) plumbing for AdamW / Muon.

Cleanroom implementation of the subset of μP we need: per-parameter width
multiplier (`param.mup_width_mult`) + a per-param LR scale rule that depends
on the optimizer family.

### What this module does

1. **Record a base-shapes dict** from a *base-width* model.
   `record_base_shapes(model)` walks `model.named_parameters()` and stores
   `{name: fan_in}`. Save once to disk per `(model_id, base_width)`; load on
   every subsequent run at that architecture.

2. **Apply μP to a target-width model** via `apply_mup(target_model, base_shapes)`.
   For each parameter, compute `width_mult = fan_in_target / fan_in_base`
   (clamped to 1.0 for embeddings and 1-D tensors, per μP convention) and
   stash it on `.mup_width_mult`. The value is used downstream by
   `mup_lr_scale` to shrink matrix-layer LRs at larger widths.

3. **Per-parameter LR scale** via `mup_lr_scale(param, "adamw" | "muon")`.
   Rules:
     AdamW matrix params:   LR_mult = 1 / width_mult
     Muon  matrix params:   LR_mult = 1 / sqrt(width_mult)
     Embeddings, biases, norms (ndim < 2 or flagged `mup_is_fan_out_only`):
                            LR_mult = 1.0

### No-op at base width

If the model being trained has width == base width, `width_mult == 1.0` for
every parameter and `mup_lr_scale` returns 1.0 for everything. The AdamW and
Muon optimisers behave byte-identically to the non-μP path. This is what we
ship in exp/06 — the plumbing is installed now, the width scaling kicks in
the first time we train at a different width (350M scale-up).

### Deferred: MuReadout output multiplier

At non-base widths, μP additionally wants the *output* of the readout head
(`lm_head`) multiplied by `1 / width_mult`. We don't wire that in this round
because it's a no-op at base width. When we train at non-base width, wrap the
lm_head's forward (or replace with a `MuReadout` subclass) so that
    logits = self.lm_head(x) / self.mup_output_mult
with `mup_output_mult` initialised from the saved base-shapes file.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base shapes: record fan_in for every parameter at the base width
# ---------------------------------------------------------------------------


def _fan_in_of(p: nn.Parameter) -> int:
    """μP fan-in for an arbitrary Parameter.

    For 2-D weight matrices in this codebase (nn.Linear stores weight as
    [out, in]), fan_in = p.shape[-1]. For embeddings (nn.Embedding weight is
    [vocab, embed_dim]), fan_in = 1 by μP convention (embeddings are input-
    side "vocab" lookups, not a matmul along a width axis).

    For 1-D params (norm weights, biases) we record fan_in = 1 so their
    width_mult is trivially 1.0.
    """
    if p.ndim < 2:
        return 1
    return int(p.shape[-1])


def record_base_shapes(model: nn.Module) -> dict[str, int]:
    """Walk `model.named_parameters()` and record `{name: fan_in}`.

    Call once on the base-width model (e.g. 20M, 124M — whichever is your
    tuning base). Save via `save_base_shapes` and re-load at every subsequent
    run of the same architecture (different widths OK).
    """
    shapes: dict[str, int] = {}
    for name, p in model.named_parameters():
        shapes[name] = _fan_in_of(p)
    return shapes


def save_base_shapes(shapes: dict[str, int], path: str | Path) -> None:
    Path(path).write_text(json.dumps(shapes, indent=2, sort_keys=True))


def load_base_shapes(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Apply μP: stash `.mup_width_mult` on every Parameter
# ---------------------------------------------------------------------------


def _is_embedding_name(name: str) -> bool:
    # `wte` (token embedding), `wpe` (learned positional embedding when RoPE
    # is off), and the tied `lm_head.weight` (same tensor as wte.weight).
    return (
        name.startswith("transformer.wte")
        or name.startswith("transformer.wpe")
        or name.startswith("lm_head")
    )


def apply_mup(model: nn.Module, base_shapes: dict[str, int]) -> None:
    """In-place: attach `.mup_width_mult` (float) to every Parameter.

    Rules:
      - Embeddings: width_mult = 1.0 always (μP treats input embeddings as
        having no output width dependence under AdamW).
      - 1-D params (norm weights, biases): width_mult = 1.0.
      - 2-D matrix params: width_mult = current_fan_in / base_fan_in. If the
        parameter name is missing from `base_shapes`, fall back to 1.0 and
        emit a warning-style print (we don't raise — adding new flagged
        matrices between runs should be tolerated).
    """
    for name, p in model.named_parameters():
        if _is_embedding_name(name) or p.ndim < 2:
            p.mup_width_mult = 1.0  # type: ignore[attr-defined]
            continue
        base_fan_in = base_shapes.get(name)
        if base_fan_in is None:
            print(f"[mup] WARN: parameter {name!r} not in base_shapes; using width_mult=1.0")
            p.mup_width_mult = 1.0  # type: ignore[attr-defined]
            continue
        cur_fan_in = _fan_in_of(p)
        p.mup_width_mult = float(cur_fan_in) / float(max(base_fan_in, 1))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LR scale rule (per optimizer family)
# ---------------------------------------------------------------------------


def mup_lr_scale(p: nn.Parameter, optimizer_kind: str) -> float:
    """Return the μP LR multiplier for this parameter.

    At base width (m == 1.0) every rule returns 1.0, so behaviour matches
    the non-μP path bit-for-bit.

    Args:
        p: parameter with `.mup_width_mult` attribute (set by `apply_mup`).
        optimizer_kind: "adamw" or "muon". Under AdamW the matrix-LR
            scaling is 1/m (because Adam's preconditioner already absorbs
            one √m). Under Muon the scaling is 1/√m (SGD-like). 1-D params
            and embeddings scale 1:1 in both families.
    """
    m = getattr(p, "mup_width_mult", 1.0)
    # Embeddings and 1-D params always 1.0 (apply_mup already pinned m=1.0
    # for those — this branch is a safety net if someone sets m manually).
    if p.ndim < 2:
        return 1.0
    if m == 1.0:
        return 1.0
    if optimizer_kind == "adamw":
        return 1.0 / m
    if optimizer_kind == "muon":
        return 1.0 / (m ** 0.5)
    raise ValueError(f"Unknown optimizer_kind={optimizer_kind!r}")


def mup_group_lr_scale(params: list[nn.Parameter], optimizer_kind: str) -> float:
    """Reduce a list of per-param scales to one per-group multiplier.

    Used by the optimizer builder to scale `group['base_lr']` once. All
    parameters within a Muon group share a shape (and thus a width_mult),
    so the mean is exact. For AdamW the group holds mixed shapes — at base
    width the mean is still 1.0, and at non-base widths this is an
    approximation (the more principled route is to split the AdamW group
    by shape too, which we can add later).
    """
    if not params:
        return 1.0
    scales = [mup_lr_scale(p, optimizer_kind) for p in params]
    return sum(scales) / len(scales)
