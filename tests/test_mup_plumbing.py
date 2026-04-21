"""μP plumbing tests: at base width, all multipliers are 1.0 (no-op)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gpt_repro.model import GPT, GPTConfig
from gpt_repro.mup import (
    apply_mup,
    load_base_shapes,
    mup_group_lr_scale,
    mup_lr_scale,
    record_base_shapes,
    save_base_shapes,
)
from gpt_repro.tokenizer import VOCAB_SIZE


def _cfg_base() -> GPTConfig:
    return GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        attention_backend="sdpa_math",
        positional_encoding="rope",
        norm_type="rmsnorm",
        mlp_type="relu2",
        qk_norm=True,
        zero_init_proj=True,
        u_net_skips=True,
        logit_softcap=30.0,
    )


def test_apply_mup_is_noop_at_base_width() -> None:
    """Self-base: every parameter gets width_mult == 1.0."""
    torch.manual_seed(0)
    cfg = _cfg_base()
    model = GPT(cfg)
    base = record_base_shapes(model)
    apply_mup(model, base)
    for name, p in model.named_parameters():
        m = getattr(p, "mup_width_mult", None)
        assert m == 1.0, f"{name}: mup_width_mult={m} (expected 1.0 at base width)"


def test_mup_lr_scale_is_one_at_base_width() -> None:
    """Every parameter, every optimizer family: scale == 1.0."""
    torch.manual_seed(0)
    cfg = _cfg_base()
    model = GPT(cfg)
    apply_mup(model, record_base_shapes(model))
    for _, p in model.named_parameters():
        for kind in ("adamw", "muon"):
            s = mup_lr_scale(p, kind)
            assert s == 1.0, f"mup_lr_scale({kind}) = {s} (expected 1.0)"


def test_mup_group_lr_scale_is_one_at_base_width() -> None:
    torch.manual_seed(0)
    cfg = _cfg_base()
    model = GPT(cfg)
    apply_mup(model, record_base_shapes(model))
    all_params = [p for _, p in model.named_parameters()]
    for kind in ("adamw", "muon"):
        assert mup_group_lr_scale(all_params, kind) == 1.0


def test_save_load_base_shapes_roundtrip(tmp_path: Path) -> None:
    cfg = _cfg_base()
    model = GPT(cfg)
    shapes = record_base_shapes(model)
    p = tmp_path / "base_shapes.json"
    save_base_shapes(shapes, p)
    round_tripped = load_base_shapes(p)
    assert round_tripped == shapes
    # File is human-readable JSON.
    parsed = json.loads(p.read_text())
    assert parsed == shapes


def test_mup_scaling_kicks_in_at_wider_width() -> None:
    """Record shapes at a narrow 'base' model, apply to a wider model: matrix
    width_mults > 1.0, embeddings / 1-D params stay at 1.0."""
    torch.manual_seed(0)
    base_cfg = _cfg_base()           # n_embd=64
    wide_cfg = GPTConfig(**{**base_cfg.__dict__, "n_embd": 128, "n_head": 4})
    base_model = GPT(base_cfg)
    wide_model = GPT(wide_cfg)
    base_shapes = record_base_shapes(base_model)
    apply_mup(wide_model, base_shapes)
    # A 2-D block matrix (e.g. attention c_attn.weight) should have width_mult ~ 2.
    found_matrix_scale = False
    for name, p in wide_model.named_parameters():
        m = p.mup_width_mult  # type: ignore[attr-defined]
        if "transformer.h.0.attn.c_attn.weight" in name:
            # c_attn fan_in = n_embd; base=64, wide=128 ⇒ width_mult = 2.0
            assert m == pytest.approx(2.0)
            found_matrix_scale = True
            assert mup_lr_scale(p, "adamw") == pytest.approx(0.5)
            assert mup_lr_scale(p, "muon") == pytest.approx(1.0 / 2.0 ** 0.5)
        if name.endswith("ln_1.weight"):
            # 1-D norm: always 1.0
            assert m == 1.0
        if name.startswith("transformer.wte"):
            assert m == 1.0
    assert found_matrix_scale, "did not find the target matrix parameter in model"
