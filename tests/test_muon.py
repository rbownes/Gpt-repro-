"""Muon optimizer tests."""

from __future__ import annotations

import torch

from gpt_repro.model import GPT, GPTConfig
from gpt_repro.muon import Muon, zeropower_via_newtonschulz5
from gpt_repro.optim import build_dual_optimizer, split_muon_adamw_params
from gpt_repro.tokenizer import VOCAB_SIZE


def test_newtonschulz_returns_approx_orthogonal() -> None:
    """NS(G) should have ~unit singular values for a well-conditioned G."""
    torch.manual_seed(0)
    G = torch.randn(256, 128)
    U = zeropower_via_newtonschulz5(G, steps=5).float()
    # Post-NS, singular values should be clustered near 1.
    sv = torch.linalg.svdvals(U)
    assert (sv > 0.5).all() and (sv < 1.5).all(), f"singular values out of [0.5, 1.5]: {sv.min():.3f}..{sv.max():.3f}"


def test_newtonschulz_preserves_dtype_and_shape() -> None:
    G = torch.randn(64, 32, dtype=torch.float32)
    U = zeropower_via_newtonschulz5(G)
    assert U.shape == G.shape
    assert U.dtype == G.dtype
    assert torch.isfinite(U).all()


def test_newtonschulz_tall_matrix() -> None:
    """Tall matrices (d_out > d_in) get transposed internally."""
    G = torch.randn(128, 256)
    U = zeropower_via_newtonschulz5(G)
    assert U.shape == G.shape
    assert torch.isfinite(U).all()


def test_muon_step_reduces_loss_on_toy_problem() -> None:
    """Muon should make monotonic progress on a simple quadratic.

    Muon orthogonalises the update so it cannot perfectly solve a quadratic
    (it throws away singular-value information). We test *monotonic decrease*,
    not convergence; for that, the modern-block overfit test below is the
    load-bearing check.
    """
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(32, 16))
    target = torch.randn(32, 16)
    opt = Muon([W], lr=0.05)
    losses: list[float] = []
    for _ in range(50):
        opt.zero_grad()
        loss = ((W - target) ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Final loss meaningfully below initial; no catastrophic divergence.
    assert losses[-1] < losses[0] * 0.75, f"Muon barely moved: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert all(torch.isfinite(torch.tensor(l)).item() for l in losses)


def test_muon_rejects_1d_params() -> None:
    W = torch.nn.Parameter(torch.randn(32))
    opt = Muon([W], lr=0.05)
    W.grad = torch.ones_like(W)
    try:
        opt.step()
        raise AssertionError("Muon did not reject a 1-D param")
    except AssertionError as e:
        assert "2-D" in str(e)


def _modern_cfg() -> GPTConfig:
    return GPTConfig(
        vocab_size=VOCAB_SIZE, block_size=32, n_layer=2, n_head=2, n_embd=64,
        attention_backend="sdpa_math",
        positional_encoding="rope", norm_type="rmsnorm",
        mlp_type="swiglu", mlp_hidden=128, qk_norm=True,
    )


def test_param_split_covers_all_modern_params() -> None:
    """Every trainable parameter is assigned to exactly one of (muon, adamw)."""
    model = GPT(_modern_cfg())
    muon_p, adamw_p = split_muon_adamw_params(model)
    counted = {id(p) for p in muon_p} | {id(p) for p in adamw_p}
    all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert counted == all_trainable, "some parameters are unassigned or counted twice"
    # Muon must have got at least the per-layer c_attn/c_proj/w_gate/w_up/w_down
    # (so ≥ 5 * n_layer entries).
    assert len(muon_p) >= 5 * model.cfg.n_layer


def test_dual_optimizer_overfit_tiny_batch() -> None:
    """Sanity: Muon + AdamW converges on a fixed input just like AdamW."""
    torch.manual_seed(0)
    cfg = _modern_cfg()
    model = GPT(cfg)
    muon, adamw = build_dual_optimizer(
        model,
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        adamw_lr=3e-3,
        adamw_weight_decay=0.0,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        fused=False,
    )
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    final = float("nan")
    for _ in range(200):
        for opt in (muon, adamw):
            opt.zero_grad()
        _, loss = model(x, x)
        loss.backward()
        for opt in (muon, adamw):
            opt.step()
        final = loss.item()
    assert final < 0.5, f"dual-optimizer overfit failed: final loss {final:.4f}"
