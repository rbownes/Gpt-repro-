"""Model forward / backward smoke tests."""

from __future__ import annotations

import math

import pytest
import torch

from gpt_repro.model import GPT, GPTConfig
from gpt_repro.tokenizer import EOT_ID, VOCAB_SIZE, decode, encode


@pytest.fixture
def small_cfg() -> GPTConfig:
    return GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        attention_backend="sdpa_math",
    )


def test_forward_shapes(small_cfg: GPTConfig) -> None:
    model = GPT(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (3, small_cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (3, small_cfg.block_size, small_cfg.vocab_size)
    assert loss.ndim == 0


def test_init_loss_near_ln_vocab(small_cfg: GPTConfig) -> None:
    torch.manual_seed(0)
    model = GPT(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (8, small_cfg.block_size))
    _, loss = model(x, x)
    expected = math.log(small_cfg.vocab_size)
    # At init, loss is close to but not exactly ln(V): tied I/O embeddings bias
    # logits toward the input tokens, and residual-scaled init means the
    # distribution is not perfectly uniform. Tolerance of 1.0 is comfortable.
    assert abs(loss.item() - expected) < 1.0, (loss.item(), expected)


def test_backward_nonzero_grads(small_cfg: GPTConfig) -> None:
    model = GPT(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (2, small_cfg.block_size))
    _, loss = model(x, x)
    loss.backward()
    any_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert any_grad


def test_tied_embeddings(small_cfg: GPTConfig) -> None:
    model = GPT(small_cfg)
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()


def test_overfit_tiny_batch() -> None:
    """Tiny model + fixed batch + many steps -> loss must collapse.

    This is the cheapest possible end-to-end gate: if the training loop is
    miswired (wrong target alignment, frozen params, etc.) loss stays at ln(V).
    Expected behavior: loss drops below 0.5 within 200 steps on CPU.
    """
    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE, block_size=32, n_layer=2, n_head=2, n_embd=64,
        attention_backend="sdpa_math",
    )
    torch.manual_seed(0)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y = x
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    final = float("nan")
    for _ in range(200):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 0.5, f"overfit failed: final loss {final:.4f}"


def test_tokenizer_roundtrip() -> None:
    s = "The quick brown fox jumps over the lazy dog."
    ids = encode(s)
    assert ids, "empty encoding"
    assert all(0 <= i < VOCAB_SIZE for i in ids)
    assert decode(ids) == s
    assert EOT_ID == 50256
