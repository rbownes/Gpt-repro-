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


# ---- Modern-block variants -------------------------------------------------


def _modern_cfg(**overrides) -> GPTConfig:
    base = dict(
        vocab_size=VOCAB_SIZE,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        attention_backend="sdpa_math",
        positional_encoding="rope",
        rope_base=10000.0,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        mlp_hidden=128,
        qk_norm=True,
    )
    base.update(overrides)
    return GPTConfig(**base)


def test_modern_block_forward_shapes() -> None:
    cfg = _modern_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_modern_block_no_wpe() -> None:
    """RoPE config removes the wpe parameter entirely."""
    cfg = _modern_cfg()
    model = GPT(cfg)
    assert "wpe" not in model.transformer
    # RoPE buffers are registered.
    assert hasattr(model, "rope_cos") and hasattr(model, "rope_sin")


def test_modern_block_param_parity_with_faithful_at_124m() -> None:
    """SwiGLU hidden=2048 and removing wpe leave the non-embedding param count
    of the modern 124M config within a small margin of the faithful one."""
    faithful = GPT(GPTConfig(n_layer=12, n_head=12, n_embd=768, mlp_hidden=None))
    modern = GPT(GPTConfig(
        n_layer=12, n_head=12, n_embd=768,
        positional_encoding="rope", norm_type="rmsnorm",
        mlp_type="swiglu", mlp_hidden=2048, qk_norm=True,
    ))
    # MLP matmul params match exactly (2 * 768 * 3072 == 3 * 768 * 2048 == 4.72M/layer);
    # SwiGLU has one extra bias per layer vs GELU so total MLP params differ by
    # `n_layer * hidden` (~12k on 124M), which is ~0.02% of params — negligible.
    faithful_mlp = sum(p.numel() for n, p in faithful.named_parameters() if ".mlp." in n)
    modern_mlp = sum(p.numel() for n, p in modern.named_parameters() if ".mlp." in n)
    assert abs(modern_mlp - faithful_mlp) / faithful_mlp < 0.001, (
        f"MLP param drift too large: {faithful_mlp} vs {modern_mlp}"
    )
    # Total params: modern is within 1% of faithful (wpe removal offset by QK-norm adds).
    f_total = sum(p.numel() for p in faithful.parameters())
    m_total = sum(p.numel() for p in modern.parameters())
    drift = abs(m_total - f_total) / f_total
    assert drift < 0.01, f"param drift too large: {drift:.3%} ({m_total} vs {f_total})"


def test_modern_block_overfit_tiny_batch() -> None:
    """Same overfit gate as the faithful test, now with the modern block."""
    cfg = _modern_cfg(block_size=32)
    torch.manual_seed(0)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    final = float("nan")
    for _ in range(200):
        opt.zero_grad()
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 0.5, f"modern overfit failed: final loss {final:.4f}"


def test_rope_freqs_basic_properties() -> None:
    """RoPE tables are smooth and deterministic."""
    from gpt_repro.model import rope_freqs

    cos, sin = rope_freqs(head_dim=64, max_seqlen=128, base=10000.0)
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)
    # cos(0)=1, sin(0)=0 at position 0 for every frequency.
    assert torch.allclose(cos[0], torch.ones(64))
    assert torch.allclose(sin[0], torch.zeros(64))
    # Paired halves: first 32 dims == last 32 dims.
    assert torch.allclose(cos[:, :32], cos[:, 32:])
    assert torch.allclose(sin[:, :32], sin[:, 32:])
