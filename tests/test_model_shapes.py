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


# ---- Modded-nanogpt tricks (exp/03) ---------------------------------------


def _modernplus_cfg(**overrides) -> GPTConfig:
    base = dict(
        vocab_size=VOCAB_SIZE,
        block_size=64,
        n_layer=4,           # small but ≥ 2 so U-Net has something to skip
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
    base.update(overrides)
    return GPTConfig(**base)


def test_relu2_mlp_forward() -> None:
    cfg = _modernplus_cfg(u_net_skips=False, zero_init_proj=False, logit_softcap=None)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_zero_init_proj_literally_zero() -> None:
    cfg = _modernplus_cfg()
    model = GPT(cfg)
    zero_keys = [
        n for n, _ in model.named_parameters()
        if n.endswith((".c_proj.weight", ".w_down.weight"))
    ]
    assert zero_keys, "expected zero-initialised projection weights to exist"
    for n in zero_keys:
        p = dict(model.named_parameters())[n]
        assert torch.all(p == 0), f"{n} not zero: max|p|={p.abs().max().item():.3e}"


def test_zero_init_off_non_zero() -> None:
    """With zero_init_proj=False, projections are non-zero (scaled init)."""
    cfg = _modernplus_cfg(zero_init_proj=False)
    model = GPT(cfg)
    any_nonzero = False
    for n, p in model.named_parameters():
        if n.endswith((".c_proj.weight", ".w_down.weight")):
            if p.abs().sum() > 0:
                any_nonzero = True
                break
    assert any_nonzero


def test_u_net_skip_wiring_stateless_on_shapes() -> None:
    """Running the same input twice with U-Net skips gives identical output
    (confirms the skip stack is properly rebuilt per-forward, not leaked)."""
    cfg = _modernplus_cfg()
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    with torch.no_grad():
        out1, _ = model(x, x)
        out2, _ = model(x, x)
    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


def test_logit_softcap_bounds_logits() -> None:
    """With softcap=10, every returned logit lies in (-10, 10)."""
    cfg = _modernplus_cfg(logit_softcap=10.0)
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    with torch.no_grad():
        logits, _ = model(x, x)
    assert logits.abs().max().item() < 10.0 + 1e-4


def test_modernplus_overfit() -> None:
    torch.manual_seed(0)
    cfg = _modernplus_cfg(block_size=32)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    final = float("nan")
    for _ in range(300):
        opt.zero_grad()
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 0.5, f"modernplus overfit failed: final {final:.4f}"


# ---- MLA (Multi-head Latent Attention, exp/10) -----------------------------


def _mla_cfg(**overrides) -> GPTConfig:
    """v0.3-style model with MLA attention swapped in, at test scale.

    n_embd=64, n_head=2 so d_head=32. MLA chunks: d_qk_nope=16, d_qk_rope=16,
    d_v=32, d_kv_comp=64. Sized so the attention layer is non-trivial but the
    whole model is tiny.
    """
    base = dict(
        vocab_size=VOCAB_SIZE,
        block_size=64,
        n_layer=4,
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
        attention_type="mla",
        mla_d_kv_comp=64,
        mla_d_qk_nope=16,
        mla_d_qk_rope=16,
        mla_d_v=32,
    )
    base.update(overrides)
    return GPTConfig(**base)


def test_mla_forward_shapes() -> None:
    cfg = _mla_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_mla_rope_buffer_sized_to_d_qk_rope() -> None:
    """MLA re-sizes the shared RoPE table to mla_d_qk_rope, not n_embd/n_head."""
    cfg = _mla_cfg(mla_d_qk_rope=16)
    model = GPT(cfg)
    assert model.rope_cos.shape[-1] == 16
    assert model.rope_sin.shape[-1] == 16


def test_mla_backward_nonzero_grads_without_zero_init() -> None:
    """With zero_init_proj=False every MLA parameter receives a non-zero grad
    on the first step. (With zero_init_proj=True, block-internal grads are
    blocked by the zero out-projections — that's the modded-nanogpt design.)"""
    cfg = _mla_cfg(zero_init_proj=False)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(x, y)
    loss.backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, f"{len(dead)} MLA params with zero grad: {dead[:5]}"


def test_mla_qk_norm_only_on_nope() -> None:
    """QK-Norm RMSNorm weights match d_qk_nope, not d_head or d_qk_rope."""
    cfg = _mla_cfg(qk_norm=True)
    model = GPT(cfg)
    attn = model.transformer.h[0].attn
    assert attn.q_norm.weight.shape == (cfg.mla_d_qk_nope,)
    assert attn.k_norm.weight.shape == (cfg.mla_d_qk_nope,)


def test_mla_switches_cleanly_from_mha() -> None:
    """Same outer GPTConfig with attention_type='mha' vs 'mla' both build and
    forward cleanly at matched hyperparams."""
    base = _mla_cfg().__dict__
    mha_kwargs = {**base, "attention_type": "mha"}
    # mla-specific fields are ignored by the MHA path, but they're still valid
    # fields on GPTConfig so no need to drop them.
    mha_model = GPT(GPTConfig(**mha_kwargs))
    mla_model = GPT(GPTConfig(**base))
    x = torch.randint(0, base["vocab_size"], (2, base["block_size"]))
    for m in (mha_model, mla_model):
        _, loss = m(x, x)
        assert torch.isfinite(loss)


def test_mla_param_count_smaller_than_mha_at_124m() -> None:
    """At 124M with these MLA chunks, the model is ~116M (attn has fewer params).
    This is a known accounting fact; fail loudly if it ever changes."""
    mha = GPT(GPTConfig(
        n_layer=12, n_head=12, n_embd=768,
        positional_encoding="rope", norm_type="rmsnorm",
        mlp_type="relu2", qk_norm=True,
        zero_init_proj=True, u_net_skips=True, logit_softcap=30.0,
        attention_type="mha",
    ))
    mla = GPT(GPTConfig(
        n_layer=12, n_head=12, n_embd=768,
        positional_encoding="rope", norm_type="rmsnorm",
        mlp_type="relu2", qk_norm=True,
        zero_init_proj=True, u_net_skips=True, logit_softcap=30.0,
        attention_type="mla",
        mla_d_kv_comp=256, mla_d_qk_nope=32, mla_d_qk_rope=32, mla_d_v=64,
    ))
    mha_n = sum(p.numel() for p in mha.parameters())
    mla_n = sum(p.numel() for p in mla.parameters())
    # MLA should be strictly fewer, within 10% of MHA (~6.5% expected).
    assert mla_n < mha_n, f"expected MLA < MHA params, got {mla_n} vs {mha_n}"
    assert (mha_n - mla_n) / mha_n < 0.10, (
        f"MLA is {(mha_n - mla_n)/mha_n*100:.2f}% smaller — expected ≤ 10%"
    )


def test_mla_overfit_tiny_batch() -> None:
    torch.manual_seed(0)
    cfg = _mla_cfg(block_size=32)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    final = float("nan")
    for _ in range(400):
        opt.zero_grad()
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 0.5, f"MLA overfit failed: final {final:.4f}"
