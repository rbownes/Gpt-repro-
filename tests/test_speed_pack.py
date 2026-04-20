"""Speed-pack tests: GQA + Liger fused CE."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gpt_repro.model import GPT, GPTConfig
from gpt_repro.tokenizer import VOCAB_SIZE


def _base_cfg(**overrides) -> GPTConfig:
    base = dict(
        vocab_size=VOCAB_SIZE,
        block_size=64,
        n_layer=2,
        n_head=4,          # n_head divisible by n_kv_head for GQA tests
        n_embd=64,
        attention_backend="sdpa_math",
        positional_encoding="rope",
        norm_type="rmsnorm",
        mlp_type="relu2",
        qk_norm=False,
    )
    base.update(overrides)
    return GPTConfig(**base)


# ---- GQA ------------------------------------------------------------------


def test_gqa_forward_shapes() -> None:
    cfg = _base_cfg(n_kv_head=2)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_gqa_qkv_projection_is_smaller() -> None:
    """At n_kv_head < n_head, the packed QKV projection output is smaller."""
    mha = GPT(_base_cfg())  # n_kv_head=None ⇒ MHA
    gqa = GPT(_base_cfg(n_kv_head=2))
    mha_out = mha.transformer.h[0].attn.c_attn.weight.shape[0]
    gqa_out = gqa.transformer.h[0].attn.c_attn.weight.shape[0]
    # MHA: 3 * n_embd; GQA: (n_head + 2*n_kv_head) * head_dim
    assert mha_out == 3 * mha.cfg.n_embd
    head_dim = gqa.cfg.n_embd // gqa.cfg.n_head
    assert gqa_out == (gqa.cfg.n_head + 2 * 2) * head_dim
    assert gqa_out < mha_out


def test_gqa_matches_mha_when_kv_heads_equals_n_head() -> None:
    """n_kv_head = n_head should be numerically equivalent to MHA (same fast path)."""
    torch.manual_seed(0)
    mha_cfg = _base_cfg(n_kv_head=None)
    torch.manual_seed(0)
    gqa_cfg = _base_cfg(n_kv_head=mha_cfg.n_head)
    mha = GPT(mha_cfg).eval()
    gqa = GPT(gqa_cfg).eval()
    gqa.load_state_dict(mha.state_dict(), strict=False)
    x = torch.randint(0, mha_cfg.vocab_size, (2, mha_cfg.block_size))
    with torch.no_grad():
        a, _ = mha(x, x)
        b, _ = gqa(x, x)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_gqa_overfit_converges() -> None:
    """GQA model overfits a fixed batch — fwd/bwd wiring is correct."""
    torch.manual_seed(0)
    cfg = _base_cfg(n_kv_head=2, block_size=32)
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
    assert final < 0.5, f"GQA overfit failed: {final:.4f}"


# ---- Liger fused CE ------------------------------------------------------


def test_liger_fused_ce_loss_matches_native() -> None:
    """Liger fused CE should give the same loss as F.cross_entropy."""
    pytest.importorskip("liger_kernel")
    if not torch.cuda.is_available():
        pytest.skip("Liger fused CE requires CUDA")

    torch.manual_seed(0)
    native_cfg = _base_cfg(n_kv_head=2, block_size=32, use_liger_fused_ce=False)
    torch.manual_seed(0)
    liger_cfg = _base_cfg(n_kv_head=2, block_size=32, use_liger_fused_ce=True)

    native = GPT(native_cfg).cuda().bfloat16()
    liger = GPT(liger_cfg).cuda().bfloat16()
    # Liger model's state_dict has an extra buffer for _liger_ce — load with strict=False.
    liger.load_state_dict(native.state_dict(), strict=False)

    x = torch.randint(0, native_cfg.vocab_size, (2, native_cfg.block_size), device="cuda")
    _, native_loss = native(x, x)
    _, liger_loss = liger(x, x)

    torch.testing.assert_close(native_loss.float(), liger_loss.float(), atol=5e-3, rtol=5e-3)


def test_liger_fused_ce_backward_finite() -> None:
    pytest.importorskip("liger_kernel")
    if not torch.cuda.is_available():
        pytest.skip("Liger fused CE requires CUDA")
    cfg = _base_cfg(n_kv_head=2, block_size=32, use_liger_fused_ce=True)
    model = GPT(cfg).cuda().bfloat16()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device="cuda")
    _, loss = model(x, x)
    loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"


def test_liger_fused_ce_and_softcap_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="softcap"):
        GPT(_base_cfg(use_liger_fused_ce=True, logit_softcap=30.0))
