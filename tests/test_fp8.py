"""FP8 path tests. Require CUDA + TransformerEngine."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
te = pytest.importorskip("transformer_engine.pytorch")

from gpt_repro.model import GPT, GPTConfig  # noqa: E402


def _fp8_cfg(**overrides) -> GPTConfig:
    base = dict(
        vocab_size=50257,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        attention_backend="sdpa_flash",
        positional_encoding="rope",
        norm_type="rmsnorm",
        mlp_type="relu2",
        qk_norm=True,
        zero_init_proj=False,
        u_net_skips=False,
        logit_softcap=None,
        use_fp8=True,
    )
    base.update(overrides)
    return GPTConfig(**base)


@cuda
def test_fp8_model_forward_runs() -> None:
    cfg = _fp8_cfg()
    model = GPT(cfg).cuda().bfloat16()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device="cuda")
    logits, loss = model(x, x)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert torch.isfinite(loss)


@cuda
def test_fp8_uses_te_linear() -> None:
    cfg = _fp8_cfg()
    model = GPT(cfg)
    # Every hidden matmul should be a te.Linear; lm_head and embeddings stay native.
    hidden_keys = [
        "transformer.h.0.attn.c_attn",
        "transformer.h.0.attn.c_proj",
        "transformer.h.0.mlp.c_fc",
        "transformer.h.0.mlp.c_proj",
    ]
    modules = dict(model.named_modules())
    for k in hidden_keys:
        assert modules[k].__class__.__module__.startswith("transformer_engine"), (
            f"{k} should be a te module, got {type(modules[k])}"
        )
    assert isinstance(modules["lm_head"], torch.nn.Linear), "lm_head should stay native"


@cuda
def test_fp8_vs_bf16_logits_close() -> None:
    """FP8 model's logits should be in the same ball-park as BF16 (not bitwise
    equal; FP8 is lossy). Tolerance is loose — we only catch catastrophic
    numerical errors, not quantisation noise."""
    torch.manual_seed(0)
    bf16_cfg = _fp8_cfg(use_fp8=False)
    bf16_cfg.attention_backend = "sdpa_math"  # cpu-friendly for this quick check
    bf16_model = GPT(bf16_cfg).cuda().bfloat16()

    torch.manual_seed(0)
    fp8_model = GPT(_fp8_cfg()).cuda().bfloat16()

    # Copy weights (both were init'd with the same seed, but te.Linear may
    # store internal scale buffers that differ; explicit copy is safest).
    bf16_sd = bf16_model.state_dict()
    fp8_sd = fp8_model.state_dict()
    for k in bf16_sd:
        if k in fp8_sd and fp8_sd[k].shape == bf16_sd[k].shape:
            fp8_sd[k].copy_(bf16_sd[k])
    fp8_model.load_state_dict(fp8_sd, strict=False)

    x = torch.randint(0, bf16_cfg.vocab_size, (2, bf16_cfg.block_size), device="cuda")
    with torch.no_grad():
        bf16_logits, _ = bf16_model(x, x)
        fp8_logits, _ = fp8_model(x, x)

    # Same argmax on most positions (coarse similarity check).
    bf16_top = bf16_logits.argmax(dim=-1)
    fp8_top = fp8_logits.argmax(dim=-1)
    agreement = (bf16_top == fp8_top).float().mean().item()
    assert agreement > 0.5, f"FP8 and BF16 disagree on most tokens: {agreement:.2%}"


@cuda
def test_fp8_backward_no_nan() -> None:
    """One-step fwd+bwd+opt on the FP8 path must produce finite grads."""
    torch.manual_seed(0)
    cfg = _fp8_cfg(block_size=32, n_layer=2)
    model = GPT(cfg).cuda().bfloat16()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _, loss = model(x, x)
    loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"
    opt.step()
