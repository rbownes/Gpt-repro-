"""v0.3-exp03 + MuonAdamW optimiser + μP plumbing (no-op at base width).

Forks `configs/gpt2_124m_modernplus.py` (= v0.3-exp03). The model config is
identical — architecture is frozen (RoPE + RMSNorm + QK-Norm + ReLU² +
zero-init + U-Net + logit_softcap). Only the optimiser and μP plumbing change.

Why μP plumbing is here even though it's a no-op at 124M:
    base_width == target_width ⇒ every `mup_width_mult = 1.0` ⇒ LR scales
    are 1.0. Mathematically identical to the non-μP path at this width, but
    the machinery is in place so future 350M or 20M-proxy scale-ups can use
    μTransfer without a codebase change.

HPs kept at v0.3 values (peak_lr=6e-4, beta2=0.95, grad_accum=32). The
autoresearch batch-32 HP findings are regime-specific and are NOT promoted
here — they'd only apply if we also shrunk effective batch.
"""

from gpt_repro.model import GPTConfig
from gpt_repro.train import TrainConfig


def make_config() -> TrainConfig:
    # Arch: byte-identical to gpt2_124m_modernplus.py (v0.3-exp03).
    model = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        attention_backend="sdpa_flash",
        tie_embeddings=True,

        positional_encoding="rope",
        rope_base=10000.0,
        norm_type="rmsnorm",
        qk_norm=True,

        mlp_type="relu2",
        mlp_hidden=None,             # auto = 4 * 768 = 3072
        zero_init_proj=True,
        u_net_skips=True,
        logit_softcap=30.0,
    )
    return TrainConfig(
        model=model,
        data_dir="data/fineweb_edu_10B",
        # Batching / schedule: unchanged from v0.3-exp03.
        micro_batch=16,
        grad_accum=32,
        block_size=1024,
        total_steps=19_073,
        warmup_steps=715,
        peak_lr=6e-4,
        min_lr_ratio=0.1,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip=1.0,
        eval_every=500,
        eval_batches=50,
        log_every=10,
        ckpt_every=2000,
        keep_last_ckpt=2,
        # --- exp/06 diff ------------------------------------------------
        optimizer="muon_adamw",
        muon_lr=0.02,                # nanochat default; per-shape √(fan) scaled inside step
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_beta2=0.9,
        use_mup=True,
        mup_base_shapes_path=None,   # self-base at 124M ⇒ no-op
        # ---------------------------------------------------------------
        run_dir="runs/06-muon-mup",
        seed=0,
        compile=True,
        compile_mode="default",
    )
