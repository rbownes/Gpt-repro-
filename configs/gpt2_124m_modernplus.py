"""Modern block + modded-nanogpt tricks (no Muon).

Forks `configs/gpt2_124m_modern.py` (= v0.2-exp01) and layers on:
- ReLU² MLP (2 matrices, hidden = 4·d, same param count as SwiGLU)
- Zero-init out-projections (`c_proj` for attention, `c_proj` for ReLU² MLP)
- U-Net cross-depth skip connections
- Logit softcap at 30.0

Optimizer remains AdamW, as in v0.2-exp01 — this run isolates the modded
tricks from Muon, which was tested separately in exp/02.
"""

from gpt_repro.model import GPTConfig
from gpt_repro.train import TrainConfig


def make_config() -> TrainConfig:
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

        # Modern block (unchanged from v0.2-exp01).
        positional_encoding="rope",
        rope_base=10000.0,
        norm_type="rmsnorm",
        qk_norm=True,

        # Modded-nanogpt tricks (the diff vs v0.2-exp01).
        mlp_type="relu2",
        mlp_hidden=None,             # auto = 4 * 768 = 3072
        zero_init_proj=True,
        u_net_skips=True,
        logit_softcap=30.0,
    )
    return TrainConfig(
        model=model,
        data_dir="data/fineweb_edu_10B",
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
        run_dir="runs/03-modded-tricks",
        seed=0,
        compile=True,
        # Pin to "default" so v0.3-exp03 tagged tok/s numbers (178 k) stay reproducible.
        compile_mode="default",
    )
