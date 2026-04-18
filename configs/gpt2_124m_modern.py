"""Modern-block GPT-2 124M: RoPE + RMSNorm + SwiGLU + QK-Norm.

Everything else (data, optimiser, schedule, token budget, hardware, precision)
is identical to `configs/gpt2_124m.py`. Only the decoder block changes.
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

        # Modern block.
        positional_encoding="rope",
        rope_base=10000.0,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        mlp_hidden=2048,     # 3 matrices * 768 * 2048 == 2 matrices * 768 * 3072; param parity
        qk_norm=True,
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
        run_dir="runs/01-modern-block",
        seed=0,
        compile=True,
    )
