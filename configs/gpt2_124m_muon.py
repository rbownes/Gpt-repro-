"""Modern block + Muon on hidden matmuls (AdamW elsewhere).

Forks `configs/gpt2_124m_modern.py` (= v0.2-exp01). Only the optimizer path
changes: hidden 2-D matmul weights go to Muon (peak_lr 0.02), everything
else stays on AdamW (peak_lr 6e-4). Schedule, data, token budget unchanged.
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
        mlp_type="swiglu",
        mlp_hidden=2048,
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
        # AdamW peak LR (for embeddings, norms, biases).
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
        # Muon on hidden matmuls.
        optimizer_type="muon+adamw",
        muon_peak_lr=0.02,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        run_dir="runs/02-muon",
        seed=0,
        compile=True,
    )
