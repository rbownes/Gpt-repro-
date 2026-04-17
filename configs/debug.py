"""Tiny config for smoke tests — trains in seconds on any hardware."""

from gpt_repro.model import GPTConfig
from gpt_repro.train import TrainConfig


def make_config() -> TrainConfig:
    model = GPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True,
        attention_backend="sdpa_math",   # most portable; no cuDNN path needed
        tie_embeddings=True,
    )
    return TrainConfig(
        model=model,
        data_dir="data/tiny",
        micro_batch=2,
        grad_accum=1,
        block_size=128,
        total_steps=200,
        warmup_steps=10,
        peak_lr=3e-3,
        min_lr_ratio=0.1,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_every=1_000_000,      # no eval during smoke
        eval_batches=2,
        log_every=10,
        ckpt_every=1_000_000,
        keep_last_ckpt=1,
        run_dir="runs/debug",
        seed=0,
        compile=False,
    )
