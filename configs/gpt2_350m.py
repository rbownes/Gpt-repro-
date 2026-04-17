"""Faithful GPT-2 medium (350M) — stub; enable after the 124M baseline is accepted."""

from gpt_repro.model import GPTConfig
from gpt_repro.train import TrainConfig


def make_config() -> TrainConfig:
    model = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        dropout=0.0,
        bias=True,
        attention_backend="sdpa_flash",
        tie_embeddings=True,
    )
    return TrainConfig(
        model=model,
        data_dir="data/fineweb_edu_10B",
        micro_batch=8,
        grad_accum=64,
        block_size=1024,
        total_steps=19_073,
        warmup_steps=715,
        peak_lr=3e-4,
        min_lr_ratio=0.1,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_every=500,
        eval_batches=50,
        log_every=10,
        ckpt_every=2000,
        keep_last_ckpt=2,
        run_dir="runs/gpt2_350m",
        seed=0,
        compile=True,
    )
