"""v0.3-exp03 with weight-tied looped transformer (LoopLLM, exp/11).

Forks `configs/gpt2_124m_modernplus.py` (= v0.3-exp03). Two changes:
- `weight_tied = True`: one shared Block instance, applied `n_layer` (= 12)
  times during forward. Parameter count drops from ~124 M → ~45 M (−63 %).
- `u_net_skips = False`: REQUIRED when `weight_tied=True` (every "layer" is
  the same module, so the U-Net cross-depth info-flow pattern is degenerate).

All other hyperparameters (batch, schedule, optimizer, data, token budget,
MLP, softcap, etc.) are unchanged from v0.3.

This experiment knowingly turns off one v0.3-accepted feature (U-Net). If
the val-loss hits the "informative-reject" band (3.050 > val > 2.984) or
the hard-reject bar (> 3.100), a follow-up control run at
(weight_tied=False, u_net_skips=False) would be needed to attribute loss
between "LoopLLM cost" vs "removing U-Net". See experiments/11-loopllm/report.md
for pre-declared criteria.
"""

from gpt_repro.model import GPTConfig
from gpt_repro.train import TrainConfig


def make_config() -> TrainConfig:
    model = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,                    # effective depth; 1 shared block × 12 iters
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        attention_backend="sdpa_flash",
        tie_embeddings=True,

        # Modern block (unchanged from v0.3-exp03).
        positional_encoding="rope",
        rope_base=10000.0,
        norm_type="rmsnorm",
        qk_norm=True,

        # Modded-nanogpt tricks: keep ReLU², zero-init, softcap; U-Net off under tying.
        mlp_type="relu2",
        mlp_hidden=None,               # auto = 4*768 = 3072
        zero_init_proj=True,
        u_net_skips=False,             # REQUIRED off under weight_tied
        logit_softcap=30.0,

        # --- exp/11 diff -------------------------------------------------
        weight_tied=True,
    )
    return TrainConfig(
        model=model,
        data_dir="data/fineweb_edu_10B",
        micro_batch=16,
        grad_accum=32,
        block_size=1024,
        total_steps=19_073,            # ~10 B tokens
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
        run_dir="runs/11-loopllm",
        seed=0,
        compile=True,
        compile_mode="default",
    )
