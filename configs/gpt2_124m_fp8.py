"""v0.3 modern+modded tricks + FP8 matmul on hidden matrices.

Forks `configs/gpt2_124m_modernplus.py` (= v0.3-exp03). The only change is
`use_fp8=True`; everything else (arch, optimizer, schedule, data, seed,
compile mode) stays identical so the single-variable comparison is clean.
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

        # v0.3 modern + modded-tricks block.
        positional_encoding="rope",
        rope_base=10000.0,
        norm_type="rmsnorm",
        qk_norm=True,
        mlp_type="relu2",
        mlp_hidden=None,             # auto = 4 * 768 = 3072
        zero_init_proj=True,
        u_net_skips=True,
        logit_softcap=30.0,

        # FP8.
        use_fp8=True,
        fp8_recipe="delayed_hybrid",
        fp8_amax_history_len=16,
        fp8_amax_compute_algo="max",
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
        run_dir="runs/04-fp8",
        seed=0,
        compile=True,
        compile_mode="default",
    )
