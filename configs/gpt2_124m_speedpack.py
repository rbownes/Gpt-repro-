"""v0.3 modernplus + four speed-oriented changes, AdamW, single RTX 5090.

Forks v0.3-exp03 and:
  - switches `compile_mode` to "max-autotune-no-cudagraphs" (+6% measured)
  - turns on Liger fused linear cross-entropy (no [B*T, V] logits materialised)
  - turns softcap off (fused CE can't coexist; also suspected to cost HellaSwag)
  - enables GQA with 4 KV heads (from 12)

Everything else (data, optimizer, schedule, token budget, seed) unchanged.
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

        # v0.3 modern + modded-tricks block (inherited).
        positional_encoding="rope",
        rope_base=10000.0,
        norm_type="rmsnorm",
        qk_norm=True,
        mlp_type="relu2",
        mlp_hidden=None,              # auto = 4 * 768 = 3072
        zero_init_proj=True,
        u_net_skips=True,

        # Speed-pack changes (Liger fused CE dropped — incompatible with
        # torch.compile on SM_120; see report for detail).
        logit_softcap=None,           # suspected HellaSwag regressor in v0.3
        n_kv_head=4,                  # GQA: 12 Q heads share 4 KV heads (3× group)
        use_liger_fused_ce=False,     # kept available as a flag for non-compiled paths
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
        run_dir="runs/05-speed-pack",
        seed=0,
        compile=True,
        compile_mode="max-autotune-no-cudagraphs",
    )
