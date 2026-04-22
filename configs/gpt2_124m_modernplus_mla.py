"""v0.3-exp03 + Multi-head Latent Attention (DeepSeek-V2).

Single-variable change on top of `gpt2_124m_modernplus.py` (v0.3-exp03):
swap `CausalSelfAttention` (MHA) for `MLAttention` (MLA). All other arch
flags, optimizer (AdamW), schedule, data, batch size are unchanged.

MLA chunk sizes at 124M (n_embd=768, n_head=12, d_head=64):
  d_kv_comp = 256  (= 4·d_head; KV latent bottleneck)
  d_qk_nope = 32   (= d_head/2; per-head no-pe Q/K chunk)
  d_qk_rope = 32   (= d_head/2; shared RoPE-rotated chunk)
  d_v       = 64   (= d_head;   per-head V dim)

This sizing gives a ~124M→116M model (−6.5%). See experiments/10-mla/report.md
for the accounting. Accept criterion is adjusted for the smaller model.
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

        # Modern block (unchanged from v0.3-exp03).
        positional_encoding="rope",
        rope_base=10000.0,
        norm_type="rmsnorm",
        qk_norm=True,

        # Modded-nanogpt tricks (unchanged from v0.3-exp03).
        mlp_type="relu2",
        mlp_hidden=None,                # auto = 4*768 = 3072
        zero_init_proj=True,
        u_net_skips=True,
        logit_softcap=30.0,

        # --- The exp/10 diff ---------------------------------------------
        attention_type="mla",
        mla_d_kv_comp=256,
        mla_d_qk_nope=32,
        mla_d_qk_rope=32,
        mla_d_v=64,
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
        run_dir="runs/10-mla",
        seed=0,
        compile=True,
        compile_mode="default",
    )
