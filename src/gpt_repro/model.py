"""Faithful GPT-2 decoder-only transformer, with optional modern-block variants.

Defaults reproduce the 2019 GPT-2 architecture exactly (LayerNorm, learned
positional embeddings, GELU MLP, no QK-Norm). Config flags turn on modern
replacements independently:

    positional_encoding: 'learned' | 'rope'
    norm_type:           'layernorm' | 'rmsnorm'
    mlp_type:            'gelu' | 'swiglu' | 'relu2'
    qk_norm:             False | True
    zero_init_proj:      False | True        # modded-nanogpt out-proj zero init
    u_net_skips:         False | True        # cross-depth skip connections
    logit_softcap:       None | float        # s * tanh(logits / s) on head output

The faithful config still parameter-matches Hugging Face's `GPT2LMHeadModel`
so `tests/test_hf_weight_load.py` continues to load released HF `gpt2` weights
into this module and verify logit parity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    attention_backend: str = "sdpa_flash"  # sdpa_flash | sdpa_cudnn | sdpa_math | sdpa_efficient | flash_attn_2
    tie_embeddings: bool = True

    # --- Modern-block flags (faithful defaults) ------------------------------
    positional_encoding: str = "learned"   # 'learned' | 'rope'
    rope_base: float = 10000.0
    norm_type: str = "layernorm"           # 'layernorm' | 'rmsnorm'
    mlp_type: str = "gelu"                 # 'gelu' | 'swiglu' | 'relu2'
    mlp_hidden: int | None = None          # None = auto (4*d for gelu/relu2; explicit for swiglu)
    qk_norm: bool = False

    # --- Modded-nanogpt tricks (all default to faithful/off) -----------------
    zero_init_proj: bool = False           # zero-init attn.c_proj and MLP out-projection
    u_net_skips: bool = False              # second half of layers receive skip from first half
    logit_softcap: float | None = None     # if set, logits go through s*tanh(x/s)


_VALID_BACKENDS = {"sdpa_cudnn", "sdpa_flash", "sdpa_math", "sdpa_efficient", "flash_attn_2"}


def select_sdpa_backend_globally(name: str) -> None:
    """Enable only the chosen SDPA backend via torch globals.

    We do this instead of wrapping each attention forward with
    `torch.nn.attention.sdpa_kernel(...)`, because a per-call context manager
    does not currently compose with `torch.compile` on SM_120 (the inductor
    generates asserts on kernel-specific strides). Setting the backend at
    process init is compile-safe and covers every attention call.
    """
    if name == "flash_attn_2":
        return
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Unknown attention backend: {name!r}")
    be = torch.backends.cuda
    be.enable_flash_sdp(name == "sdpa_flash")
    be.enable_mem_efficient_sdp(name == "sdpa_efficient")
    be.enable_math_sdp(name == "sdpa_math")
    be.enable_cudnn_sdp(name == "sdpa_cudnn")


# --- Norms ------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (Zhang & Sennrich 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate the norm in fp32 for stability under bf16.
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).to(dtype) * self.weight


def make_norm(cfg: GPTConfig, dim: int) -> nn.Module:
    if cfg.norm_type == "layernorm":
        return nn.LayerNorm(dim, bias=cfg.bias)
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {cfg.norm_type!r}")


# --- Rotary position embedding ---------------------------------------------


def rope_freqs(head_dim: int, max_seqlen: int, base: float, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos / sin of shape [max_seqlen, head_dim] for paired-half RoPE."""
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(max_seqlen, dtype=torch.float32, device=device)
    freqs = torch.outer(t, theta)              # [T, D/2]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [T, D]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # [T, D]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to `x` with shape [..., T, D].

    `cos`, `sin` are [T, D] (same T and D as `x`'s last two dims). Rotation is
    paired-halves (Llama convention).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rot * sin).to(x.dtype)


# --- Attention -------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.use_flash_attn_2 = cfg.attention_backend == "flash_attn_2"
        self.use_rope = cfg.positional_encoding == "rope"
        self.use_qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope:
            assert cos is not None and sin is not None, "RoPE requires cos/sin"
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.use_flash_attn_2:
            from flash_attn import flash_attn_func  # type: ignore

            y = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            ).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# --- MLPs ------------------------------------------------------------------


class GELUMLP(nn.Module):
    """Faithful GPT-2 MLP: 2 matrices, hidden = 4 * d, GELU (tanh approx)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_hidden or (4 * cfg.n_embd)
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class ReLU2MLP(nn.Module):
    """ReLU-squared MLP (Primer, So et al. 2021; used in modded-nanogpt).

    Same shape as GELUMLP — two matrices, hidden = 4 * d — just a different
    activation. Empirically competitive with SwiGLU at small scale while
    being cheaper (no gate pre-multiply, two matmuls instead of three).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_hidden or (4 * cfg.n_embd)
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.c_fc(x))
        return self.dropout(self.c_proj(h * h))


class SwiGLU(nn.Module):
    """SwiGLU MLP (Shazeer 2020).

    Three matrices — `w_gate`, `w_up`, `w_down`. With hidden = 8/3 * d, total
    params match the GELU MLP at hidden = 4 * d. For our 124M (d=768), we use
    hidden = 2048 to exactly match 4*768*2 = 2048*3 params.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_hidden
        assert hidden is not None, "SwiGLU requires explicit mlp_hidden"
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


def make_mlp(cfg: GPTConfig) -> nn.Module:
    if cfg.mlp_type == "gelu":
        return GELUMLP(cfg)
    if cfg.mlp_type == "swiglu":
        return SwiGLU(cfg)
    if cfg.mlp_type == "relu2":
        return ReLU2MLP(cfg)
    raise ValueError(f"Unknown mlp_type: {cfg.mlp_type!r}")


# --- Block -----------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.use_rope = cfg.positional_encoding == "rope"
        # Keep the faithful parameter names (ln_1, ln_2, attn, mlp) so the HF
        # weight loader continues to work on the faithful config. The class of
        # `ln_1`/`ln_2` varies with `cfg.norm_type`; HF gpt2 weights only load
        # into LayerNorm-shaped keys, which is the faithful default.
        self.ln_1 = make_norm(cfg, cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = make_norm(cfg, cfg.n_embd)
        self.mlp = make_mlp(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cos=cos, sin=sin)
        x = x + self.mlp(self.ln_2(x))
        return x


# --- Top-level model --------------------------------------------------------


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.use_rope = cfg.positional_encoding == "rope"

        modules: dict[str, nn.Module] = {
            "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
            "drop": nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            "h": nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            "ln_f": make_norm(cfg, cfg.n_embd),
        }
        if not self.use_rope:
            modules["wpe"] = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.transformer = nn.ModuleDict(modules)

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

        if self.use_rope:
            head_dim = cfg.n_embd // cfg.n_head
            cos, sin = rope_freqs(head_dim, cfg.block_size, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Out-projection init:
        # - default: N(0, 0.02 / sqrt(2 * n_layer))  (Radford et al. 2019 scaling)
        # - zero_init_proj=True: literal zeros  (modded-nanogpt / ReZero)
        if cfg.zero_init_proj:
            for pn, p in self.named_parameters():
                if pn.endswith((".c_proj.weight", ".w_down.weight")):
                    nn.init.zeros_(p)
        else:
            scale = 1.0 / math.sqrt(2 * cfg.n_layer)
            for pn, p in self.named_parameters():
                if pn.endswith((".c_proj.weight", ".w_down.weight")):
                    nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n -= self.transformer.wpe.weight.numel()
        return n

    def _maybe_softcap(self, logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.logit_softcap is not None:
            s = self.cfg.logit_softcap
            return s * torch.tanh(logits / s)
        return logits

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"

        x = self.transformer.wte(idx)
        if not self.use_rope:
            pos = torch.arange(T, device=idx.device, dtype=torch.long)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)

        if self.use_rope:
            cos = self.rope_cos[:T]
            sin = self.rope_sin[:T]
        else:
            cos = sin = None

        blocks = self.transformer.h
        if self.cfg.u_net_skips:
            # U-Net style cross-depth shortcuts: layer i ≥ n/2 adds the
            # post-block output of layer n-1-i to its input. First half
            # pushes onto a LIFO, second half pops. Matches modded-nanogpt.
            n = len(blocks)
            half = n // 2
            skip_stack: list[torch.Tensor] = []
            for i, block in enumerate(blocks):
                if i >= half:
                    x = x + skip_stack.pop()
                x = block(x, cos=cos, sin=sin)
                if i < half:
                    skip_stack.append(x)
        else:
            for block in blocks:
                x = block(x, cos=cos, sin=sin)
        x = self.transformer.ln_f(x)

        if targets is None:
            logits = self._maybe_softcap(self.lm_head(x[:, [-1], :]))
            return logits, None

        logits = self._maybe_softcap(self.lm_head(x))
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    # ---- HF weight loading (only valid on faithful config) ------------------
    @classmethod
    def from_pretrained_gpt2(
        cls,
        model_name: str = "gpt2",
        *,
        attention_backend: str = "sdpa_math",
    ) -> "GPT":
        """Load released HF GPT-2 weights into this module (faithful config only).

        HF's `transformers.pytorch_utils.Conv1D` stores weights as the transpose
        of `nn.Linear`, so we explicitly transpose on copy.
        """
        from transformers import GPT2LMHeadModel  # type: ignore

        presets = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        cfg = GPTConfig(
            **presets[model_name],
            vocab_size=50257,
            block_size=1024,
            attention_backend=attention_backend,
            # All modern flags left at faithful defaults.
        )
        model = cls(cfg)

        hf = GPT2LMHeadModel.from_pretrained(model_name)
        hf_sd = hf.state_dict()
        sd = model.state_dict()

        hf_skip = {"lm_head.weight"}
        transpose_suffixes = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        for k, v in hf_sd.items():
            if k in hf_skip:
                continue
            if k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"):
                continue
            assert k in sd, f"HF key {k!r} not in our state dict"
            target = sd[k]
            src = v.t() if k.endswith(transpose_suffixes) else v
            assert src.shape == target.shape, f"{k}: HF {tuple(src.shape)} vs ours {tuple(target.shape)}"
            with torch.no_grad():
                target.copy_(src)
        return model
