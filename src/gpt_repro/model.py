"""Faithful GPT-2 decoder-only transformer.

Parameter layout intentionally matches Hugging Face's `GPT2LMHeadModel` so that
`tests/test_hf_weight_load.py` can load the released HF `gpt2` weights into this
module and compare logits for architectural parity.
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
        # External kernel; we don't need to restrict torch's SDPA routing.
        return
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Unknown attention backend: {name!r}")
    be = torch.backends.cuda
    # Disable all, then enable the one we want.
    be.enable_flash_sdp(name == "sdpa_flash")
    be.enable_mem_efficient_sdp(name == "sdpa_efficient")
    be.enable_math_sdp(name == "sdpa_math")
    be.enable_cudnn_sdp(name == "sdpa_cudnn")


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_flash_attn_2:
            # Opt-in external kernel. Imported lazily so core deps stay small.
            from flash_attn import flash_attn_func  # type: ignore

            y = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            ).transpose(1, 2)
        else:
            # SDPA backend was selected globally at process init via
            # `select_sdpa_backend_globally` (torch globals compose with
            # torch.compile, unlike per-forward sdpa_kernel context managers).
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        # Residual projection init per GPT-2 paper: scale by 1/sqrt(2*n_layer).
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
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
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is None:
            # Inference path: only compute logits on the last position to save memory.
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

        logits = self.lm_head(x)
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

    # ---- HF weight loading (for correctness parity tests) -------------------
    @classmethod
    def from_pretrained_gpt2(
        cls,
        model_name: str = "gpt2",
        *,
        attention_backend: str = "sdpa_math",
    ) -> "GPT":
        """Load released HF GPT-2 weights into this module.

        HF uses `transformers.pytorch_utils.Conv1D` instead of `nn.Linear` in the
        attention/MLP projections; its `.weight` is the transpose of a Linear's.
        We copy with an explicit transpose so our `nn.Linear` modules match
        HF's computation exactly.

        `attention_backend` defaults to `sdpa_math` because this path is used
        mainly for correctness checks that often run on CPU; pass
        `attention_backend="sdpa_cudnn"` when loading for GPU inference.
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
        )
        model = cls(cfg)

        hf = GPT2LMHeadModel.from_pretrained(model_name)
        hf_sd = hf.state_dict()
        sd = model.state_dict()

        # Keys we do not expect to find / do not want to copy.
        hf_skip = {"lm_head.weight"}  # tied; copied from wte
        # HF Conv1D weights are transposed vs nn.Linear.
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
                # HF stores the causal mask as a buffer; we use SDPA is_causal instead.
                continue
            assert k in sd, f"HF key {k!r} not in our state dict"
            target = sd[k]
            src = v.t() if k.endswith(transpose_suffixes) else v
            assert src.shape == target.shape, f"{k}: HF {tuple(src.shape)} vs ours {tuple(target.shape)}"
            with torch.no_grad():
                target.copy_(src)
        return model
