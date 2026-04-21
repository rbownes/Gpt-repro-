"""Autoresearch single-file trainer — Phase A (speed).

The agent edits ONLY this file. Reads frozen data prep from `prepare.py`,
trains for exactly TIME_BUDGET seconds on FineWeb-Edu-10B with a v0.3-style
(modern + modded-tricks) GPT-2 124M, and prints a structured summary at
the end:

    val_bpb: <float>
    tok_per_sec: <float>
    peak_vram_mb: <float>
    num_steps: <int>
    training_seconds: <float>
    total_seconds: <float>

These lines are grep'd by the research loop to build `results.tsv`.

Don't touch `prepare.py` (eval math is ground truth). Don't change any
model-architecture constant marked # FROZEN_ARCH — those decisions are
quality, not speed.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# CONFIG
# ============================================================================

# ---- Budget ----------------------------------------------------------------
TIME_BUDGET = 300.0          # seconds of pure training wall-clock

# ---- Data ------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fineweb_edu_10B"
TOKEN_BYTES_PATH = DATA_DIR / "token_bytes.pt"

# ---- Model architecture (FROZEN — quality decisions, not speed) ------------
VOCAB_SIZE = 50304           # FROZEN_ARCH: padded from 50257 for matmul alignment
BLOCK_SIZE = 1024            # FROZEN_ARCH
N_LAYER = 12                 # FROZEN_ARCH
N_HEAD = 12                  # FROZEN_ARCH
N_KV_HEAD = 12               # FROZEN_ARCH (= N_HEAD ⇒ MHA; GQA hurts quality at 124M)
N_EMBD = 768                 # FROZEN_ARCH
MLP_HIDDEN = 3072            # FROZEN_ARCH (4*N_EMBD, ReLU² 2-matrix MLP)
ROPE_BASE = 10000.0          # FROZEN_ARCH
USE_RMSNORM = True           # FROZEN_ARCH (v0.3)
USE_QK_NORM = True           # FROZEN_ARCH (v0.3)
USE_ZERO_INIT_PROJ = True    # FROZEN_ARCH (v0.3)
USE_UNET_SKIPS = True        # FROZEN_ARCH (v0.3)
LOGIT_SOFTCAP = 30.0         # FROZEN_ARCH (v0.3); set None to disable

# ---- Training mechanics (speed knobs — agent MAY edit these) ---------------
MICRO_BATCH = 16
GRAD_ACCUM = 2               # effective batch = MICRO_BATCH * GRAD_ACCUM * BLOCK_SIZE
PEAK_LR = 9e-4
MIN_LR_RATIO = 0.3
WEIGHT_DECAY = 0.0
BETA1 = 0.9
BETA2 = 0.9995
EPS = 1e-8
GRAD_CLIP = 0.5
WARMUP_FRAC = 0.04            # fraction of budget spent warming LR up (~12 s @ 300 s)
SEED = 0

# ---- Optimizer choice (nanochat-inspired MuonAdamW) ------------------------
USE_MUON = True               # True: Muon on 2D block matrices, AdamW on embeddings/1D
MUON_LR = 0.02                # Muon base LR (Kimi/Keller defaults; scaled per-shape)
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
MUON_BETA2 = 0.9

# ---- Compute / kernels (speed knobs) ---------------------------------------
COMPILE = True
COMPILE_MODE = "max-autotune-no-cudagraphs"  # default | max-autotune | reduce-overhead | max-autotune-no-cudagraphs
ATTENTION_BACKEND = "sdpa_flash"  # sdpa_flash | sdpa_cudnn | sdpa_math | sdpa_efficient | flash_attn_2

# ---- Evaluation (speed knob within reason; changes val_bpb variance) ------
EVAL_TOKENS = 2_000_000      # ≈ 4 batches × 1024 × 16 × 30 → gives low-variance BPB

# ============================================================================
# UTILITIES
# ============================================================================


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tune_pytorch_globals() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def select_sdpa_backend(name: str) -> None:
    if name == "flash_attn_2":
        return
    be = torch.backends.cuda
    be.enable_flash_sdp(name == "sdpa_flash")
    be.enable_mem_efficient_sdp(name == "sdpa_efficient")
    be.enable_math_sdp(name == "sdpa_math")
    be.enable_cudnn_sdp(name == "sdpa_cudnn")


def human_time(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ============================================================================
# MODEL
# ============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).to(dtype) * self.weight


def rope_freqs(head_dim: int, max_seqlen: int, base: float, device=None):
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(max_seqlen, dtype=torch.float32, device=device)
    freqs = torch.outer(t, theta)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rot * sin).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.head_dim = N_EMBD // N_HEAD
        self.q_dim = N_HEAD * self.head_dim
        self.kv_dim = N_KV_HEAD * self.head_dim
        self.c_attn = nn.Linear(N_EMBD, self.q_dim + 2 * self.kv_dim, bias=True)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=True)
        if USE_QK_NORM:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.enable_gqa = N_KV_HEAD < N_HEAD

    def forward(self, x, cos=None, sin=None):
        B, T, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=2)
        q = q.view(B, T, N_HEAD, self.head_dim).transpose(1, 2)
        k = k.view(B, T, N_KV_HEAD, self.head_dim).transpose(1, 2)
        v = v.view(B, T, N_KV_HEAD, self.head_dim).transpose(1, 2)
        if USE_QK_NORM:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if ATTENTION_BACKEND == "flash_attn_2":
            from flash_attn import flash_attn_func  # type: ignore

            y = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, causal=True,
            ).transpose(1, 2)
        else:
            sdpa_kwargs = {"is_causal": True, "dropout_p": 0.0}
            if self.enable_gqa:
                sdpa_kwargs["enable_gqa"] = True
            y = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.q_dim)
        return self.c_proj(y)


class ReLU2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, MLP_HIDDEN, bias=True)
        self.c_proj = nn.Linear(MLP_HIDDEN, N_EMBD, bias=True)

    def forward(self, x):
        h = F.relu(self.c_fc(x))
        return self.c_proj(h * h)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = RMSNorm(N_EMBD) if USE_RMSNORM else nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = RMSNorm(N_EMBD) if USE_RMSNORM else nn.LayerNorm(N_EMBD)
        self.mlp = ReLU2MLP()

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.ln_1(x), cos=cos, sin=sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBD) if USE_RMSNORM else nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight  # tied
        cos, sin = rope_freqs(N_EMBD // N_HEAD, BLOCK_SIZE, ROPE_BASE)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # Out-projection init: either zero-init (modded-nanogpt) or 1/sqrt(2L) scaled normal.
        if USE_ZERO_INIT_PROJ:
            for pn, p in self.named_parameters():
                if pn.endswith((".attn.c_proj.weight", ".mlp.c_proj.weight")):
                    nn.init.zeros_(p)
        else:
            scale = 1.0 / math.sqrt(2 * N_LAYER)
            for pn, p in self.named_parameters():
                if pn.endswith((".attn.c_proj.weight", ".mlp.c_proj.weight")):
                    nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        if USE_UNET_SKIPS:
            half = N_LAYER // 2
            stack = []
            for i, block in enumerate(self.blocks):
                if i >= half:
                    x = x + stack.pop()
                x = block(x, cos=cos, sin=sin)
                if i < half:
                    stack.append(x)
        else:
            for block in self.blocks:
                x = block(x, cos=cos, sin=sin)
        x = self.ln_f(x)
        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            if LOGIT_SOFTCAP is not None:
                logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
            return logits, None
        logits = self.lm_head(x)
        if LOGIT_SOFTCAP is not None:
            logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="mean",
        )
        return logits, loss


# ============================================================================
# DATA
# ============================================================================


class ShardLoader:
    """Random-access loader over uint16 memmap shards (matches scripts/prepare_fineweb_edu.py).

    Uses a pre-allocated pinned-memory pair of (x, y) buffers so the H2D
    copy via `.to(device, non_blocking=True)` is genuinely async and can
    overlap with the previous batch's forward/backward.
    """

    def __init__(self, data_dir: Path, split: str, block_size: int, batch_size: int):
        import glob
        paths = sorted(glob.glob(str(data_dir / f"{split}_*.bin")))
        if not paths:
            raise FileNotFoundError(f"No {split}_*.bin shards in {data_dir}")
        self.memmaps = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.block_size = block_size
        self.batch_size = batch_size
        self.total_tokens = sum(m.shape[0] for m in self.memmaps)
        # Pinned CPU tensors — source of the async H2D copy.
        self._pin_x = torch.empty((batch_size, block_size), dtype=torch.int64, pin_memory=True)
        self._pin_y = torch.empty((batch_size, block_size), dtype=torch.int64, pin_memory=True)
        self._pin_x_np = self._pin_x.numpy()  # aliased view for fast memmap copies
        self._pin_y_np = self._pin_y.numpy()

    def next_batch(self, rng: np.random.Generator, device):
        for i in range(self.batch_size):
            shard = self.memmaps[rng.integers(len(self.memmaps))]
            start = int(rng.integers(0, shard.shape[0] - self.block_size - 1))
            seq = shard[start : start + self.block_size + 1]
            self._pin_x_np[i] = seq[:-1]
            self._pin_y_np[i] = seq[1:]
        x = self._pin_x.to(device, non_blocking=True)
        y = self._pin_y.to(device, non_blocking=True)
        return x, y

    def iter_val(self, device, max_batches=None):
        shard = self.memmaps[0]
        pos = 0
        n = 0
        stride = self.block_size
        while pos + self.batch_size * stride + 1 <= shard.shape[0]:
            buf = np.stack([
                np.asarray(shard[pos + i * stride : pos + i * stride + self.block_size + 1], dtype=np.int64)
                for i in range(self.batch_size)
            ])
            pos += self.batch_size * stride
            x = torch.from_numpy(buf[:, :-1]).to(device, non_blocking=True)
            y = torch.from_numpy(buf[:, 1:]).to(device, non_blocking=True)
            yield x, y
            n += 1
            if max_batches is not None and n >= max_batches:
                return


# ============================================================================
# EVALUATION — FROZEN MATH (mirrors evaluate_bpb from prepare.py semantics)
# ============================================================================


@torch.no_grad()
def evaluate_bpb(model, val_loader, token_bytes: torch.Tensor, device, amp_dtype) -> float:
    """Bits per byte on the held-out val shard.

    Computes per-token cross-entropy (nats) then converts to bits/byte using
    UTF-8 byte lengths of target tokens. Special tokens (byte length 0) are
    excluded from both numerator and denominator.
    """
    model.eval()
    total_nats = 0.0
    total_bytes = 0
    steps = max(1, EVAL_TOKENS // (MICRO_BATCH * BLOCK_SIZE))
    for x, y in val_loader.iter_val(device, max_batches=steps):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(x, y)
        loss_flat = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            y.view(-1),
            ignore_index=-1,
            reduction="none",
        )
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += int(nbytes.sum().item())
    model.train()
    if total_bytes == 0:
        return float("nan")
    return total_nats / (math.log(2) * total_bytes)


# ============================================================================
# OPTIMIZER + SCHEDULE
# ============================================================================


# ----------------------------------------------------------------------------
# MuonAdamW — ported from nanochat/optim.py (single-GPU version).
# Muon handles 2D matrix parameters (block Linears); AdamW handles embeddings
# and 1D tensors (norms, biases). Both steps are torch.compile'd with 0-D CPU
# hyperparam tensors so LR/beta changes don't trigger recompilation.
# ----------------------------------------------------------------------------

_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def _adamw_step_fused(p, grad, exp_avg, exp_avg_sq,
                     step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


@torch.compile(dynamic=False, fullgraph=True)
def _muon_step_fused(stacked_grads, stacked_params, momentum_buffer,
                    second_momentum_buffer, momentum_t, lr_t, wd_t, beta2_t,
                    ns_steps: int, red_dim: int):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar Express orthogonalisation (bf16 for speed)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_sz = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_sz
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_sz) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined Muon (2D matrices) + AdamW (embeddings/1D) optimiser.

    Each group dict must carry 'kind': 'muon' or 'adamw'. Muon groups must
    have all params of the same shape (we pre-group by shape in build).
    """

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors — mutated by .fill_() so values change without
        # invalidating torch.compile's captured graph.
        self._aw = {k: torch.tensor(0.0) for k in
                    ("step", "lr", "beta1", "beta2", "eps", "wd")}
        self._mu = {k: torch.tensor(0.0) for k in
                    ("momentum", "lr", "wd", "beta2")}

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            if not st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            self._aw["step"].fill_(st["step"])
            self._aw["lr"].fill_(group["lr"])
            self._aw["beta1"].fill_(group["betas"][0])
            self._aw["beta2"].fill_(group["betas"][1])
            self._aw["eps"].fill_(group["eps"])
            self._aw["wd"].fill_(group["weight_decay"])
            _adamw_step_fused(p, p.grad, st["exp_avg"], st["exp_avg_sq"],
                             self._aw["step"], self._aw["lr"],
                             self._aw["beta1"], self._aw["beta2"],
                             self._aw["eps"], self._aw["wd"])

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        st = self.state[p0]
        n, shape, device, dtype = len(params), p0.shape, p0.device, p0.dtype
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros(n, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in st:
            sm_shape = (n, shape[-2], 1) if shape[-2] >= shape[-1] else (n, 1, shape[-1])
            st["second_momentum_buffer"] = torch.zeros(sm_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._mu["momentum"].fill_(group["momentum"])
        self._mu["beta2"].fill_(group["beta2"])
        self._mu["lr"].fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._mu["wd"].fill_(group["weight_decay"])
        _muon_step_fused(stacked_grads, stacked_params,
                        st["momentum_buffer"], st["second_momentum_buffer"],
                        self._mu["momentum"], self._mu["lr"],
                        self._mu["wd"], self._mu["beta2"],
                        group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            if g["kind"] == "adamw":
                self._step_adamw(g)
            elif g["kind"] == "muon":
                self._step_muon(g)
            else:
                raise ValueError(f"Unknown optim kind: {g['kind']}")


def build_optimizer(model: nn.Module):
    if not USE_MUON:
        decay, no_decay = [], []
        for _n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.ndim >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        try:
            return torch.optim.AdamW(groups, lr=PEAK_LR, betas=(BETA1, BETA2), eps=EPS, fused=True)
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(groups, lr=PEAK_LR, betas=(BETA1, BETA2), eps=EPS)

    # MuonAdamW: embeddings + 1D → AdamW; 2D block matrices → Muon (grouped by shape)
    adamw_params = []
    muon_by_shape: dict = {}
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if name.startswith(("wte", "lm_head")) or p.ndim != 2:
            adamw_params.append(p)
        else:
            muon_by_shape.setdefault(tuple(p.shape), []).append(p)

    groups = [dict(kind="adamw", params=adamw_params,
                   lr=PEAK_LR, betas=(BETA1, BETA2), eps=EPS,
                   weight_decay=WEIGHT_DECAY, base_lr=PEAK_LR)]
    for shape, ps in muon_by_shape.items():
        groups.append(dict(kind="muon", params=ps,
                           lr=MUON_LR, momentum=MUON_MOMENTUM,
                           ns_steps=MUON_NS_STEPS, beta2=MUON_BETA2,
                           weight_decay=WEIGHT_DECAY, base_lr=MUON_LR))
    return MuonAdamW(groups)


def lr_frac_at_progress(progress: float) -> float:
    """Returns the LR multiplier in [0, 1] — linear warmup → cosine to MIN_LR_RATIO."""
    progress = max(0.0, min(1.0, progress))
    if progress < WARMUP_FRAC:
        return progress / WARMUP_FRAC
    rel = (progress - WARMUP_FRAC) / max(1e-9, 1.0 - WARMUP_FRAC)
    cos = 0.5 * (1.0 + math.cos(math.pi * rel))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cos


def lr_at_progress(progress: float) -> float:
    """Absolute AdamW LR (legacy single-optimizer path)."""
    return PEAK_LR * lr_frac_at_progress(progress)


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    t_total_start = time.monotonic()
    tune_pytorch_globals()
    seed_everything(SEED)

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    select_sdpa_backend(ATTENTION_BACKEND)

    # Data
    train_loader = ShardLoader(DATA_DIR, "train", BLOCK_SIZE, MICRO_BATCH)
    val_loader = ShardLoader(DATA_DIR, "val", BLOCK_SIZE, MICRO_BATCH)
    if not TOKEN_BYTES_PATH.exists():
        raise FileNotFoundError(f"Missing {TOKEN_BYTES_PATH}. Run autoresearch/prepare.py.")
    token_bytes = torch.load(TOKEN_BYTES_PATH, weights_only=True).to(device)

    # Model
    model = GPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = build_optimizer(model)

    # Compile
    if COMPILE:
        model_fwd = torch.compile(model, mode=COMPILE_MODE) if COMPILE_MODE != "default" else torch.compile(model)
    else:
        model_fwd = model
    print(f"compile: {COMPILE} mode={COMPILE_MODE} | sdpa: {ATTENTION_BACKEND}")

    tokens_per_step = MICRO_BATCH * GRAD_ACCUM * BLOCK_SIZE
    amp_dtype = torch.bfloat16
    rng = np.random.default_rng(SEED)

    # Training
    t_train_start = time.monotonic()
    total_tokens = 0
    num_steps = 0
    last_log_t = t_train_start
    last_log_step = 0
    smooth_loss = None
    # "Warm" throughput excludes the first 5% of budget (compile warmup bleed).
    warm_start_t = None
    warm_start_tokens = 0
    warm_start_steps = 0

    print(f"budget: {TIME_BUDGET:.0f}s | tokens/step: {tokens_per_step:,}")

    while True:
        elapsed = time.monotonic() - t_train_start
        if elapsed >= TIME_BUDGET:
            break
        frac = lr_frac_at_progress(elapsed / TIME_BUDGET)
        for g in optimizer.param_groups:
            # MuonAdamW groups carry 'base_lr'; legacy AdamW groups don't.
            base = g.get("base_lr", PEAK_LR)
            g["lr"] = base * frac
        lr = PEAK_LR * frac  # retained for print

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for _ in range(GRAD_ACCUM):
            x, y = train_loader.next_batch(rng, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model_fwd(x, y)
            loss = loss / GRAD_ACCUM
            loss.backward()
            loss_accum += loss.detach()

        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        step_loss = float(loss_accum)
        if not math.isfinite(step_loss) or step_loss > 100:
            print(f"abort: loss {step_loss} at step {num_steps+1}; something is broken")
            return 1
        smooth_loss = step_loss if smooth_loss is None else (0.9 * smooth_loss + 0.1 * step_loss)
        num_steps += 1
        total_tokens += tokens_per_step

        # Mark the end of compile warmup (~5 % of budget) so steady-state tok/s
        # isn't dragged down by the big first-step compile spike.
        if warm_start_t is None and (time.monotonic() - t_train_start) >= 0.05 * TIME_BUDGET:
            warm_start_t = time.monotonic()
            warm_start_tokens = total_tokens
            warm_start_steps = num_steps

        now = time.monotonic()
        if now - last_log_t >= 10.0 or num_steps <= 3:
            tok_s = (num_steps - last_log_step) * tokens_per_step / max(1e-9, now - last_log_t)
            print(f"step {num_steps:>4} | loss {smooth_loss:.4f} | lr {lr:.2e} "
                  f"| {tok_s/1e3:.1f}k tok/s | {human_time(elapsed)}")
            last_log_t = now
            last_log_step = num_steps

    t_train_end = time.monotonic()
    training_seconds = t_train_end - t_train_start

    # Eval
    print(f"eval: evaluating val_bpb on {EVAL_TOKENS:,} tokens ...")
    val_bpb = evaluate_bpb(model_fwd, val_loader, token_bytes, device, amp_dtype)
    # Full (includes compile warmup) and warm (excludes it) throughput numbers.
    tok_per_sec_full = total_tokens / max(1e-9, training_seconds)
    if warm_start_t is not None and num_steps > warm_start_steps:
        warm_dt = t_train_end - warm_start_t
        warm_tokens = total_tokens - warm_start_tokens
        tok_per_sec_warm = warm_tokens / max(1e-9, warm_dt)
    else:
        tok_per_sec_warm = tok_per_sec_full
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    total_seconds = time.monotonic() - t_total_start

    print()
    print("=" * 64)
    print(f"val_bpb: {val_bpb:.6f}")
    print(f"tok_per_sec: {tok_per_sec_warm:.1f}")  # post-warmup, comparable across trials
    print(f"tok_per_sec_full: {tok_per_sec_full:.1f}")  # including compile warmup
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"num_steps: {num_steps}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds: {total_seconds:.1f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
