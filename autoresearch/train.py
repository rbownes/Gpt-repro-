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
PEAK_LR = 5e-4
MIN_LR_RATIO = 1.0
WEIGHT_DECAY = 0.0
BETA1 = 0.9
BETA2 = 0.99
EPS = 1e-8
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.02            # fraction of budget spent warming LR up (~6 s @ 300 s)
SEED = 0

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


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
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


def lr_at_progress(progress: float) -> float:
    """Time-based LR: linear warmup to PEAK_LR, then cosine decay to MIN_LR_RATIO*PEAK_LR."""
    progress = max(0.0, min(1.0, progress))
    if progress < WARMUP_FRAC:
        return PEAK_LR * progress / WARMUP_FRAC
    rel = (progress - WARMUP_FRAC) / max(1e-9, 1.0 - WARMUP_FRAC)
    cos = 0.5 * (1.0 + math.cos(math.pi * rel))
    return PEAK_LR * (MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cos)


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
        lr = lr_at_progress(elapsed / TIME_BUDGET)
        for g in optimizer.param_groups:
            g["lr"] = lr

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
