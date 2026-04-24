"""Supervised fine-tune a pretrained GPT checkpoint on SmolTalk.

Usage:
    uv run python scripts/sft.py \\
        --pretrain-ckpt runs/03-modded-tricks/best_val.pt \\
        --run-dir runs/sft-03-modded-tricks \\
        --sft-tokens 500000000

Loads the checkpoint's saved GPTConfig via the schema-drift-tolerant helper,
rebuilds the model on this branch (which has every arch feature merged),
then runs a text-marker-template SFT using SmolTalk. Optimizer is fresh
AdamW regardless of pretrain optimizer — we want the SFT side held
constant across the exp/14 matrix so pretrain choice is the only varying
axis.

Schedule: linear warmup → constant peak → linear warmdown to
`final_lr_frac × peak_lr`. No cosine (too short a run to benefit).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make `from gpt_repro...` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt_repro.model import GPT, select_sdpa_backend_globally  # noqa: E402
from gpt_repro.sft_data import SFTDataLoader  # noqa: E402
from gpt_repro.tasks import SmolTalk  # noqa: E402
from gpt_repro.utils import (  # noqa: E402
    JSONLLogger,
    autocast_dtype,
    device_str,
    ensure_dir,
    env_fingerprint,
    human_time,
    load_gpt_config_from_ckpt,
    rotate_checkpoints,
    save_checkpoint,
    seed_everything,
    tune_pytorch_globals,
    write_env_fingerprint,
)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def sft_lr_frac(step: int, *, total: int, warmup: int, warmdown_start: int, final_frac: float) -> float:
    """Linear warmup → constant peak → linear warmdown to `final_frac`."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if step < warmdown_start:
        return 1.0
    # Linear warmdown from 1.0 at warmdown_start down to final_frac at total.
    if step >= total:
        return final_frac
    progress = (step - warmdown_start) / max(total - warmdown_start, 1)
    return 1.0 - progress * (1.0 - final_frac)


# ---------------------------------------------------------------------------
# Optimizer — simple AdamW, weight-decay on 2-D params, decay-exempt on 1-D.
# Mirrors `optim.build_param_groups` but inline so we don't pull the whole
# TrainConfig machinery into the SFT script.
# ---------------------------------------------------------------------------


def build_sft_optimizer(
    model: torch.nn.Module, *, peak_lr: float, weight_decay: float,
    betas: tuple[float, float], eps: float,
) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay, "base_lr": peak_lr},
        {"params": no_decay, "weight_decay": 0.0, "base_lr": peak_lr},
    ]
    try:
        return torch.optim.AdamW(groups, lr=peak_lr, betas=betas, eps=eps, fused=True)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, lr=peak_lr, betas=betas, eps=eps)


# ---------------------------------------------------------------------------
# Evaluation: masked-token val loss on a held-out SFT split
# ---------------------------------------------------------------------------


@torch.no_grad()
def sft_evaluate(model: torch.nn.Module, val_loader: SFTDataLoader, device: str,
                 amp_dtype: torch.dtype, max_batches: int) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(max_batches):
        x, y = val_loader.next_batch(device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
            logits, _ = model(x, y)  # model forward computes loss internally when targets given
        # Recompute loss with ignore_index=-1 to be explicit (model.forward uses same convention).
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-1, reduction="mean",
        )
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrain-ckpt", required=True, type=str)
    p.add_argument("--run-dir", required=True, type=str)
    p.add_argument("--sft-tokens", type=int, default=500_000_000)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--block-size", type=int, default=1024)
    # SFT LR 10× smaller than pretrain peak (6e-4) per nanochat + general SFT lore.
    p.add_argument("--peak-lr", type=float, default=3e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--warmdown-ratio", type=float, default=0.5)
    p.add_argument("--final-lr-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)  # nanochat SFT uses wd=0
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--keep-last-ckpt", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--compile-mode", type=str, default="default")
    p.add_argument("--buffer-size", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tune_pytorch_globals()
    seed_everything(args.seed)
    device = device_str()

    # --- Load pretrained ----------------------------------------------------
    pretrain_path = Path(args.pretrain_ckpt)
    print(f"loading pretrained checkpoint: {pretrain_path}")
    ck = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    cfg = load_gpt_config_from_ckpt(ck)
    # Normalise attention backend for whatever device we're on.
    if device == "cuda":
        select_sdpa_backend_globally(cfg.attention_backend)
    model = GPT(cfg).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict: missing={missing[:5]} unexpected={unexpected[:5]}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params | arch: attention_type={cfg.attention_type} "
          f"weight_tied={cfg.weight_tied} n_kv_head={cfg.n_kv_head} mlp={cfg.mlp_type}")

    # --- Data ---------------------------------------------------------------
    print("loading SmolTalk (first time takes ~1 min + ~500 MB cache)...")
    train_task = SmolTalk(split="train")
    val_task = SmolTalk(split="test")
    print(f"SmolTalk: train={len(train_task):,} val={len(val_task):,} conversations")
    train_loader = SFTDataLoader(
        train_task, block_size=args.block_size, batch_size=args.micro_batch,
        buffer_size=args.buffer_size, seed=args.seed,
    )
    val_loader = SFTDataLoader(
        val_task, block_size=args.block_size, batch_size=args.micro_batch,
        buffer_size=args.buffer_size, seed=0,
    )

    # --- Schedule math ------------------------------------------------------
    tokens_per_step = args.micro_batch * args.grad_accum * args.block_size
    total_steps = max(1, args.sft_tokens // tokens_per_step)
    warmup_steps = int(args.warmup_ratio * total_steps)
    warmdown_start = max(warmup_steps + 1, int((1 - args.warmdown_ratio) * total_steps))
    print(
        f"tokens/step={tokens_per_step:,} total_steps={total_steps:,} "
        f"(warmup={warmup_steps}, warmdown_start={warmdown_start})"
    )

    # --- Optimizer ---------------------------------------------------------
    optimizer = build_sft_optimizer(
        model, peak_lr=args.peak_lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), eps=args.eps,
    )
    print(f"optimizer: AdamW | param groups: {len(optimizer.param_groups)}")

    # --- Compile -----------------------------------------------------------
    if args.compile and device == "cuda":
        mode = args.compile_mode
        model = torch.compile(model, mode=mode) if mode != "default" else torch.compile(model)
        print(f"torch.compile mode: {mode}")

    # --- Run dir + logging --------------------------------------------------
    run_dir = ensure_dir(args.run_dir)
    (run_dir / "config.json").write_text(json.dumps({
        "sft": vars(args),
        "model": cfg.__dict__,
        "schedule": {"total_steps": total_steps, "warmup_steps": warmup_steps,
                     "warmdown_start": warmdown_start},
    }, indent=2, default=str))
    write_env_fingerprint(run_dir / "env.json")
    log = JSONLLogger(run_dir / "metrics.jsonl")
    amp_dtype = autocast_dtype()
    print(f"autocast: {amp_dtype} | device: {device}")
    print(f"env: {env_fingerprint()}")

    # --- Train loop ---------------------------------------------------------
    best_val = float("inf")
    t_start = time.monotonic()
    t_last_log = t_start

    for step in range(total_steps):
        frac = sft_lr_frac(step, total=total_steps, warmup=warmup_steps,
                           warmdown_start=warmdown_start, final_frac=args.final_lr_frac)
        for g in optimizer.param_groups:
            g["lr"] = g.get("base_lr", args.peak_lr) * frac
        lr = args.peak_lr * frac

        optimizer.zero_grad(set_to_none=True)
        loss_accum_t = torch.zeros((), device=device)
        for _ in range(args.grad_accum):
            x, y = train_loader.next_batch(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            loss.backward()
            loss_accum_t += loss.detach()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        loss_accum = loss_accum_t.item()

        if (step + 1) % args.log_every == 0 or step == 0:
            now = time.monotonic()
            dt = now - t_last_log
            t_last_log = now
            tok_per_s = (args.log_every * tokens_per_step / dt) if dt > 0 else 0.0
            log.log({
                "event": "train_step", "step": step + 1, "loss": loss_accum, "lr": lr,
                "tok_per_s": tok_per_s, "elapsed_s": now - t_start,
            })
            print(
                f"step {step+1:>5}/{total_steps} | loss {loss_accum:.4f} | lr {lr:.2e} "
                f"| {tok_per_s/1e3:.1f}k tok/s | {human_time(now - t_start)}"
            )

        if (step + 1) % args.eval_every == 0 or (step + 1) == total_steps:
            val_loss = sft_evaluate(model, val_loader, device, amp_dtype, args.eval_batches)
            log.log({"event": "eval", "step": step + 1, "val_loss": val_loss})
            print(f"  [eval] step {step+1} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    run_dir / "best_val.pt",
                    model=getattr(model, "_orig_mod", model),
                    optimizer=optimizer,
                    step=step + 1,
                    config=cfg,
                    extra={"val_loss": val_loss},
                )

        if (step + 1) % args.ckpt_every == 0:
            save_checkpoint(
                run_dir / f"step_{step+1}.pt",
                model=getattr(model, "_orig_mod", model),
                optimizer=optimizer,
                step=step + 1,
                config=cfg,
            )
            rotate_checkpoints(run_dir, args.keep_last_ckpt)

    log.log({"event": "done", "best_val": best_val, "elapsed_s": time.monotonic() - t_start})
    log.close()
    print(f"done. best val loss: {best_val:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
