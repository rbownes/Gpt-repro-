"""Single-GPU BF16 training loop for faithful GPT-2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gpt_repro.data import DataConfig, ShardLoader
from gpt_repro.model import GPT, GPTConfig, select_sdpa_backend_globally
from gpt_repro.optim import build_dual_optimizer, build_optimizer, lr_at_step, set_lr
from gpt_repro.utils import (
    JSONLLogger,
    autocast_dtype,
    device_str,
    ensure_dir,
    env_fingerprint,
    human_time,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
    seed_everything,
    tune_pytorch_globals,
    write_env_fingerprint,
)


@dataclass
class TrainConfig:
    model: GPTConfig = field(default_factory=GPTConfig)

    # Data
    data_dir: str = "data/fineweb_edu_10B"

    # Batching (effective batch = micro_batch * grad_accum * block_size tokens)
    micro_batch: int = 16
    grad_accum: int = 32
    block_size: int = 1024  # kept in sync with model.block_size

    # Schedule
    total_steps: int = 19_073       # ~10B tokens at 0.5M tok/step
    warmup_steps: int = 715
    peak_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Training loop
    eval_every: int = 500
    eval_batches: int = 50
    log_every: int = 10
    ckpt_every: int = 2000
    keep_last_ckpt: int = 2

    # Optimizer choice
    optimizer_type: str = "adamw"       # "adamw" | "muon+adamw"
    muon_peak_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5

    # Run metadata
    run_dir: str = "runs/baseline"
    seed: int = 0
    compile: bool = True
    resume_from: str | None = None


def _init_run(cfg: TrainConfig) -> tuple[Path, JSONLLogger]:
    run_dir = ensure_dir(cfg.run_dir)
    (run_dir / "config.json").write_text(__import__("json").dumps(
        {"train": {k: v for k, v in cfg.__dict__.items() if k != "model"},
         "model": cfg.model.__dict__}, indent=2, default=str))
    write_env_fingerprint(run_dir / "env.json")
    return run_dir, JSONLLogger(run_dir / "metrics.jsonl")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: ShardLoader,
    device: str,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for x, y in val_loader.iter_val(device, max_batches=max_batches):
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(cfg: TrainConfig) -> None:
    tune_pytorch_globals()
    seed_everything(cfg.seed)
    device = device_str()
    if device != "cuda":
        print("[warn] CUDA unavailable; falling back to CPU (debug only).")
    else:
        select_sdpa_backend_globally(cfg.model.attention_backend)

    # Model
    model_cfg = cfg.model
    if model_cfg.block_size != cfg.block_size:
        raise ValueError(f"model.block_size={model_cfg.block_size} != train.block_size={cfg.block_size}")
    model = GPT(model_cfg).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    train_loader = ShardLoader(DataConfig(
        data_dir=cfg.data_dir, block_size=cfg.block_size,
        batch_size=cfg.micro_batch, split="train",
    ))
    val_loader = ShardLoader(DataConfig(
        data_dir=cfg.data_dir, block_size=cfg.block_size,
        batch_size=cfg.micro_batch, split="val",
    ))
    print(f"train tokens: {train_loader.total_tokens:,} | val tokens: {val_loader.total_tokens:,}")

    # Optimizer(s). A single AdamW covers the faithful and modern-block
    # baselines; Muon + AdamW splits hidden matmul weights off to Muon.
    if cfg.optimizer_type == "adamw":
        adamw = build_optimizer(
            model,
            lr=cfg.peak_lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )
        muon = None
        optimizers: list = [adamw]
        print(f"optimizer: AdamW, {sum(len(g['params']) for g in adamw.param_groups)} param groups")
    elif cfg.optimizer_type == "muon+adamw":
        muon, adamw = build_dual_optimizer(
            model,
            muon_lr=cfg.muon_peak_lr,
            muon_momentum=cfg.muon_momentum,
            muon_nesterov=cfg.muon_nesterov,
            muon_ns_steps=cfg.muon_ns_steps,
            adamw_lr=cfg.peak_lr,
            adamw_weight_decay=cfg.weight_decay,
            adamw_betas=(cfg.beta1, cfg.beta2),
            adamw_eps=cfg.eps,
        )
        optimizers = [muon, adamw]
        muon_n = sum(p.numel() for g in muon.param_groups for p in g["params"])
        adamw_n = sum(p.numel() for g in adamw.param_groups for p in g["params"])
        print(
            f"optimizer: Muon({muon_n/1e6:.1f}M params, peak_lr={cfg.muon_peak_lr}) "
            f"+ AdamW({adamw_n/1e6:.1f}M params, peak_lr={cfg.peak_lr})"
        )
    else:
        raise ValueError(f"unknown optimizer_type: {cfg.optimizer_type!r}")

    # Resume?
    start_step = 0
    if cfg.resume_from is not None:
        ck = load_checkpoint(cfg.resume_from, model=model, optimizer=adamw, map_location=device)
        if muon is not None and "muon" in ck:
            muon.load_state_dict(ck["muon"])
        start_step = int(ck.get("step", 0))
        print(f"resumed from {cfg.resume_from} @ step {start_step}")

    # Compile after loading weights (torch.compile wraps the module; HF
    # weight-load must target the raw module).
    if cfg.compile and device == "cuda":
        model = torch.compile(model)  # type: ignore[assignment]

    run_dir, log = _init_run(cfg)
    amp_dtype = autocast_dtype()
    print(f"autocast dtype: {amp_dtype} | SDPA backend: {cfg.model.attention_backend}")
    print(f"env: {env_fingerprint()}")

    # Training state
    rng = np.random.default_rng(cfg.seed)
    best_val = float("inf")
    tokens_per_step = cfg.micro_batch * cfg.grad_accum * cfg.block_size
    print(f"tokens / step: {tokens_per_step:,} | total tokens: {tokens_per_step * cfg.total_steps:,}")

    t_start = time.monotonic()
    t_last_log = t_start

    # Schedule driver: one scalar t that scales each optimizer's peak LR.
    def schedule_t(step: int) -> float:
        return lr_at_step(
            step,
            peak_lr=1.0,  # unit peak — scale per optimizer below
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps,
            min_lr_ratio=cfg.min_lr_ratio,
        )

    for step in range(start_step, cfg.total_steps):
        t = schedule_t(step)
        adamw_lr = cfg.peak_lr * t
        set_lr(adamw, adamw_lr)
        if muon is not None:
            set_lr(muon, cfg.muon_peak_lr * t)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum):
            x, y = train_loader.next_batch(rng, device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
                _, loss = model(x, y)
            loss = loss / cfg.grad_accum
            loss.backward()
            loss_accum += loss.item()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
        lr = adamw_lr  # for logging continuity

        if (step + 1) % cfg.log_every == 0 or step == start_step:
            now = time.monotonic()
            dt = now - t_last_log
            t_last_log = now
            tok_per_s = (cfg.log_every * tokens_per_step / dt) if dt > 0 else 0.0
            log.log({
                "event": "train_step",
                "step": step + 1,
                "loss": loss_accum,
                "lr": lr,
                "tok_per_s": tok_per_s,
                "elapsed_s": now - t_start,
            })
            print(
                f"step {step+1:>6}/{cfg.total_steps} | loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | {tok_per_s/1e3:.1f}k tok/s | {human_time(now - t_start)}"
            )

        if (step + 1) % cfg.eval_every == 0 or (step + 1) == cfg.total_steps:
            val_loss = evaluate(model, val_loader, device, amp_dtype, cfg.eval_batches)
            log.log({"event": "eval", "step": step + 1, "val_loss": val_loss})
            print(f"  [eval] step {step+1} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    Path(run_dir) / "best_val.pt",
                    model=getattr(model, "_orig_mod", model),
                    optimizer=adamw,
                    step=step + 1,
                    config=cfg.model,
                    extra={
                        "val_loss": val_loss,
                        **({"muon": muon.state_dict()} if muon is not None else {}),
                    },
                )

        if (step + 1) % cfg.ckpt_every == 0:
            save_checkpoint(
                Path(run_dir) / f"step_{step+1}.pt",
                model=getattr(model, "_orig_mod", model),
                optimizer=adamw,
                step=step + 1,
                config=cfg.model,
                extra=({"muon": muon.state_dict()} if muon is not None else None),
            )
            rotate_checkpoints(run_dir, cfg.keep_last_ckpt)

    log.log({"event": "done", "best_val": best_val, "elapsed_s": time.monotonic() - t_start})
    log.close()
    print(f"done. best val loss: {best_val:.4f}")


def train_from_dict(d: dict[str, Any]) -> None:
    model_cfg = GPTConfig(**d.pop("model", {}))
    train(TrainConfig(model=model_cfg, **d))
