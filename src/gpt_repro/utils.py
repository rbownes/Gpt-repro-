"""Shared utilities: seeding, logging, checkpointing, env fingerprint."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_sha(cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def env_fingerprint() -> dict[str, Any]:
    fp: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        fp["gpu_name"] = torch.cuda.get_device_name(0)
        fp["gpu_capability"] = torch.cuda.get_device_capability(0)
        fp["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    fp["git_sha"] = git_sha()
    return fp


class JSONLLogger:
    """Append-only JSONL metrics writer.

    One file per run; each line is a flat JSON record. Keeps its own file
    handle open for speed and flushes on every write so a kill -9 doesn't
    eat the last N steps of metrics.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)  # line-buffered

    def log(self, record: dict[str, Any]) -> None:
        record = {"ts": time.time(), **record}
        self._fh.write(json.dumps(record, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": asdict(config) if is_dataclass(config) else config,
        "env": env_fingerprint(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def rotate_checkpoints(dir_: str | Path, keep_last: int) -> None:
    d = Path(dir_)
    ckpts = sorted(d.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for p in ckpts[:-keep_last]:
        try:
            p.unlink()
        except OSError:
            pass


def format_si(x: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(x) < 1000:
            return f"{x:.2f}{unit}"
        x /= 1000
    return f"{x:.2f}P"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast_dtype() -> torch.dtype:
    # Prefer BF16 on any Ampere+ or Blackwell; fall back to FP32 on CPU.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def ensure_gitignore(entries: list[str], gitignore_path: str = ".gitignore") -> None:
    p = Path(gitignore_path)
    existing = p.read_text().splitlines() if p.exists() else []
    for e in entries:
        if e not in existing:
            existing.append(e)
    p.write_text("\n".join(existing) + "\n")


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


# Convenience accessor for PyTorch's env-sensitive knobs.
def tune_pytorch_globals() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# --- Misc ----------------------------------------------------------------
def human_time(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def write_env_fingerprint(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(env_fingerprint(), indent=2, default=str))


def is_tty() -> bool:
    return os.isatty(1)
