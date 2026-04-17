"""Sharded memmap data loader matching the llm.c / nanoGPT-speedrun layout.

Each shard is a flat uint16 `.bin` file of GPT-2 token ids, with EOT (50256)
inserted between source documents at prep time. We never slice across shards
within one sequence: each example is a contiguous window from a single shard.

Train loader: deterministic per-step seed -> random shard + random offset.
Val loader:   deterministic walk over the first shard from offset 0.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DataConfig:
    data_dir: str
    block_size: int = 1024
    batch_size: int = 16
    split: str = "train"  # 'train' or 'val'


def _shard_paths(data_dir: str, split: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
    if not paths:
        raise FileNotFoundError(
            f"No {split} shards at {data_dir} (expected {split}_NNNN.bin). "
            "Run `uv run python scripts/prepare_fineweb_edu.py` first."
        )
    return paths


class ShardLoader:
    """Random-access loader over uint16 memmap shards."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.shards = _shard_paths(cfg.data_dir, cfg.split)
        self._memmaps: list[np.memmap] = []
        for p in self.shards:
            mm = np.memmap(p, dtype=np.uint16, mode="r")
            self._memmaps.append(mm)
        self.total_tokens = sum(m.shape[0] for m in self._memmaps)

    def __len__(self) -> int:
        return self.total_tokens

    def _sample_window(self, rng: np.random.Generator) -> np.ndarray:
        shard = self._memmaps[rng.integers(len(self._memmaps))]
        max_start = shard.shape[0] - self.cfg.block_size - 1
        start = int(rng.integers(0, max_start))
        return np.asarray(shard[start : start + self.cfg.block_size + 1], dtype=np.int64)

    def next_batch(self, rng: np.random.Generator, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Random batch for training. Uses the passed `rng` for reproducibility."""
        buf = np.stack([self._sample_window(rng) for _ in range(self.cfg.batch_size)])
        x = torch.from_numpy(buf[:, :-1]).to(device, non_blocking=True)
        y = torch.from_numpy(buf[:, 1:]).to(device, non_blocking=True)
        return x, y

    def iter_val(self, device: str | torch.device, max_batches: int | None = None):
        """Deterministic, contiguous walk over the val shard(s) for repeatable eval."""
        bs = self.cfg.batch_size
        ctx = self.cfg.block_size
        shard = self._memmaps[0]
        stride = ctx  # non-overlapping windows
        pos = 0
        n = 0
        while pos + bs * stride + 1 <= shard.shape[0]:
            windows = np.stack(
                [
                    np.asarray(shard[pos + i * stride : pos + i * stride + ctx + 1], dtype=np.int64)
                    for i in range(bs)
                ]
            )
            pos += bs * stride
            x = torch.from_numpy(windows[:, :-1]).to(device, non_blocking=True)
            y = torch.from_numpy(windows[:, 1:]).to(device, non_blocking=True)
            yield x, y
            n += 1
            if max_batches is not None and n >= max_batches:
                return
