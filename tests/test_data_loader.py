"""Data loader tests using a synthetic tiny shard."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gpt_repro.data import DataConfig, ShardLoader


def _make_shards(tmp: Path, train_tokens: int = 10_000, val_tokens: int = 2_000) -> Path:
    rng = np.random.default_rng(0)
    (tmp / "val_0000.bin").write_bytes(
        rng.integers(0, 50_257, size=val_tokens, dtype=np.uint16).tobytes()
    )
    (tmp / "train_0001.bin").write_bytes(
        rng.integers(0, 50_257, size=train_tokens, dtype=np.uint16).tobytes()
    )
    return tmp


def test_train_loader_shapes(tmp_path: Path) -> None:
    _make_shards(tmp_path)
    loader = ShardLoader(DataConfig(data_dir=str(tmp_path), block_size=128, batch_size=4, split="train"))
    rng = np.random.default_rng(42)
    x, y = loader.next_batch(rng, "cpu")
    assert x.shape == (4, 128)
    assert y.shape == (4, 128)
    assert (x < 50_257).all() and (y < 50_257).all()
    # y is x shifted by one position
    assert (y[:, :-1] == x[:, 1:]).all()


def test_val_loader_is_deterministic(tmp_path: Path) -> None:
    _make_shards(tmp_path)
    loader = ShardLoader(DataConfig(data_dir=str(tmp_path), block_size=128, batch_size=4, split="val"))
    a = list(loader.iter_val("cpu", max_batches=3))
    b = list(loader.iter_val("cpu", max_batches=3))
    assert len(a) == len(b) == 3
    for (xa, ya), (xb, yb) in zip(a, b, strict=True):
        assert (xa == xb).all() and (ya == yb).all()
