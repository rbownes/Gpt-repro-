"""Prepare OpenWebText as uint16 token shards (same layout as FineWeb prep).

This exists for the faithful OWT reproduction path. Output shape is
identical to `prepare_fineweb_edu.py` so the data loader is unchanged.

Usage:
    uv run python scripts/prepare_openwebtext.py \
        --out data/openwebtext_9B \
        --shard-size 100_000_000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


EOT = 50256


def _init_tokenizer() -> None:
    import tiktoken  # noqa: F401

    global _ENC
    _ENC = __import__("tiktoken").get_encoding("gpt2")  # type: ignore[name-defined]


def _tokenize(doc: dict) -> np.ndarray:
    ids = [EOT, *_ENC.encode_ordinary(doc["text"])]  # type: ignore[name-defined]
    arr = np.asarray(ids, dtype=np.int32)
    assert (0 <= arr).all() and (arr < 2**16).all()
    return arr.astype(np.uint16)


def _write_shard(out: Path, split: str, idx: int, buf: np.ndarray) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{split}_{idx:04d}.bin", "wb") as f:
        f.write(buf.tobytes())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/openwebtext_9B")
    p.add_argument("--shard-size", type=int, default=100_000_000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    args = p.parse_args()

    from datasets import load_dataset  # type: ignore

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)

    out = Path(args.out)
    buf = np.empty(args.shard_size, dtype=np.uint16)
    pos = 0
    shard = 0
    pbar = tqdm(unit=" tok", unit_scale=True)

    with mp.Pool(args.workers, initializer=_init_tokenizer) as pool:
        for tokens in pool.imap(_tokenize, ds, chunksize=16):
            if pos + len(tokens) < args.shard_size:
                buf[pos : pos + len(tokens)] = tokens
                pos += len(tokens)
                pbar.update(len(tokens))
            else:
                remainder = args.shard_size - pos
                buf[pos:] = tokens[:remainder]
                _write_shard(out, "val" if shard == 0 else "train", shard, buf)
                pbar.update(remainder)
                shard += 1
                leftover = tokens[remainder:]
                buf[: leftover.shape[0]] = leftover
                pos = leftover.shape[0]

    if pos > 0:
        _write_shard(out, "val" if shard == 0 else "train", shard, buf[:pos])

    pbar.close()
    print(f"wrote {shard + 1} shards to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
