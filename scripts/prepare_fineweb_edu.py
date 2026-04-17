"""Prepare FineWeb-Edu as uint16 token shards.

Streams `HuggingFaceFW/fineweb-edu` (sample-10BT by default) from the Hub,
tokenises with tiktoken gpt2 in a multiprocess pool, prepends EOT between
documents, and writes fixed-size uint16 shards to disk.

Shard layout matches karpathy/llm.c and nanoGPT-speedrun so drop-in tooling
works. First shard is reserved as the val split.

Usage:
    uv run python scripts/prepare_fineweb_edu.py \
        --out data/fineweb_edu_10B \
        --subset sample-10BT \
        --shard-size 100_000_000

~10 B tokens × 2 B = ~20 GB on disk.
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
    """Tokenise a single doc and prepend EOT. Runs in a worker process."""
    ids = [EOT, *_ENC.encode_ordinary(doc["text"])]  # type: ignore[name-defined]
    arr = np.asarray(ids, dtype=np.int32)
    assert (0 <= arr).all() and (arr < 2**16).all(), "token out of uint16 range"
    return arr.astype(np.uint16)


def _write_shard(out: Path, split: str, idx: int, buf: np.ndarray) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}_{idx:04d}.bin"
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/fineweb_edu_10B")
    p.add_argument("--subset", default="sample-10BT",
                   help="FineWeb-Edu subset: sample-10BT | sample-100BT | sample-350BT | CC-MAIN-YYYY-WW")
    p.add_argument("--shard-size", type=int, default=100_000_000,
                   help="tokens per shard; default 100 M")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    p.add_argument("--limit-docs", type=int, default=0, help="debug cap on document count")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset  # type: ignore

    print(f"streaming HuggingFaceFW/fineweb-edu/{args.subset} ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.subset,
        split="train",
        streaming=True,
    )
    if args.limit_docs:
        ds = ds.take(args.limit_docs)

    shard_size = args.shard_size
    buf = np.empty(shard_size, dtype=np.uint16)
    shard_count = 0
    pos = 0
    pbar = tqdm(unit=" tok", unit_scale=True, desc="tokens")

    with mp.Pool(args.workers, initializer=_init_tokenizer) as pool:
        for tokens in pool.imap(_tokenize, ds, chunksize=16):
            if pos + len(tokens) < shard_size:
                buf[pos : pos + len(tokens)] = tokens
                pos += len(tokens)
                pbar.update(len(tokens))
            else:
                remainder = shard_size - pos
                buf[pos:] = tokens[:remainder]
                split = "val" if shard_count == 0 else "train"
                _write_shard(out, split, shard_count, buf)
                pbar.update(remainder)
                shard_count += 1
                leftover = tokens[remainder:]
                buf[: leftover.shape[0]] = leftover
                pos = leftover.shape[0]

    # Flush the tail as the final train shard.
    if pos > 0:
        split = "val" if shard_count == 0 else "train"
        _write_shard(out, split, shard_count, buf[:pos])

    pbar.close()
    print(f"wrote {shard_count + 1} shards to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
