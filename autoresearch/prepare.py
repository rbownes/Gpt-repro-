"""Autoresearch data-preparation gate — FROZEN, agent must not edit.

Two jobs:

1. Verify that the FineWeb-Edu-10B shards exist at `data/fineweb_edu_10B/`
   (prepared once by the top-level `scripts/prepare_fineweb_edu.py`). If
   they're missing, print instructions and exit non-zero. No re-tokenisation.

2. Build and cache `data/fineweb_edu_10B/token_bytes.pt` — a length-50304
   int tensor (rounded up from 50257 for matmul alignment) whose entry `i`
   is the UTF-8 byte length of tiktoken GPT-2 token `i`. Special tokens
   (EOT = 50256, padding 50257..50303) are recorded as 0 bytes so they're
   masked out of the BPB sum in `evaluate_bpb`.

This script is the ground-truth eval pathway. DO NOT modify.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tiktoken
import torch


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fineweb_edu_10B"
TOKEN_BYTES_PATH = DATA_DIR / "token_bytes.pt"

# GPT-2 vocab + padding to a multiple of 128 for matmul alignment. The real
# vocab is 50257 (0..50256, with 50256 = EOT); anything at index ≥ 50257 is
# unreachable in the dataset. Byte length = 0 ⇒ masked out of BPB math.
REAL_VOCAB_SIZE = 50257
VOCAB_SIZE = 50304  # padded (50257 rounded up to 128-multiple)
EOT_ID = 50256


def _compute_token_bytes() -> torch.Tensor:
    enc = tiktoken.get_encoding("gpt2")
    bytes_per_token = torch.zeros(VOCAB_SIZE, dtype=torch.int32)
    # Token IDs 0..50255 are regular BPE tokens.
    for tok_id in range(REAL_VOCAB_SIZE - 1):
        # decode_single_token_bytes returns the raw bytes for this token.
        try:
            b = enc.decode_single_token_bytes(tok_id)
            bytes_per_token[tok_id] = len(b)
        except KeyError:
            bytes_per_token[tok_id] = 0
    # 50256 = EOT (special) → 0 bytes (masked)
    # 50257..50303 = padding (unreachable in data) → 0 bytes
    return bytes_per_token


def main() -> int:
    if not DATA_DIR.is_dir():
        print(
            f"[prepare] FATAL: {DATA_DIR} does not exist. Run\n"
            f"    uv run python scripts/prepare_fineweb_edu.py --out {DATA_DIR}\n"
            f"from the project root to tokenise the 10 B-token corpus."
        )
        return 1

    train_shards = sorted(DATA_DIR.glob("train_*.bin"))
    val_shards = sorted(DATA_DIR.glob("val_*.bin"))
    if not train_shards or not val_shards:
        print(
            f"[prepare] FATAL: no train_*.bin / val_*.bin shards in {DATA_DIR}.\n"
            f"Rerun scripts/prepare_fineweb_edu.py."
        )
        return 1
    total_bytes = sum(p.stat().st_size for p in train_shards + val_shards)
    print(f"[prepare] found {len(train_shards)} train + {len(val_shards)} val shards "
          f"({total_bytes / 1e9:.1f} GB on disk)")

    if TOKEN_BYTES_PATH.exists():
        tb = torch.load(TOKEN_BYTES_PATH, weights_only=True)
        print(f"[prepare] token_bytes.pt cache exists ({tuple(tb.shape)}, sum={int(tb.sum())})")
    else:
        print("[prepare] computing token_bytes.pt ...")
        tb = _compute_token_bytes()
        torch.save(tb, TOKEN_BYTES_PATH)
        print(f"[prepare] wrote {TOKEN_BYTES_PATH} (shape {tuple(tb.shape)}, sum={int(tb.sum())})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
