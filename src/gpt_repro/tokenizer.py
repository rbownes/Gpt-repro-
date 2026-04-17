"""Thin wrapper over tiktoken's GPT-2 BPE.

Same vocab as Radford et al. 2019 (50,257 tokens, EOT = 50256).
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


EOT_ID: int = 50256
VOCAB_SIZE: int = 50257


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def encode(text: str, *, add_eot: bool = False) -> list[int]:
    enc = get_encoding()
    ids = enc.encode_ordinary(text)
    if add_eot:
        ids.append(EOT_ID)
    return ids


def decode(ids: list[int]) -> str:
    return get_encoding().decode(ids)


def encode_batch(texts: list[str]) -> list[list[int]]:
    return get_encoding().encode_ordinary_batch(texts)
