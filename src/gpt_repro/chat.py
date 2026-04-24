"""Text-marker chat template for SFT on tiktoken-GPT-2 tokenizer.

Our pretraining tokenizer is plain tiktoken `gpt2` — no reserved special
tokens for chat roles. Rather than extend the vocab (which would break
`tests/test_hf_weight_load.py` and complicate future HF interop), we use
plain ASCII markers that the BPE encodes as a handful of tokens each:

    <|endoftext|>  (token 50256, pretrain document separator — reused as BOS)
    \\n<user>\\n                ... \\n</user>\\n
    \\n<assistant>\\n           ... \\n</assistant>\\n

The markers aren't single tokens — `<user>` BPE-encodes to 5 tokens under the
gpt-2 vocab (`<`, `user`, `>`) — but that cost is < 10 tokens per conversation
and SFT learns the pattern in the first few hundred steps.

### Loss-mask convention

`render_conversation(messages)` returns `(ids, mask)` of identical length:

    mask[i] = 1 if ids[i] is a token the assistant must learn to emit.
    mask[i] = 0 for user turns, markers, and the BOS/EOT.

Assistant content is loss-weighted, AND the closing `</assistant>` marker is
loss-weighted too — the model needs to learn when to stop a turn. The opening
`<assistant>` marker is mask=0 (it's an unambiguous trigger that precedes
generation; not something the model predicts).

Downstream the `sft_data.py` loader shifts the mask by 1 to align with
next-token targets and zeroes out non-assistant positions via `-1` (the
`ignore_index` for `F.cross_entropy`).
"""

from __future__ import annotations

import copy

from gpt_repro.tokenizer import EOT_ID, get_encoding

# Text markers. All include explicit newlines so consecutive messages don't
# glue together at the BPE level.
USER_OPEN = "\n<user>\n"
USER_CLOSE = "\n</user>\n"
ASSISTANT_OPEN = "\n<assistant>\n"
ASSISTANT_CLOSE = "\n</assistant>\n"


def render_conversation(messages: list[dict]) -> tuple[list[int], list[int]]:
    """Tokenize a chat conversation into (ids, loss_mask).

    `messages` is a list of `{"role": "system"|"user"|"assistant", "content": str}`
    dicts with alternating user/assistant roles (system message, if present,
    is merged into the first user turn — same convention as nanochat).

    Returns two lists of equal length. `mask[i] = 1` iff `ids[i]` is an
    assistant-emitted token (including the closing `</assistant>` marker).
    """
    enc = get_encoding()
    ids: list[int] = []
    mask: list[int] = []

    # Merge optional leading system message into first user turn.
    msgs = list(messages)
    if msgs and msgs[0].get("role") == "system":
        assert len(msgs) >= 2 and msgs[1]["role"] == "user", (
            "System message must be followed by a user message"
        )
        msgs = copy.deepcopy(msgs)
        sys_content = msgs[0]["content"]
        msgs[1]["content"] = sys_content + "\n\n" + msgs[1]["content"]
        msgs = msgs[1:]

    assert len(msgs) >= 2, f"Conversation must have ≥ 2 messages after system merge: {len(msgs)}"

    def add(tokens: list[int], mask_val: int) -> None:
        ids.extend(tokens)
        mask.extend([mask_val] * len(tokens))

    # Document-level BOS (reused EOT token).
    add([EOT_ID], 0)

    for i, msg in enumerate(msgs):
        role = msg["role"]
        content = msg["content"]
        expected = "user" if i % 2 == 0 else "assistant"
        assert role == expected, f"Message {i} role={role!r} expected {expected!r}"
        assert isinstance(content, str), f"Message {i} content must be str, got {type(content)}"

        if role == "user":
            add(enc.encode_ordinary(USER_OPEN), 0)
            add(enc.encode_ordinary(content), 0)
            add(enc.encode_ordinary(USER_CLOSE), 0)
        else:  # assistant
            add(enc.encode_ordinary(ASSISTANT_OPEN), 0)     # trigger, not predicted
            add(enc.encode_ordinary(content), 1)            # the stuff we want to learn
            add(enc.encode_ordinary(ASSISTANT_CLOSE), 1)    # end-of-turn signal

    return ids, mask


def marker_token_counts() -> dict[str, int]:
    """Debug helper: how many tokens each marker consumes under the BPE."""
    enc = get_encoding()
    return {
        "USER_OPEN": len(enc.encode_ordinary(USER_OPEN)),
        "USER_CLOSE": len(enc.encode_ordinary(USER_CLOSE)),
        "ASSISTANT_OPEN": len(enc.encode_ordinary(ASSISTANT_OPEN)),
        "ASSISTANT_CLOSE": len(enc.encode_ordinary(ASSISTANT_CLOSE)),
    }
