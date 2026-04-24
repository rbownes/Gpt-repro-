"""SFT harness tests: chat template + packing dataloader.

These tests don't need `datasets` / HF downloads — they build minimal synthetic
Task subclasses in-process and exercise the render/pack pipeline end-to-end.
"""

from __future__ import annotations

import pytest
import torch

from gpt_repro.chat import (
    ASSISTANT_CLOSE,
    ASSISTANT_OPEN,
    USER_CLOSE,
    USER_OPEN,
    marker_token_counts,
    render_conversation,
)
from gpt_repro.sft_data import SFTDataLoader
from gpt_repro.tasks import Task, TaskMixture
from gpt_repro.tokenizer import EOT_ID, decode


# ---------------------------------------------------------------------------
# Synthetic Task subclass — avoids any HF dataset download
# ---------------------------------------------------------------------------


class _InMemoryTask(Task):
    def __init__(self, conversations: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self._convs = conversations

    def num_examples(self) -> int:
        return len(self._convs)

    def get_example(self, index: int) -> dict:
        return self._convs[index]


def _canonical_conversation() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I am a test assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4."},
        ]
    }


def _conversation_with_system() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello!"},
        ]
    }


# ---------------------------------------------------------------------------
# chat.render_conversation
# ---------------------------------------------------------------------------


def test_render_conversation_lengths_match() -> None:
    ids, mask = render_conversation(_canonical_conversation()["messages"])
    assert len(ids) == len(mask) > 0


def test_render_conversation_starts_with_eot() -> None:
    ids, _ = render_conversation(_canonical_conversation()["messages"])
    assert ids[0] == EOT_ID


def test_render_conversation_mask_is_binary() -> None:
    _, mask = render_conversation(_canonical_conversation()["messages"])
    assert set(mask).issubset({0, 1})


def test_render_conversation_mask_on_assistant_only() -> None:
    """mask=1 positions correspond to assistant turns (content + </assistant>)."""
    ids, mask = render_conversation(_canonical_conversation()["messages"])
    # Decode the masked-positions-only substring. It should be the concatenation
    # of the two assistant turns (content + closing marker).
    masked_text = decode([i for i, m in zip(ids, mask) if m == 1])
    # Expected content: "I am a test assistant." + ASSISTANT_CLOSE + "4." + ASSISTANT_CLOSE.
    assert "I am a test assistant." in masked_text
    assert "4." in masked_text
    assert "</assistant>" in masked_text
    # And no user content.
    assert "Hello, who are you?" not in masked_text
    assert "What is 2 + 2?" not in masked_text


def test_render_conversation_mask_zero_on_markers_and_user() -> None:
    ids, mask = render_conversation(_canonical_conversation()["messages"])
    # Any token that is part of the opening `<user>` or `<assistant>` marker
    # should have mask=0. The easiest check: the assistant-open marker's first
    # token (the newline) should appear somewhere with mask=0.
    unmasked_text = decode([i for i, m in zip(ids, mask) if m == 0])
    assert "<user>" in unmasked_text
    assert "</user>" in unmasked_text
    assert "<assistant>" in unmasked_text  # opener (closer has mask=1 by design)
    # User content unmasked.
    assert "Hello, who are you?" in unmasked_text


def test_render_conversation_system_merges_into_first_user() -> None:
    """System message prefixes the first user turn and disappears as a role."""
    ids, mask = render_conversation(_conversation_with_system()["messages"])
    # The user content should include the system prompt inline.
    unmasked_text = decode([i for i, m in zip(ids, mask) if m == 0])
    assert "helpful assistant" in unmasked_text
    assert "Hi." in unmasked_text


def test_render_conversation_rejects_wrong_role_order() -> None:
    """Two user turns in a row without an assistant between → fail loudly."""
    with pytest.raises(AssertionError):
        render_conversation([
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "Hello"},
        ])


def test_marker_token_counts_small() -> None:
    """Sanity: markers BPE to reasonable token counts (< 10 each)."""
    counts = marker_token_counts()
    for name, n in counts.items():
        assert 1 <= n <= 10, f"{name} encodes to {n} tokens — surprising"


# ---------------------------------------------------------------------------
# tasks.Task / TaskMixture
# ---------------------------------------------------------------------------


def test_inmemory_task_len_and_get() -> None:
    convs = [_canonical_conversation(), _conversation_with_system()]
    t = _InMemoryTask(convs)
    assert len(t) == 2
    assert t[0] == convs[0]
    assert t[1] == convs[1]


def test_task_slice_view() -> None:
    convs = [{"messages": [{"role": "user", "content": str(i)},
                           {"role": "assistant", "content": "ok"}]} for i in range(10)]
    t = _InMemoryTask(convs, start=2, stop=8, step=2)
    # Logical indices: 2, 4, 6  → 3 items
    assert len(t) == 3
    assert t[0]["messages"][0]["content"] == "2"
    assert t[2]["messages"][0]["content"] == "6"


def test_task_mixture_deterministic() -> None:
    t1 = _InMemoryTask([{"messages": [{"role": "user", "content": "a"},
                                      {"role": "assistant", "content": "A"}]}])
    t2 = _InMemoryTask([{"messages": [{"role": "user", "content": "b"},
                                      {"role": "assistant", "content": "B"}]}])
    m1 = TaskMixture([t1, t2])
    m2 = TaskMixture([t1, t2])
    # Two TaskMixtures constructed with the same tasks produce the same index map.
    assert m1.index_map == m2.index_map
    assert len(m1) == 2


# ---------------------------------------------------------------------------
# sft_data.SFTDataLoader
# ---------------------------------------------------------------------------


def _build_loader(block_size: int = 128, batch_size: int = 2, buffer_size: int = 8) -> SFTDataLoader:
    convs = [_canonical_conversation(), _conversation_with_system()] * 20
    task = _InMemoryTask(convs)
    return SFTDataLoader(
        task, block_size=block_size, batch_size=batch_size, buffer_size=buffer_size, seed=0,
    )


def test_sft_dataloader_batch_shape() -> None:
    loader = _build_loader(block_size=128, batch_size=3)
    x, y = loader.next_batch()
    assert x.shape == (3, 128)
    assert y.shape == (3, 128)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_sft_dataloader_mask_applied_as_ignore_index() -> None:
    """At least some y values should be -1 (masked), at least some ≥ 0 (kept)."""
    loader = _build_loader(block_size=128, batch_size=4)
    _, y = loader.next_batch()
    assert (y == -1).any(), "expected some masked (ignored) target positions"
    assert (y >= 0).any(), "expected some unmasked (loss-bearing) target positions"


def test_sft_dataloader_targets_are_shifted_inputs_where_unmasked() -> None:
    """Where the mask is 1, y[i, j] == ids[i, j+1] — i.e. the next-token target."""
    loader = _build_loader(block_size=64, batch_size=2)
    x, y = loader.next_batch()
    # Build back: the target at position j is the input at position j+1 for
    # unmasked positions. We can reconstruct ids[:-1]=x and check the overlap.
    for b in range(x.shape[0]):
        for j in range(x.shape[1] - 1):
            if y[b, j] >= 0:
                assert y[b, j] == x[b, j + 1], (
                    f"row {b} col {j}: y={int(y[b, j])} vs x[{j+1}]={int(x[b, j + 1])}"
                )


def test_sft_dataloader_does_not_drop_conversations() -> None:
    """Even a tiny row_capacity keeps short conversations — they get truncated, not dropped."""
    loader = _build_loader(block_size=32, batch_size=1, buffer_size=4)
    x, y = loader.next_batch()
    # At least *some* position in the row has mask=1 (i.e., ≥ 0 in y), meaning
    # an assistant token survived the packing.
    assert (y >= 0).any()


def test_sft_dataloader_token_stats_report() -> None:
    loader = _build_loader(block_size=256, batch_size=4, buffer_size=16)
    stats = loader.token_stats(n_batches=2)
    assert stats["positions_total"] == 2 * 4 * 256
    assert 0 < stats["positions_with_loss"] < stats["positions_total"]
    assert 0 <= stats["fraction_with_loss"] <= 1
