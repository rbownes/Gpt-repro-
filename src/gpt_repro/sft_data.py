"""Best-fit conversation packing for SFT training.

Each batch row is `block_size + 1` tokens long (the `+1` is for the shifted
target at the last position). Conversations are rendered via
`chat.render_conversation(...)` and packed greedily: at each step the largest
buffered conversation that fits into the remaining row space is placed. When
nothing fits, the remainder is padded with EOT and mask=0 — no tokens are
silently dropped.

The yielded `(x, y)` tensors match the pretraining `ShardLoader` contract:

    x: (batch, block_size) input tokens, dtype int64
    y: (batch, block_size) target tokens, dtype int64, with
       y[i, j] = -1 wherever the assistant mask was 0

`F.cross_entropy(ignore_index=-1)` then skips all non-assistant-content
positions automatically, so the existing training loop consumes SFT batches
unchanged.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from gpt_repro.chat import render_conversation
from gpt_repro.tasks import Task
from gpt_repro.tokenizer import EOT_ID


class SFTDataLoader:
    """Best-fit packing dataloader over a Task.

    Args:
        task: a `Task` that yields `{"messages": [...]}` dicts.
        block_size: sequence length per row; actual row holds `block_size + 1`
            tokens so that after shifting there are `block_size` (input, target)
            pairs. Matches the pretrain loader's contract.
        batch_size: number of rows per batch.
        buffer_size: how many conversations to keep in the best-fit pool.
        seed: cycles the dataset deterministically.
        render: override the default chat template (mostly for testing).
    """

    def __init__(
        self,
        task: Task,
        *,
        block_size: int,
        batch_size: int,
        buffer_size: int = 100,
        seed: int = 0,
        render: Callable[[list[dict]], tuple[list[int], list[int]]] = render_conversation,
    ) -> None:
        self.task = task
        self.block_size = block_size
        self.row_capacity = block_size + 1
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.render = render
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        # Visit ordering: pre-shuffled index map so restarts are deterministic.
        self._order = self._rng.permutation(len(task)).tolist()
        self._cursor = 0
        self._buffer: list[tuple[list[int], list[int]]] = []

    def _next_conversation_index(self) -> int:
        idx = self._order[self._cursor % len(self._order)]
        self._cursor += 1
        if self._cursor % len(self._order) == 0:
            # Reshuffle at each epoch boundary so long runs don't memorise order.
            self._rng.shuffle(self._order)
        return int(idx)

    def _refill_buffer(self) -> None:
        while len(self._buffer) < self.buffer_size:
            conv = self.task[self._next_conversation_index()]
            ids, mask = self.render(conv["messages"])
            # Truncate conversations longer than the row capacity. (An
            # alternative is to drop them entirely — we choose truncation so
            # even pathological long rows contribute partial signal.)
            if len(ids) > self.row_capacity:
                ids = ids[: self.row_capacity]
                mask = mask[: self.row_capacity]
            self._buffer.append((ids, mask))

    def _build_row(self) -> tuple[list[int], list[int]]:
        """Best-fit-pack a single row of length `row_capacity`."""
        row: list[int] = []
        mask_row: list[int] = []
        while len(row) < self.row_capacity:
            if len(self._buffer) < self.buffer_size:
                self._refill_buffer()
            remaining = self.row_capacity - len(row)
            best_i = -1
            best_len = 0
            for i, (ids, _m) in enumerate(self._buffer):
                L = len(ids)
                if L <= remaining and L > best_len:
                    best_i, best_len = i, L
            if best_i >= 0:
                ids, mask = self._buffer.pop(best_i)
                row.extend(ids)
                mask_row.extend(mask)
            else:
                # Nothing fits in the remainder — pad with EOT, mask=0.
                row.extend([EOT_ID] * remaining)
                mask_row.extend([0] * remaining)
                break
        return row[: self.row_capacity], mask_row[: self.row_capacity]

    def next_batch(self, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble a batch of `(x, y)` tensors shaped `(batch, block_size)`.

        `y[i, j] = -1` wherever the assistant-mask was 0, so the training
        loop's `F.cross_entropy(ignore_index=-1)` ignores those positions.
        """
        rows, masks = [], []
        for _ in range(self.batch_size):
            r, m = self._build_row()
            rows.append(r)
            masks.append(m)
        batch = np.asarray(rows, dtype=np.int64)       # (B, T+1)
        mask_batch = np.asarray(masks, dtype=np.int64)  # (B, T+1)
        x = torch.from_numpy(batch[:, :-1]).to(device)       # (B, T)
        y_tokens = torch.from_numpy(batch[:, 1:]).to(device)  # (B, T)
        y_mask = torch.from_numpy(mask_batch[:, 1:]).to(device)  # mask at target positions
        y = torch.where(y_mask.bool(), y_tokens, torch.full_like(y_tokens, -1))
        return x, y

    def token_stats(self, n_batches: int = 1) -> dict:
        """Return coverage metrics over `n_batches` — debugging only."""
        total = 0
        masked = 0
        padded = 0
        for _ in range(n_batches):
            x, y = self.next_batch()
            total += y.numel()
            masked += (y >= 0).sum().item()
            # EOT tokens at the row tail that the mask skipped are "padding".
            # A rough proxy: count positions where x is EOT and y is -1.
            padded += ((x == EOT_ID) & (y < 0)).sum().item()
        return {
            "positions_total": total,
            "positions_with_loss": masked,
            "fraction_with_loss": masked / max(1, total),
            "approx_pad_positions": padded,
        }
