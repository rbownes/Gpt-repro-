"""Task abstraction for SFT / eval datasets, ported from nanochat.

A `Task` is a conversation dataset with a uniform `{"messages": [...]}`
interface. Subclasses wrap HuggingFace `datasets` or local files. The
`TaskMixture` combiner lets SFT runs train on a deterministic shuffle of
several datasets at once (e.g. SmolTalk + GSM8K).

Usage:
    from gpt_repro.tasks import SmolTalk, TaskMixture
    train = SmolTalk(split="train")
    for i in range(3):
        print(train[i])   # {"messages": [{"role": "user", ...}, ...]}
"""

from __future__ import annotations

import random
from typing import Any


class Task:
    """Base class: a lazy view over a conversation dataset.

    Subclasses override `num_examples()` and `get_example(index)`. The
    logical slice `[start:stop:step]` is applied on top of the underlying
    dataset at `__getitem__` time.
    """

    def __init__(self, start: int = 0, stop: int | None = None, step: int = 1) -> None:
        assert start >= 0 and step >= 1
        assert stop is None or stop >= start
        self.start = start
        self.stop = stop
        self.step = step

    def num_examples(self) -> int:
        raise NotImplementedError

    def get_example(self, index: int) -> dict[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        span = stop - start
        return (span + self.step - 1) // self.step  # ceil div

    def __getitem__(self, index: int) -> dict[str, Any]:
        assert isinstance(index, int)
        physical_index = self.start + index * self.step
        return self.get_example(physical_index)


class TaskMixture(Task):
    """Combine multiple Tasks into a deterministically-shuffled stream.

    Trick: to oversample a task, pass it in multiple times — e.g.
    `TaskMixture([smoltalk, gsm8k, gsm8k, gsm8k])` runs GSM8K at 3× weight.
    """

    def __init__(self, tasks: list[Task], **kwargs) -> None:
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(t) for t in tasks]
        self.num_conversations = sum(self.lengths)
        # Build (task_idx, local_idx) pairs and shuffle deterministically.
        self.index_map: list[tuple[int, int]] = []
        for ti, n in enumerate(self.lengths):
            self.index_map.extend((ti, li) for li in range(n))
        rng = random.Random(42)
        rng.shuffle(self.index_map)

    def num_examples(self) -> int:
        return self.num_conversations

    def get_example(self, index: int) -> dict[str, Any]:
        ti, li = self.index_map[index]
        return self.tasks[ti][li]


# ---------------------------------------------------------------------------
# Concrete tasks
# ---------------------------------------------------------------------------


class SmolTalk(Task):
    """HuggingFaceTB/smol-smoltalk — lightweight general chat data.

    ~460 k train rows, ~24 k test rows. Each row is a `messages` list with
    alternating user/assistant turns (optional leading system message).
    """

    def __init__(self, split: str = "train", **kwargs) -> None:
        super().__init__(**kwargs)
        from datasets import load_dataset
        assert split in {"train", "test"}
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        messages = row["messages"]
        # Light validation — bail on footguns early.
        assert len(messages) >= 1
        first = messages[0]
        rest = messages[1:] if first["role"] == "system" else messages
        assert len(rest) >= 2, "SmolTalk row needs ≥2 user/assistant turns"
        for i, m in enumerate(rest):
            expected = "user" if i % 2 == 0 else "assistant"
            assert m["role"] == expected, f"row {index} msg {i}: {m['role']} != {expected}"
            assert isinstance(m["content"], str)
        return {"messages": messages}
