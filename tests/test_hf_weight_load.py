"""HF weight-load parity test — the baseline correctness gate.

Loads the released Hugging Face `gpt2` weights into our faithful module and
verifies that next-token logits match HF's `GPT2LMHeadModel` within a tight
tolerance. If this test passes, the architecture is correct; if it fails,
the model deviates from the published GPT-2.

This test is gated on network (first-time HF cache fill) and transformers
being installed. Mark it slow so `pytest -m 'not slow'` can skip it.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["gpt2"])
def test_hf_logits_match(model_name: str) -> None:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    from gpt_repro.model import GPT

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    prompts = [
        "The capital of France is",
        "Once upon a time, in a faraway land,",
        "def hello_world():\n    ",
    ]

    hf = GPT2LMHeadModel.from_pretrained(model_name).eval()
    ours = GPT.from_pretrained_gpt2(model_name).eval()

    # GPT-2 has no pad token; tokenise one prompt at a time and compare per-prompt.
    with torch.no_grad():
        for prompt in prompts:
            ids = tok(prompt, return_tensors="pt")["input_ids"]
            hf_logits = hf(ids).logits
            # targets=ids triggers the full-sequence logits path in our module
            ours_logits, _ = ours.forward(ids, ids)
            torch.testing.assert_close(
                ours_logits,
                hf_logits,
                atol=1e-3,
                rtol=1e-3,
                msg=f"logits differ from HF on prompt {prompt!r}",
            )
