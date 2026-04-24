"""Cross-checkpoint load smoke test.

For every `runs/*/best_val.pt` that exists, rebuild a GPT from the saved
GPTConfig (via `load_gpt_config_from_ckpt` which tolerates schema drift),
load the weights, and forward-pass a small batch. This is the end-to-end
verification that the unified exp/14-sft branch can serve as a drop-in
replacement for every historical experiment's code branch — a prerequisite
for the SFT matrix.

Skipped when CUDA is unavailable (we build on CPU but many of these
checkpoints were saved with state_dict keys that may need GPU tensors
initialised).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpt_repro.model import GPT
from gpt_repro.utils import load_checkpoint, load_gpt_config_from_ckpt


RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

EXPECTED_CHECKPOINTS = [
    "baseline",
    "01-modern-block",
    "02-muon",
    "03-modded-tricks",
    "05-speed-pack",
    "06-muon-mup",
    "10-mla",
    "11-loopllm",
]


def _ckpt_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "best_val.pt"


@pytest.mark.parametrize("run_id", EXPECTED_CHECKPOINTS)
def test_ckpt_config_loads(run_id: str) -> None:
    """Smoke: saved GPTConfig round-trips to a live GPTConfig on this branch."""
    path = _ckpt_path(run_id)
    if not path.exists():
        pytest.skip(f"checkpoint not present at {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "config" in ckpt, f"{path} has no 'config' key"
    cfg = load_gpt_config_from_ckpt(ckpt)
    # Every pretrain used 124M-style dims (n_embd=768, n_layer=12, n_head=12)
    # or the 45M variant for LoopLLM. All have block_size=1024.
    assert cfg.block_size == 1024
    assert cfg.n_layer == 12
    assert cfg.n_head == 12
    assert cfg.n_embd == 768


@pytest.mark.parametrize("run_id", EXPECTED_CHECKPOINTS)
def test_ckpt_model_loads_and_forward_passes(run_id: str) -> None:
    """The full load path: saved state_dict fits the rebuilt model + forward is finite."""
    path = _ckpt_path(run_id)
    if not path.exists():
        pytest.skip(f"checkpoint not present at {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = load_gpt_config_from_ckpt(ckpt)
    # Force sdpa_math backend for CPU testing — ignore whatever the ckpt saved.
    cfg.attention_backend = "sdpa_math"
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    # Tiny forward pass to confirm shapes + finite logits.
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (1, 1, cfg.vocab_size)  # targets=None → last-position slice
    assert torch.isfinite(logits).all(), f"non-finite logits from {run_id}"
