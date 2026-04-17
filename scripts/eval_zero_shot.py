"""CLI: zero-shot eval of a trained checkpoint.

Usage:
    uv run python scripts/eval_zero_shot.py \
        --ckpt runs/baseline/best_val.pt \
        --hellaswag-limit 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from gpt_repro.data import DataConfig, ShardLoader
from gpt_repro.eval import EvalResult, hellaswag, val_loss
from gpt_repro.model import GPT, GPTConfig
from gpt_repro.utils import autocast_dtype, device_str, load_checkpoint, tune_pytorch_globals


def build_model_from_checkpoint(ckpt_path: str, device: str) -> GPT:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    load_checkpoint(ckpt_path, model=model, map_location="cpu")
    return model.to(device)


def format_row(r: EvalResult) -> str:
    return f"  {r.metric:<20} {r.value:.4f}  (n={r.n})"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    p.add_argument("--data-dir", default="data/fineweb_edu_10B")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--val-batches", type=int, default=100)
    p.add_argument("--hellaswag-limit", type=int, default=1000, help="0 = all ~10k val examples")
    p.add_argument("--out", default=None, help="optional results.json path")
    args = p.parse_args()

    tune_pytorch_globals()
    device = device_str()
    amp = autocast_dtype()

    model = build_model_from_checkpoint(args.ckpt, device)
    print(f"loaded {args.ckpt} | device {device} | amp {amp}")

    results: list[EvalResult] = []

    # Validation loss
    if Path(args.data_dir).exists():
        vl = ShardLoader(DataConfig(
            data_dir=args.data_dir, block_size=model.cfg.block_size,
            batch_size=args.micro_batch, split="val",
        ))
        r = val_loss(
            model,
            vl.iter_val(device, max_batches=args.val_batches),
            amp,
        )
        results.append(r)
        print(format_row(r))
    else:
        print(f"[skip] val loss: {args.data_dir} missing")

    # HellaSwag (streams from HF Hub)
    limit = args.hellaswag_limit or None
    try:
        r = hellaswag(model, amp_dtype=amp, limit=limit)
        results.append(r)
        print(format_row(r))
    except Exception as e:  # noqa: BLE001 — user-visible summary is fine.
        print(f"[skip] hellaswag: {e}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            [{"metric": r.metric, "value": r.value, "n": r.n} for r in results],
            indent=2,
        ))
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
