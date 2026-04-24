"""Run the log-likelihood eval battery against a run directory.

Usage:
    # Against a pretrained checkpoint (pre-SFT baseline):
    uv run python scripts/sft_eval.py --run-dir runs/03-modded-tricks

    # Against an SFT'd checkpoint:
    uv run python scripts/sft_eval.py --run-dir runs/sft-03-modded-tricks

By default picks `best_val.pt` in the run dir. Writes `eval_results.json`
alongside the checkpoint with per-task accuracy + metadata.

Battery: HellaSwag, MMLU (all), ARC-Easy, ARC-Challenge. All are 4-way
log-likelihood multiple-choice (length-normalised mean per-token LL).
No decoder/generation needed — matches the pretrain eval conventions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt_repro.eval import (  # noqa: E402
    EvalResult, arc_challenge, arc_easy, hellaswag, mmlu,
)
from gpt_repro.model import GPT, select_sdpa_backend_globally  # noqa: E402
from gpt_repro.utils import (  # noqa: E402
    autocast_dtype, device_str, load_gpt_config_from_ckpt, tune_pytorch_globals,
)


def load_model_from_ckpt(ckpt_path: Path, device: str) -> GPT:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_gpt_config_from_ckpt(ck)
    if device == "cuda":
        select_sdpa_backend_globally(cfg.attention_backend)
    model = GPT(cfg).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model


def format_row(r: EvalResult) -> str:
    return f"  {r.metric:<24} {r.value:.4f}  (n={r.n})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=str)
    p.add_argument("--ckpt-name", default="best_val.pt")
    p.add_argument("--out", default=None, help="override output path (default: <run-dir>/eval_results.json)")
    # Per-task limits (0 = full split).
    p.add_argument("--hellaswag-limit", type=int, default=1000)
    p.add_argument("--mmlu-limit", type=int, default=0)       # MMLU val is 1540 — fast
    p.add_argument("--arc-easy-limit", type=int, default=0)
    p.add_argument("--arc-challenge-limit", type=int, default=0)
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["hellaswag", "mmlu", "arc_easy", "arc_challenge"])
    p.add_argument("--only", nargs="*", default=None,
                   choices=["hellaswag", "mmlu", "arc_easy", "arc_challenge"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tune_pytorch_globals()
    device = device_str()
    amp = autocast_dtype()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / args.ckpt_name
    if not ckpt_path.exists():
        print(f"[error] no checkpoint at {ckpt_path}", file=sys.stderr)
        return 1

    print(f"loading {ckpt_path} ...")
    model = load_model_from_ckpt(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params | device: {device} | amp: {amp}")

    tasks: list[str] = ["hellaswag", "mmlu", "arc_easy", "arc_challenge"]
    if args.only:
        tasks = [t for t in tasks if t in args.only]
    tasks = [t for t in tasks if t not in args.skip]

    results: list[EvalResult] = []

    def lim(n: int) -> int | None:
        return n if n > 0 else None

    if "hellaswag" in tasks:
        try:
            r = hellaswag(model, amp_dtype=amp, limit=lim(args.hellaswag_limit))
            results.append(r)
            print(format_row(r))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] hellaswag: {e}")

    if "mmlu" in tasks:
        try:
            r = mmlu(model, amp_dtype=amp, limit=lim(args.mmlu_limit))
            results.append(r)
            print(format_row(r))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] mmlu: {e}")

    if "arc_easy" in tasks:
        try:
            r = arc_easy(model, amp_dtype=amp, limit=lim(args.arc_easy_limit))
            results.append(r)
            print(format_row(r))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] arc_easy: {e}")

    if "arc_challenge" in tasks:
        try:
            r = arc_challenge(model, amp_dtype=amp, limit=lim(args.arc_challenge_limit))
            results.append(r)
            print(format_row(r))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] arc_challenge: {e}")

    out_path = Path(args.out) if args.out else run_dir / "eval_results.json"
    payload = {
        "ckpt": str(ckpt_path),
        "n_params": n_params,
        "results": [
            {"metric": r.metric, "value": r.value, "n": r.n, "extra": r.extra}
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
