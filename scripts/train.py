"""CLI: train a GPT-2 model from a named config.

Usage:
    uv run python scripts/train.py --config configs.gpt2_124m
    uv run python scripts/train.py --config configs.debug
    uv run python scripts/train.py --config configs.gpt2_124m --override run_dir=runs/exp-03-muon

Overrides use `dotted.key=value` for TrainConfig fields; model overrides
use `model.key=value`.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any


# Make `configs.*` importable without a package, and `gpt_repro` via src/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


from gpt_repro.train import TrainConfig, train  # noqa: E402


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:
        # no type hint available; leave as string
        return value
    return type(current)(value)


def apply_override(cfg: TrainConfig, key: str, value: str) -> None:
    if key.startswith("model."):
        sub = key.removeprefix("model.")
        current = getattr(cfg.model, sub)
        setattr(cfg.model, sub, _coerce(value, current))
        return
    current = getattr(cfg, key)
    if is_dataclass(current):
        raise ValueError(f"cannot override dataclass field {key!r} with a scalar")
    setattr(cfg, key, _coerce(value, current))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="e.g. configs.gpt2_124m")
    p.add_argument(
        "--override", "-o", action="append", default=[],
        help="key=value (repeatable); dotted for nested fields, e.g. model.n_layer=6",
    )
    args = p.parse_args()

    cfg_mod = importlib.import_module(args.config)
    cfg: TrainConfig = cfg_mod.make_config()

    for kv in args.override:
        if "=" not in kv:
            raise ValueError(f"bad override {kv!r}; expected key=value")
        k, v = kv.split("=", 1)
        apply_override(cfg, k, v)

    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
