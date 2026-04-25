"""Extract per-checkpoint config diffs vs faithful GPT-2 baseline.

Reads each `runs/{ckpt}/config.json`, compares against the baseline's
GPTConfig, and prints a markdown table of the non-default fields per
ckpt. Useful as appendix material for `experiments/RETROSPECTIVE.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CKPTS = ["baseline", "01-modern-block", "02-muon", "03-modded-tricks",
         "05-speed-pack", "06-muon-mup", "10-mla", "11-loopllm"]

INTERESTING_FIELDS = [
    "positional_encoding", "norm_type", "mlp_type", "qk_norm",
    "zero_init_proj", "u_net_skips", "logit_softcap",
    "attention_type", "weight_tied",
    "n_kv_head", "use_liger_fused_ce",
    "optimizer", "use_mup",
]


def load_cfg(ckpt: str) -> dict:
    p = Path(f"runs/{ckpt}/config.json")
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    # config.json shape varies — pull out model config dict
    if "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/RETROSPECTIVE_assets/config_diffs.md")
    args = ap.parse_args()

    rows: list[tuple[str, dict]] = []
    for c in CKPTS:
        cfg = load_cfg(c)
        if cfg:
            rows.append((c, cfg))

    if not rows:
        print("no configs found")
        return 1

    base_cfg = rows[0][1]

    lines = ["# Per-checkpoint config diffs vs baseline\n"]
    lines.append("Only fields that differ from the faithful GPT-2 baseline are shown.\n")

    for name, cfg in rows[1:]:
        diffs = []
        for f in INTERESTING_FIELDS:
            if cfg.get(f) != base_cfg.get(f):
                diffs.append((f, base_cfg.get(f), cfg.get(f)))
        if not diffs:
            lines.append(f"\n## {name}\n\n*(no diffs vs baseline on tracked fields)*\n")
            continue
        lines.append(f"\n## {name}\n")
        lines.append("| field | baseline | this checkpoint |")
        lines.append("|-------|----------|-----------------|")
        for f, b, n in diffs:
            lines.append(f"| `{f}` | `{b}` | `{n}` |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
