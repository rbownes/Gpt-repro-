"""Aggregate the SFT matrix results into a single comparison table.

Reads:
    runs/{ckpt}/results.json              — pretrain val_loss + HellaSwag
    runs/{ckpt}/eval_results.json         — pre-SFT LL battery
    runs/sft-{ckpt}/eval_results.json     — post-SFT LL battery

Writes a JSON payload and prints a markdown-ready table to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHECKPOINTS_DEFAULT = [
    "baseline",
    "01-modern-block",
    "02-muon",
    "03-modded-tricks",
    "05-speed-pack",
    "06-muon-mup",
    "10-mla",
    "11-loopllm",
]

METRICS = ["hellaswag_acc", "mmlu_all_acc", "arc_easy_acc", "arc_challenge_acc"]


def read_json_safe(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"[warn] {p}: {e}", file=sys.stderr)
        return None


def extract_metric(eval_payload: dict | None, metric: str) -> float | None:
    """Both {metric, value, n} top-level list and {results: [...]} shapes handled."""
    if eval_payload is None:
        return None
    results = eval_payload.get("results", eval_payload) if isinstance(eval_payload, dict) else eval_payload
    if isinstance(results, list):
        for r in results:
            if r.get("metric") == metric:
                return float(r["value"])
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS_DEFAULT)
    p.add_argument("--out", default="experiments/14-sft-matrix/matrix.json")
    args = p.parse_args()

    rows = []
    for ckpt in args.checkpoints:
        pretrain_results = read_json_safe(Path(f"runs/{ckpt}/results.json"))
        pre_eval = read_json_safe(Path(f"runs/{ckpt}/eval_results.json"))
        post_eval = read_json_safe(Path(f"runs/sft-{ckpt}/eval_results.json"))

        row = {
            "checkpoint": ckpt,
            "pretrain_val_loss": (
                pretrain_results.get("val_loss") if isinstance(pretrain_results, dict) else None
            ),
        }
        for metric in METRICS:
            pre = extract_metric(pre_eval, metric)
            post = extract_metric(post_eval, metric)
            row[f"pre_{metric}"] = pre
            row[f"post_{metric}"] = post
            row[f"delta_{metric}"] = (post - pre) if (pre is not None and post is not None) else None
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows, "metrics": METRICS}, indent=2))
    print(f"wrote {out_path}")

    # ---- Markdown table -------------------------------------------------
    header = ["checkpoint", "pretrain_val"]
    for m in METRICS:
        short = m.replace("_acc", "").replace("mmlu_all", "mmlu")
        header += [f"pre_{short}", f"post_{short}", f"Δ_{short}"]
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")

    def fmt(v: float | None) -> str:
        if v is None:
            return "—"
        return f"{v:+.3f}" if abs(v) < 0.1 and v != 0.0 else f"{v:.3f}"

    for row in rows:
        cells = [row["checkpoint"]]
        cells.append(fmt(row["pretrain_val_loss"]))
        for m in METRICS:
            cells += [fmt(row[f"pre_{m}"]), fmt(row[f"post_{m}"]), fmt(row[f"delta_{m}"])]
        print("| " + " | ".join(cells) + " |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
