"""Produce the matplotlib figures used by experiments/RETROSPECTIVE.md.

Reads the master_matrix.json (and per-ckpt training metrics where
helpful) and saves PNGs under experiments/RETROSPECTIVE_assets/.

Figures:
    fig_pretrain_val.png       — per-ckpt pretrain val_loss (bar)
    fig_sft_val.png            — per-ckpt SFT best val (bar)
    fig_delta_rl.png           — Δ_RL per ckpt × task (grouped bar)
    fig_ll_vs_gen_gap.png      — pre-RL LL→gen gap per ckpt × task
    fig_rl_trajectory.png      — eval_r vs step for the 8 RL runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CKPTS = ["baseline", "01-modern-block", "02-muon", "03-modded-tricks",
         "05-speed-pack", "06-muon-mup", "10-mla", "11-loopllm"]
SHORT = {"baseline": "base", "01-modern-block": "01-mod",
         "02-muon": "02-muon", "03-modded-tricks": "03-trk",
         "05-speed-pack": "05-pak", "06-muon-mup": "06-mup",
         "10-mla": "10-mla", "11-loopllm": "11-loop"}
TASKS = ["hswag", "mmlu", "arc_e", "arc_c"]
TASK_LABELS = ["HellaSwag", "MMLU", "ARC-E", "ARC-C"]

# Pretrain val from metrics.jsonl, hard-coded from final read (2026-04-25)
PRETRAIN_VAL = {
    "baseline": 3.041, "01-modern-block": 2.988, "02-muon": 2.988,
    "03-modded-tricks": 2.964, "05-speed-pack": 2.992, "06-muon-mup": 2.974,
    "10-mla": 2.981, "11-loopllm": 3.406,
}

SFT_BEST = {
    "baseline": 1.345, "01-modern-block": 1.308, "02-muon": 1.498,
    "03-modded-tricks": 1.292, "05-speed-pack": 1.324, "06-muon-mup": 1.259,
    "10-mla": 1.304, "11-loopllm": 1.712,
}

CLUSTER = {
    "baseline": "clean", "01-modern-block": "clean", "03-modded-tricks": "clean",
    "10-mla": "clean", "05-speed-pack": "degraded", "06-muon-mup": "degraded",
    "02-muon": "shattered", "11-loopllm": "shattered",
}
CLUSTER_COLOR = {"clean": "#2ca02c", "degraded": "#ff7f0e", "shattered": "#d62728"}


def load_matrix(path: Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def fig_pretrain_val(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = list(range(len(CKPTS)))
    vals = [PRETRAIN_VAL[c] for c in CKPTS]
    colors = [CLUSTER_COLOR[CLUSTER[c]] for c in CKPTS]
    ax.bar(xs, vals, color=colors)
    ax.axhline(PRETRAIN_VAL["03-modded-tricks"], color="gray", linestyle="--", linewidth=1,
               label="v0.3 baseline (2.964)")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[c] for c in CKPTS], rotation=15)
    ax.set_ylabel("val loss @ 10B tokens")
    ax.set_title("Pretrain val loss (lower = better) — colour by SFT cluster")
    ax.set_ylim(2.9, 3.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_pretrain_val.png", dpi=120)
    plt.close()


def fig_sft_val(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = list(range(len(CKPTS)))
    vals = [SFT_BEST[c] for c in CKPTS]
    colors = [CLUSTER_COLOR[CLUSTER[c]] for c in CKPTS]
    ax.bar(xs, vals, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[c] for c in CKPTS], rotation=15)
    ax.set_ylabel("SFT best val loss")
    ax.set_title("SFT best val (lower = better) — colour by SFT cluster")
    ax.set_ylim(1.0, 1.8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_sft_val.png", dpi=120)
    plt.close()


def fig_delta_rl(rows: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    n_tasks = len(TASKS)
    width = 0.18
    xs = np.arange(len(CKPTS))
    for i, (t, label) in enumerate(zip(TASKS, TASK_LABELS)):
        deltas = [r.get(f"{t}_delta_rl", 0) for r in rows]
        deltas = [d if d is not None else 0 for d in deltas]
        ax.bar(xs + (i - n_tasks/2 + 0.5) * width, deltas, width, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[c] for c in CKPTS], rotation=15)
    ax.set_ylabel("Δ_RL (post-RL − pre-RL gen acc)")
    ax.set_title("RL Δ per checkpoint × task — shattered cluster dominates")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_delta_rl.png", dpi=120)
    plt.close()


def fig_ll_vs_gen_gap(rows: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    n_tasks = len(TASKS)
    width = 0.18
    xs = np.arange(len(CKPTS))
    for i, (t, label) in enumerate(zip(TASKS, TASK_LABELS)):
        gaps = []
        for r in rows:
            ll = r.get(f"{t}_sft_ll")
            gen = r.get(f"{t}_sft_gen")
            gaps.append(ll - gen if (ll is not None and gen is not None) else 0)
        ax.bar(xs + (i - n_tasks/2 + 0.5) * width, gaps, width, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[c] for c in CKPTS], rotation=15)
    ax.set_ylabel("LL → gen gap (positive = LL > gen)")
    ax.set_title("Pre-RL verbalisation gap: 'unverbalised knowledge' per ckpt × task")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ll_vs_gen_gap.png", dpi=120)
    plt.close()


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def fig_rl_trajectory(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for ckpt in CKPTS:
        events = _read_jsonl(Path(f"runs/rl-{ckpt}/metrics.jsonl"))
        evals = [(e["step"], e["eval_reward"]) for e in events if e.get("event") == "eval"]
        if not evals:
            continue
        steps, rewards = zip(*evals)
        color = CLUSTER_COLOR[CLUSTER[ckpt]]
        linestyle = "-" if CLUSTER[ckpt] != "shattered" else "--"
        ax.plot(steps, rewards, label=SHORT[ckpt], color=color, linestyle=linestyle, alpha=0.85)
    ax.set_xlabel("RL step")
    ax.set_ylabel("eval reward (held-out 64-prompt MC subset)")
    ax.set_title("RL eval reward trajectory — shattered cluster catches up")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_rl_trajectory.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Per-step training curves (one line per checkpoint, coloured by experiment)
# ---------------------------------------------------------------------------

# Categorical palette — 8 visually distinct colors, one per ckpt.
PALETTE = {
    "baseline":          "#1f77b4",  # blue
    "01-modern-block":   "#ff7f0e",  # orange
    "02-muon":           "#d62728",  # red
    "03-modded-tricks":  "#2ca02c",  # green
    "05-speed-pack":     "#9467bd",  # purple
    "06-muon-mup":       "#8c564b",  # brown
    "10-mla":            "#e377c2",  # pink
    "11-loopllm":        "#17becf",  # cyan
}


def _smooth(xs: list[float], window: int = 50) -> list[float]:
    """Simple moving-average smoothing for jittery training-loss curves."""
    if window <= 1 or len(xs) < window:
        return xs
    out = []
    s = 0.0
    for i, v in enumerate(xs):
        s += v
        if i >= window:
            s -= xs[i - window]
            out.append(s / window)
        else:
            out.append(s / (i + 1))
    return out


def _curve_from_jsonl(path: Path, event: str, key: str) -> tuple[list[int], list[float]]:
    """Pull (step, value) tuples for a given event/key from a metrics.jsonl."""
    if not path.exists():
        return [], []
    steps, vals = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == event and key in d and "step" in d:
            steps.append(int(d["step"]))
            vals.append(float(d[key]))
    return steps, vals


def fig_pretrain_curves(out_dir: Path) -> None:
    """Pretrain val_loss vs step, one line per ckpt."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for ckpt in CKPTS:
        steps, vals = _curve_from_jsonl(
            Path(f"runs/{ckpt}/metrics.jsonl"), "eval", "val_loss",
        )
        if not steps:
            continue
        ax.plot(steps, vals, label=ckpt, color=PALETTE[ckpt], linewidth=1.5, alpha=0.9)
    ax.set_xlabel("pretrain step (524k tokens/step × 19,073 steps = 10B tokens)")
    ax.set_ylabel("val loss")
    ax.set_title("Stage 1 — pretraining: val loss vs step (lower = better)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.7, 6.5)  # crop early high-loss spike
    plt.tight_layout()
    plt.savefig(out_dir / "fig_curves_pretrain.png", dpi=120)
    plt.close()


def fig_sft_curves(out_dir: Path) -> None:
    """SFT val_loss vs step, one line per ckpt."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for ckpt in CKPTS:
        steps, vals = _curve_from_jsonl(
            Path(f"runs/sft-{ckpt}/metrics.jsonl"), "eval", "val_loss",
        )
        if not steps:
            continue
        ax.plot(steps, vals, label=ckpt, color=PALETTE[ckpt], linewidth=1.5, alpha=0.9)
    ax.set_xlabel("SFT step (65k tokens/step × 7,629 steps = 500M SmolTalk tokens)")
    ax.set_ylabel("val loss")
    ax.set_title("Stage 2 — SFT: val loss vs step (lower = better)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_curves_sft.png", dpi=120)
    plt.close()


def fig_rl_curves(out_dir: Path) -> None:
    """RL eval reward vs step, one line per ckpt — the post-training accuracy curve."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for ckpt in CKPTS:
        steps, vals = _curve_from_jsonl(
            Path(f"runs/rl-{ckpt}/metrics.jsonl"), "eval", "eval_reward",
        )
        if not steps:
            continue
        ax.plot(steps, vals, label=ckpt, color=PALETTE[ckpt], linewidth=1.5, alpha=0.9)
    ax.set_xlabel("RL step (32 rollouts/step × 500 steps)")
    ax.set_ylabel("eval reward (= accuracy on held-out 64-prompt MC subset)")
    ax.set_title("Stage 3 — RL: eval reward vs step (higher = better)")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_curves_rl.png", dpi=120)
    plt.close()


def fig_rl_train_loss(out_dir: Path) -> None:
    """RL train loss vs step (smoothed), one line per ckpt — companion to fig_curves_rl."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for ckpt in CKPTS:
        steps, vals = _curve_from_jsonl(
            Path(f"runs/rl-{ckpt}/metrics.jsonl"), "train_step", "loss",
        )
        if not steps:
            continue
        ax.plot(steps, _smooth(vals, window=20), label=ckpt,
                color=PALETTE[ckpt], linewidth=1.2, alpha=0.85)
    ax.set_xlabel("RL step")
    ax.set_ylabel("GRPO loss (smoothed window=20, can be ±)")
    ax.set_title("Stage 3 — RL: GRPO loss vs step (smoothed)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_curves_rl_loss.png", dpi=120)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="experiments/15-rl-matrix/master_matrix.json")
    ap.add_argument("--out-dir", default="experiments/RETROSPECTIVE_assets")
    args = ap.parse_args()

    rows = load_matrix(Path(args.matrix))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_pretrain_val(out_dir)
    fig_sft_val(out_dir)
    fig_delta_rl(rows, out_dir)
    fig_ll_vs_gen_gap(rows, out_dir)
    fig_rl_trajectory(out_dir)
    fig_pretrain_curves(out_dir)
    fig_sft_curves(out_dir)
    fig_rl_curves(out_dir)
    fig_rl_train_loss(out_dir)
    print(f"wrote 9 PNGs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
