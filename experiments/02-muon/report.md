---
id: 02-muon
status: in-progress
baseline_run: runs/01-modern-block/
experiment_run: runs/02-muon/
baseline_tag: v0.2-exp01
date: 2026-04-19
author: rjbownes
seeds: [0]
---

# Experiment 02 — Muon optimizer on hidden matmuls

## Previous baseline

- **Config:** `configs/gpt2_124m_modern.py` @ commit `a8141ed`, git tag `v0.2-exp01`.
- **Arch:** modern block — RoPE + RMSNorm + SwiGLU + QK-Norm, 124 M params.
- **Optimizer:** AdamW (β = 0.9 / 0.95, wd = 0.1, peak LR 6 e-4, cosine decay, 715 warmup) on **all** parameters.
- **Data:** FineWeb-Edu-10B, 19 073 steps @ 524 288 tok/step (~10 B tokens).
- **Baseline metrics (single seed):**
  - val loss @ 1 B / 5 B / 10 B tokens: **3.4641 / 3.1009 / 2.9884**
  - HellaSwag (1 000 val examples): **0.3830**
  - tokens/s median: **181 732** · wall-clock 1 epoch: **15 h 20 min**

## The change

Swap AdamW → **Muon** on every 2-D hidden matmul weight (attention projections:
`c_attn`, `c_proj`; MLP: `w_gate`, `w_up`, `w_down`). Keep AdamW for everything
else (embeddings, RMSNorm gains, per-head QK-Norm gains, any biases). Learning
rates decoupled: **Muon peak 0.02**, **AdamW peak 6 e-4**; both follow the same
cosine schedule with the same 715-step linear warmup so the ratio is constant.

- **Diff:** on branch `exp/02-muon`. See pending commits.
- **Files touched:** `src/gpt_repro/muon.py` (new), `src/gpt_repro/optim.py`,
  `src/gpt_repro/train.py`, `src/gpt_repro/utils.py`, `configs/gpt2_124m_muon.py` (new), `tests/test_muon.py` (new).
- **Hyperparameters introduced / changed:**
  - `optimizer_type`: `"adamw"` → `"muon+adamw"`
  - `muon_peak_lr`: — → `0.02`
  - `muon_momentum`: — → `0.95`
  - `muon_nesterov`: — → `True`
  - `muon_ns_steps`: — → `5`
  - AdamW peak LR **unchanged** at `6 e-4`.
  - Neither optimizer receives any weight decay in this experiment (Muon
    weight-decay interacts nontrivially with its orthogonalised update; keep
    it clean and match nanogpt-speedrun defaults).

## Why it might improve

Reference: [Keller Jordan, "Muon: A practical optimizer for transformer hidden
weights"](https://kellerjordan.github.io/posts/muon/) and ["Muon is Scalable
for LLM Training" (arXiv 2502.16982)](https://arxiv.org/abs/2502.16982).

**Mechanism (one paragraph):** After applying momentum, Newton-Schulz
orthogonalises the per-weight-matrix update so every singular direction gets a
step of roughly equal magnitude. For SGD-momentum and Adam-style updates the
largest singular directions dominate and smaller directions barely move. Muon
argues this is the wrong prior for transformer hidden matmuls — orthogonal
updates train the whole matrix at the same rate, and empirically reach the
same loss in ~30 – 35 % fewer tokens at GPT-2 – Llama scale. Embeddings and the
LM head are *not* matmuls in the same sense (they're lookup-like operations
whose singular spectrum is naturally long-tailed), so those stay on AdamW as
the Muon author recommends.

### Predicted effect (written BEFORE running)

- **val loss Δ @ 10 B tokens:** −0.03 to −0.10. Point prediction **−0.05**
  (target val loss **~2.94**). Muon's usual gain is larger at small scales;
  we are already on the modern block which partially eats the same budget,
  so I'm expecting the lower end of the typical range.
- **tokens/s Δ:** **−2 % to +1 %.** Newton-Schulz adds ≈10 bf16 matmuls per
  step per 2-D parameter (60-ish matmuls per step), but these run on a 5090
  tensor core and are tiny compared to the main forward/backward. Should be
  near neutral on this GPU.
- **HellaSwag Δ:** **+0.5 to +2 pp**. Usually tracks val loss at small scales.
- **Stability:** no regression expected. Muon with Newton-Schulz is well-
  validated at larger scales (Kimi K2, GLM-4.5, INTELLECT-3 all used it).

### Accept criteria

- **Accept if** val loss @ 10 B tokens drops by **≥ 0.03** (i.e. ≤ 2.958),
  **and** tokens/s regression is **≤ 5 %** (≥ 172.6 k tok/s). Both must hold.
- **Reject** if val loss Δ < 0.01, or tokens/s regression > 10 %, or training
  instability (NaN, loss spikes absent from baseline, divergent gradient
  norms, orthogonalisation producing non-finite updates).
- **Kill-early:** at the 1 B-token eval (step ≈ 2 000), if val loss is > 0.05
  *worse* than the v0.2 baseline at the same step (i.e. > 3.51), stop.

## Implementation notes

- **Muon impl:** `src/gpt_repro/muon.py`. Newton-Schulz uses the standard
  quintic iteration `a, b, c = (3.4445, -4.7750, 2.0315)`, 5 iterations,
  normalised input via `X / (X.norm() + 1e-7)`, bf16 internal math,
  back-cast to the param's dtype. Post-orthogonalisation scale factor
  `max(1, d_out/d_in)^0.5` keeps the update magnitude scale-invariant
  across layer shapes.
- **Param splitting:** purely by shape + name:
  - Muon: `p.ndim == 2` and name ends in one of `c_attn.weight`,
    `c_proj.weight`, `w_gate.weight`, `w_up.weight`, `w_down.weight`.
  - AdamW: everything else — `wte.weight` (tied, so also `lm_head`), RMSNorm
    `weight`, QK-Norm `q_norm.weight` / `k_norm.weight`, any `*.bias`.
- **Schedule:** one scalar `t ∈ [0,1]` drives both LRs; Muon scales with
  `muon_peak_lr × schedule(t)` and AdamW with `adamw_peak_lr × schedule(t)`.
  Shared 715-step linear warmup, cosine decay to 10 % of peak. This keeps the
  comparison fair to the baseline's schedule shape.
- **Grad clip:** still applied to the union of all parameters at 1.0.
- **torch.compile:** only the model's forward is compiled; optimiser steps
  are not. The Muon step executes eagerly, so no compile-incompatibility risk.
- **Checkpointing:** both optimisers' states saved and loaded in `.pt`.

## Result

(To be filled in after training completes.)

| metric                          | baseline (v0.2) | exp/02 | Δ |
|---------------------------------|---------------:|-------:|---:|
| val loss @ 1 B tokens           | 3.4641         |        |   |
| val loss @ 5 B tokens           | 3.1009         |        |   |
| val loss @ 10 B tokens          | 2.9884         |        |   |
| tokens / s median               | 181 732        |        |   |
| wall-clock 1 epoch              | 15 h 20 min    |        |   |
| HellaSwag acc (1 000 examples)  | 0.3830         |        |   |

## Verdict

(Pending.)
