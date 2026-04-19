---
id: 02-muon
status: rejected
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

| metric                          | baseline (v0.2) | exp/02 | Δ |
|---------------------------------|---------------:|-------:|---:|
| val loss @ 1 B tokens           | 3.4641         | **3.3753** | **−0.089** |
| val loss @ 5 B tokens           | 3.1009         | **3.0792** | **−0.022** |
| val loss @ 10 B tokens (best)   | 2.9884         | **2.9877** | **−0.0007** |
| val loss (200-batch held-out)   | 2.9946         | 2.9945   | −0.0001 |
| HellaSwag acc (1 000 examples)  | 0.3830         | **0.3850** | +0.002 (+0.2 pp) |
| tokens / s median               | 181 732        | 180 476  | **−0.7 %** |
| tokens / s mean                 | 181 240        | 179 893  | −0.7 % |
| wall-clock 1 epoch              | 15 h 20 min    | 15 h 27 min | +7 min |
| time to val loss 3.5            | 96.6 min       | **72.8 min** | **−25 %** |
| time to val loss 3.2            | 290 min        | **243 min**  | **−16 %** |
| time to val loss 3.10           | 483 min        | **413 min**  | **−14 %** |
| time to val loss ≈ 2.99         | N/A (baseline's ≈final) | 899 min | ~97 % of baseline wall-clock |

### Predicted vs actual

- Predicted val loss Δ: **−0.03 to −0.10** (point **−0.05**). Actual: **−0.0007** — **miss**.
  The gap collapsed steadily through training: −0.46 at step 500, −0.11 at 1 500,
  −0.04 at 5 000, −0.01 at 13 500, ~0 at the end. Cosine decay + enough tokens
  lets AdamW catch up completely.
- Predicted tok/s Δ: **−2 % to +1 %**. Actual: **−0.7 %** — in range.
  The smoke run suggested −10 %; that was compile-warmup noise and small-batch
  dominance of the Newton-Schulz step. Steady-state on the full run is
  essentially neutral on SM_120. Prior under-estimate of Muon's kernel overhead
  was wrong; the honest number is near-free.
- Predicted HellaSwag Δ: **+0.5 to +2 pp**. Actual: **+0.2 pp** — below range
  but directionally consistent.

### What actually went on

Muon converges to **the same loss** as AdamW on this setup. It also *gets
there sooner* — 25 % faster to val loss 3.5, 14 % faster to 3.1 — consistent
with the mechanism (orthogonal updates help escape the warmup region faster).
But on a fixed 10 B-token budget with cosine decay to 10 % peak, AdamW has
enough steps to match Muon's late-training behaviour, so the asymptotic
val loss is a dead tie.

This is a **documented null result at 124 M × 10 B tokens**, not a Muon bug.
At larger scale (Kimi K2, DeepSeek-V3, INTELLECT-3) Muon still wins because
those runs never converge the optimizer curves. Here we *do* converge them.

## Verdict

**Reject.** The pre-declared accept criterion was val loss Δ ≥ 0.03 at 10 B
tokens; actual is −0.0007. Even though Muon comfortably cleared the throughput
budget (−0.7 % vs −5 % allowed) and posted a real time-to-intermediate-loss
win, the chosen primary metric was asymptotic loss at fixed tokens, and on
that metric Muon did not beat AdamW.

### What we learned that the null result doesn't hide

1. **Muon is ~free on SM_120.** The tok/s regression I feared didn't
   materialize steady-state. Future experiments with truncated schedules
   or early-stopping could adopt Muon at effectively zero cost.
2. **Time-to-target-loss is a real axis Muon wins on.** If the goal ever
   shifts from "10 B tokens" to "shortest wall-clock to val loss 3.1", Muon
   is the right optimizer — 14 % faster, no stability issues.
3. **Asymptotic-loss wins from Muon don't ship at 124 M × 10 B with full
   cosine decay.** Don't bundle Muon into the full modded-nanogpt recipe
   until we can run without full cosine decay (e.g. as part of μTransfer
   sweeps that are LR-limited by time, not by target loss).

### Follow-up

- `v0.2-exp01` remains the baseline tag. **No new `v0.x` tag** — rejected
  experiments don't advance the version.
- `exp/02-muon` branch + this report committed and preserved for audit.
- Next candidate is **exp/03: full modded-nanogpt recipe** (ReLU² MLP,
  zero-init projections, embedding→block skip, logit softcap). The
  speedrun work accumulates Muon *alongside* these tricks, not in isolation;
  introducing them together may unlock Muon's edge on 10 B.

