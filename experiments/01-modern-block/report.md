---
id: 01-modern-block
status: accepted
baseline_run: runs/baseline/
experiment_run: runs/01-modern-block/
baseline_tag: v0.1-baseline
date: 2026-04-19
author: rjbownes
seeds: [0]
---

# Experiment 01 — Modern block: RoPE + RMSNorm + SwiGLU + QK-Norm

## Previous baseline

- **Config:** `configs/gpt2_124m.py` @ commit `626509c`, git tag `v0.1-baseline`.
- **Arch:** faithful GPT-2 124M (LayerNorm, learned pos-emb, GELU, no QK-Norm).
- **Optimizer:** AdamW (β = 0.9 / 0.95, wd = 0.1, peak LR 6 e-4, cosine decay, 715 warmup).
- **Data:** FineWeb-Edu-10B, 1 epoch = 19 073 steps @ 524 288 tok/step.
- **Baseline metrics (single seed):**
  - val loss @ 1 B / 5 B / 10 B tokens: **3.5984 / 3.1669 / 3.0407**
  - HellaSwag (1 000 val examples): **0.3680**
  - tokens/s median: **190 004** · wall-clock 1 epoch: **14 h 41 min**

## The change

Swap four pieces of the decoder block from the 2019 GPT-2 recipe to the 2024-era
Llama-style recipe, all simultaneously as one bundled "modernization" step. Same
parameter count, same optimizer, same LR schedule, same data, same token budget.

- **Diff:** on branch `exp/01-modern-block`. See pending commit.
- **Files touched:** `src/gpt_repro/model.py`, `configs/gpt2_124m_modern.py`
  (new), `tests/test_model_shapes.py`.
- **Hyperparameters introduced / changed:**
  - `positional_encoding`: `"learned"` → `"rope"` (`rope_base = 10000.0`)
  - `norm_type`: `"layernorm"` → `"rmsnorm"`
  - `mlp_type`: `"gelu"` → `"swiglu"`
  - `qk_norm`: `false` → `true`
  - MLP hidden size: `4·d = 3072` (GELU, 2 matrices) → `2048` ≈ `8/3·d`
    (SwiGLU, 3 matrices) for parameter parity
  - `wpe` embedding is dropped entirely (replaced by RoPE)

## Why it might improve

References (each with its own consensus evidence base):

- **RoPE** ([Su et al. 2021](https://arxiv.org/abs/2104.09864)) — relative
  positional information via rotation of Q/K, avoids the learned `wpe`
  embedding and generalises better to longer contexts. Adopted by
  Llama/Qwen/DeepSeek/modded-nanogpt.
- **RMSNorm** ([Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)) —
  drops the centering term of LayerNorm, ~5 – 10 % faster, no quality
  regression in large-scale practice.
- **SwiGLU** ([Shazeer 2020](https://arxiv.org/abs/2002.05202)) — `SiLU(Wg·x) ·
  (Wu·x)` then `Wd`. Three matrices instead of two; with hidden `8/3 · d` the
  total params match the original `4 · d` GELU MLP. Reliably ~1 – 2 %
  perplexity win across scales.
- **QK-Norm** ([Henry et al. 2020](https://arxiv.org/abs/2010.04245),
  popularised by modded-nanogpt) — RMSNorm applied per-head to Q and K before
  the attention matmul. Prevents logit-magnitude blowup, enables higher
  learning rates, improves stability. Used in every modded-nanogpt speedrun.

### Mechanism

Collectively these four changes are the industry-standard "post-GPT-2 decoder
block". Each contributes a small win. Together, the community's consensus
gain over a faithful GPT-2 block at 124 M scale on a clean pre-training corpus
is **~0.05 – 0.15 val-loss reduction at fixed tokens**.

### Predicted effect (written BEFORE running)

- val loss Δ @ 10 B tokens: **−0.05 to −0.12**. Point prediction: **−0.08**
  (target val loss **~2.96**).
- HellaSwag acc Δ: **+0.5 to +2 %** absolute (target **~37 – 38 %**).
- tokens/s Δ: **−3 % to +5 %**. RMSNorm is slightly faster than LayerNorm,
  SwiGLU adds a matmul, RoPE adds a cheap rotation; QK-Norm adds another
  RMSNorm. Net should be near neutral on this GPU.
- No expected regression on stability; if anything QK-Norm should make
  optimisation safer at the same LR.

### Accept criteria

- **Accept if** val loss @ 10 B tokens drops by **≥ 0.05** (i.e. ≤ 2.99), **and**
  tokens/s regression is **≤ 5 %** (i.e. ≥ 180.5 k tok/s). Both must hold.
- **Reject** if val loss Δ < 0.02, or if tokens/s regresses > 10 %, or if the
  training curve shows instability (loss spikes not seen in baseline, NaN,
  divergent gradient norms).

## Implementation notes

- The faithful baseline must remain reproducible from the same codebase.
  Architecture variants are gated by new `GPTConfig` flags with faithful
  defaults, so `configs/gpt2_124m.py` produces exactly the baseline run and
  `tests/test_hf_weight_load.py` continues to pass (it loads HF gpt2 weights
  into the faithful layout).
- SwiGLU hidden dim set to **2 048** (not the more common `8/3 · 768 = 2 048`
  rounded down — this is already an integer) so the total MLP params are
  `3 · 768 · 2 048 = 4 718 592`, matching GELU MLP's `2 · 768 · 3 072 =
  4 718 592`. Zero parameter drift vs baseline from the MLP swap.
- Dropping `wpe` removes `1 024 · 768 = 786 432` params from the embedding
  side. Total trainable-param count is therefore **~0.7 % lower** than the
  faithful baseline (123.7 M vs 124.4 M). This is a well-understood asymmetry
  and does not affect the comparison at fixed tokens.
- RoPE is applied inside `CausalSelfAttention` on Q and K only (never on V),
  with fp32 rotation and bf16 output, standard practice.
- QK-Norm uses dim-`head_dim` RMSNorm (per-head, not per-tensor), matching
  modded-nanogpt's convention.

## Result

| metric                          | baseline | exp/01 | Δ |
|---------------------------------|---------:|-------:|---:|
| val loss @ 1 B tokens           | 3.5984   | **3.4641** | **−0.134** |
| val loss @ 5 B tokens           | 3.1669   | **3.1009** | **−0.066** |
| val loss @ 10 B tokens (best)   | 3.0402   | **2.9884** | **−0.052** |
| val loss (200-batch held-out)   | 3.0473   | **2.9946** | **−0.053** |
| HellaSwag acc (1 000 examples)  | 0.3680   | **0.3830** | **+0.015** |
| tokens / s median               | 190 004  | 181 732    | −4.4 %  |
| tokens / s mean                 | 189 345  | 181 240    | −4.3 %  |
| wall-clock 1 epoch              | 14 h 41 min | 15 h 20 min | +39 min |
| time to val loss 3.5            | 140 min  | **96.6 min**  | **−31 %** |
| time to val loss 3.2            | 394 min  | **290 min**   | **−26 %** |
| time to val loss 3.10           | 601 min  | **483 min**   | **−20 %** |
| time to val loss 3.04 (baseline's final) | N/A | **652 min** | matches baseline quality in **74 %** of baseline wall-clock |

### Predicted vs actual

- Predicted val loss Δ: **−0.05 to −0.12** (point **−0.08**). Actual: **−0.052** — landed at the bottom of the predicted range. A respectful win but *smaller* than the point estimate, which is worth noting: the modernization wins most of its edge early in training (−0.134 at 1 B tokens, shrinking to −0.052 by 10 B) because both curves are still converging and faithful GPT-2 catches up with enough tokens.
- Predicted tok/s Δ: **−3 % to +5 %**. Actual: **−4.4 %** — slightly worse than the range, but within the 5 % accept threshold. The SwiGLU extra matmul + two RMSNorms for QK-Norm dominate the cost; RMSNorm was not fast enough to offset.
- Predicted HellaSwag Δ: **+0.5 to +2 pp**. Actual: **+1.5 pp** — mid-range.

### Loss curve shape

Modern config is consistently ahead of faithful at every eval step after step 500. The Δ is largest early (−0.61 at step 500, −0.27 at step 1 000) and converges asymptotically (−0.05 by step 19 073). This is consistent with the "cleaner gradient flow early" reading of RMSNorm + QK-Norm — they mostly save you from the awkward warmup phase.

## Verdict

**Accept.** All pre-declared criteria met:

- val loss @ 10 B Δ: **−0.052** (required ≥ −0.05) ✅
- tok/s regression: **4.4 %** (required ≤ 5 %) ✅
- No training instability; no NaN, no loss spikes, smooth cosine decay to completion.
- HellaSwag +1.5 pp (secondary, pre-predicted range +0.5 to +2 pp) ✅

**Advance `v0.2-exp01` tag** at this commit. Future experiments branch from
**`v0.2-exp01`** and treat the modern block as the new faithful baseline.

Explicitly not changing `configs/gpt2_124m.py` — that remains the faithful
Radford baseline for reproducibility purposes; new experiments should fork
`configs/gpt2_124m_modern.py` instead.

