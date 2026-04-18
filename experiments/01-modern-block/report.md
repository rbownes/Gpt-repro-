---
id: 01-modern-block
status: in-progress
baseline_run: runs/baseline/
experiment_run: runs/01-modern-block/
baseline_tag: v0.1-baseline
date: 2026-04-18
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

(To be filled in after training completes. See accept/reject criteria above.)

| metric                          | baseline | experiment | Δ |
|---------------------------------|---------:|-----------:|---|
| val loss @ 1 B tokens           | 3.5984   |            |   |
| val loss @ 5 B tokens           | 3.1669   |            |   |
| val loss @ 10 B tokens          | 3.0407   |            |   |
| tokens / s median               | 190 004  |            |   |
| wall-clock 1 epoch              | 14 h 41 min |         |   |
| HellaSwag acc (1 000 examples)  | 0.3680   |            |   |

## Verdict

(Pending.)
