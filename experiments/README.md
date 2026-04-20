# Experiments

This directory is the experiment log. Every improvement to the baseline is run
as a **single-variable experiment** in its own subdirectory, with a filled-in
`report.md` based on [`TEMPLATE.md`](./TEMPLATE.md).

## Discipline

1. **Start from a frozen baseline.** `00-baseline/` is the reference. Tag its
   commit `v0.1-baseline`. Improvements branch off that tag.
2. **One diff, one experiment.** Never change arch + optimizer together.
3. **Match tokens-seen**, not wall-clock, when comparing quality. Every run
   trains on exactly 10 B tokens unless the experiment *is about* compute
   efficiency at fixed tokens.
4. **Write the hypothesis and accept/reject criteria before training.**
5. **Multi-seed (0, 1, 2)** for any effect under 0.05 val loss.
6. **Kill-early**: if at 1 B tokens the val loss is > 0.05 worse than baseline
   and the change is supposed to *help*, stop.

## Metrics reported every experiment

- Validation cross-entropy on the held-out FineWeb-Edu val shard (same positions every run).
- `tokens/s` (median over last 1000 steps after compile warmup).
- Time-to-val-loss-3.0, wall-clock to fixed token budget.
- `val_loss @ 1 B / 5 B / 10 B tokens` (compute-efficiency curve).
- End-of-training paper-battery: WikiText-103 PPL, LAMBADA acc, HellaSwag acc
  (via `lm-eval` for comparability).

## Roadmap

Ordered by expected ROI on a single RTX 5090 at 124M scale. See the project
plan for the full rationale and references.

| # | Improvement | Ref | Status |
|---|---|---|---|
| 00 | Faithful GPT-2 124M on FineWeb-Edu-10B | — | **accepted** (val 3.040, HellaSwag 36.8 %, 190 k tok/s, 14 h 41 min) |
| 01 | Block modernization: RoPE + RMSNorm + SwiGLU + QK-Norm | [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | **accepted** (val 2.988, HellaSwag 38.3 %, 182 k tok/s, 15 h 20 min) |
| 02 | AdamW → Muon on hidden matmuls | [Muon blog](https://kellerjordan.github.io/posts/muon/) | **rejected** (val Δ −0.0007 @ 10 B; time-to-target 14–25 % faster; tok/s −0.7 %) |
| 03 | Full modded-nanogpt recipe (ReLU², zero-init, U-Net skips, logit softcap) | [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | **accepted** (val 2.964, HellaSwag 37.8 %, 178 k tok/s, 15 h 39 min) |
| 04 | FP8 matmul via TransformerEngine | [TE](https://github.com/NVIDIA/TransformerEngine) | pending |
| 05 | μP / μTransfer HP sweep on 20 M proxy | [mup](https://github.com/microsoft/mup) | pending |
| 06 | GQA (4 KV heads) | Llama-2 | pending |
| 07 | `flash-attn==2.8.0.post2` opt-in backend (calibration) | [FA #2016](https://github.com/Dao-AILab/flash-attention/issues/2016) | pending |
| 08 | Differential Transformer | [arXiv 2410.05258](https://arxiv.org/abs/2410.05258) | pending |
| 09 | Native Sparse Attention (NSA) | [arXiv 2502.11089](https://arxiv.org/abs/2502.11089) | pending |
| 10 | MLA (DeepSeek-V2) | [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) | pending |
| 11 | DeepSeek Engram (associative memory) | [arXiv 2601.07372](https://arxiv.org/abs/2601.07372) | pending |
| 12 | Mixture of Recursions / Looped transformers | [arXiv 2507.10524](https://arxiv.org/abs/2507.10524) | pending |
| 13 | BitNet b1.58 (inference-footprint comparison only) | [arXiv 2504.12285](https://arxiv.org/abs/2504.12285) | pending |

## Index of runs

| ID | Summary | Status | val_loss | tok/s | Verdict |
|---|---|---|---|---|---|
| 00-baseline | Faithful 124M on FW-Edu-10B | accepted | **3.040** | 190 k | reference run; tagged `v0.1-baseline` @ `626509c` |
| 01-modern-block | RoPE + RMSNorm + SwiGLU + QK-Norm | accepted | **2.988** (Δ −0.052) | 182 k (−4.4 %) | accept; HellaSwag +1.5 pp; tagged `v0.2-exp01` |
| 02-muon | AdamW → Muon (hidden matmuls) | rejected | 2.988 (Δ −0.0007) | 180 k (−0.7 %) | reject on val-loss axis; time-to-val-3.1 −14 %; null-result preserved |
| 03-modded-tricks | ReLU² + zero-init + U-Net skips + logit softcap | accepted | **2.964** (Δ −0.024) | 178 k (−1.9 %) | accept; HellaSwag −0.5 pp (within noise); tagged `v0.3-exp03` |
