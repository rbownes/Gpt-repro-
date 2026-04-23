---
id: 11-loopllm
status: in-progress
baseline_run: runs/03-modded-tricks/
experiment_run: runs/11-loopllm/
baseline_tag: v0.3-exp03
date: 2026-04-23
author: rjbownes
seeds: [0]
---

# Experiment 11 — LoopLLM (pure weight-tied looping, K = n_layer = 12)

## Previous baseline

- **Config:** `configs/gpt2_124m_modernplus.py` @ git tag `v0.3-exp03` (commit `e70512c`).
- **Arch:** modern block (RoPE + RMSNorm + QK-Norm) + modded-nanogpt tricks (ReLU² MLP, zero-init out-projections, U-Net skips, `logit_softcap=30`), plain MHA attention.
- **Optimizer:** AdamW β=(0.9, 0.95), wd=0.1, peak LR 6e-4, cosine decay to 0.1×, 715 warmup steps.
- **Data:** FineWeb-Edu-10B, 19 073 steps @ 524 288 tok/step (= 10 B tokens).
- **Baseline metrics (seed 0):**
  - val loss @ 1 B / 5 B / 10 B tokens: **3.4729 / 3.0780 / 2.9641** (best = **2.9694** on 200-batch held-out)
  - HellaSwag acc (1 000 examples): **0.3780**
  - tokens/s median: **178 215** · wall-clock: **15 h 39 min**
  - 123.7 M params

## The change

Two simultaneous modifications on v0.3:

1. **`weight_tied = True`** — a single `Block` module is instantiated and applied `n_layer = 12` times during forward. Gradients accumulate into the one shared set of block weights.
2. **`u_net_skips = False`** — forced off by a hard `ValueError` in `GPT.__init__` when `weight_tied=True` (the skip pattern is degenerate when every "layer" is the same module).

All other flags — `qk_norm`, `zero_init_proj`, `logit_softcap=30`, `mlp_type="relu2"`, RoPE — plus the optimizer, schedule, batch size, data, and token budget — are **identical to v0.3-exp03**.

- **Diff:** branch `exp/11-loopllm` (off `main`, at `71ae6ad`, the exp/03 perf tip).
- **Files touched:**
  - `src/gpt_repro/model.py`: new `weight_tied` field on `GPTConfig`; branch in `GPT.__init__` builds a single `Block` and a length-1 `transformer.h`; `GPT.forward` loops the shared block `n_layer` times; `ValueError` if `weight_tied + u_net_skips`.
  - `configs/gpt2_124m_modernplus_looped.py`: the exp/11 config.
  - `tests/test_model_shapes.py`: 5 new tests (single-block instance, n_layer forward-calls, u_net_skips rejection, param-count ratio, overfit sanity).
- **Hyperparameters introduced:** `weight_tied: bool = False` (off by default; existing configs unchanged).

### Parameter accounting — model is 63 % smaller

| component                    | v0.3 untied | exp/11 tied | Δ |
|------------------------------|------------:|------------:|---:|
| embeddings (tied wte/lm_head) | 38.6 M      | 38.6 M      | 0 |
| block (attn + MLP + norms)    | 12 × 7.08 M = **85.0 M** | 1 × 7.08 M = **7.08 M** | **−77.9 M** |
| final LN                      | 0.001 M     | 0.001 M     | 0 |
| **total**                     | **123.7 M** | **45.7 M**  | **−78.0 M (−63 %)** |

The test `test_weight_tied_param_count_at_124m` guards this ratio at 0.35–0.42.

### Scope note (important)

This diff bundles two changes (tying + U-Net off) because one mathematically requires the other. It is *not* strictly a "one-diff, one-experiment" run per the `experiments/README.md` discipline — U-Net off alone was accepted-feature-removal worth ~+0.024 per exp/03's accept delta. If exp/11 rejects marginally (val loss in the 2.984–3.020 range), a follow-up control at `weight_tied=False, u_net_skips=False` would separate the two contributions; this is flagged as exp/13 in the plan but not pre-committed.

## Why it might (or might not) improve

- **References:**
  - Mixture of Recursions / Looped transformers: [arXiv 2507.10524](https://arxiv.org/abs/2507.10524).
  - Recurrent Depth Transformer (Geiping et al. 2025): [arXiv 2502.05171](https://arxiv.org/abs/2502.05171) — latent reasoning via iterated shared block.
  - ALBERT (Lan et al. 2019): full cross-layer parameter sharing; ~12× fewer params at comparable quality on NLU tasks at medium scale.
- **Mechanism.** A single block applied K times forms K sequential residual updates from the same learned transform. Classical argument: fixed-point / deep-equilibrium models show that iterating a shared nonlinearity can compose more complex functions than its own depth would suggest, if the transform is a contraction-ish map. In practice, pure-tying on pretraining LMs has been found to lose quality at matched tokens (ALBERT matched at NLU but lagged on generative perplexity). At 124 M with 10 B tokens, expect capacity loss to dominate.
- **Why this could fail hard.** The model loses 63 % of its weights. Chinchilla-scaling extrapolation for a 45 M model trained to 10 B tokens suggests val loss ≈ 3.15–3.25, *before* adding any tying-dynamics penalty. This experiment asks "does 12-iter sharing of a 7 M block reach closer to a 45 M untied baseline than to a 124 M untied baseline?"
- **Predicted val loss Δ @ 10 B tokens:** **+0.05 to +0.15**. Point prediction: **+0.09** (target val ≈ 3.054). Outside the accept band; inside the "informative reject" band.
- **Predicted tok/s Δ:** **−1 % to +2 %**. Same per-step FLOPs (12 block applications either way); slightly smaller activations-and-weights footprint, possibly marginal throughput gain. Compile cache and kernel-launch overhead unchanged.
- **HellaSwag Δ:** **−2 to +0.5 pp**. At reduced capacity HellaSwag typically drops ~1–2 pp per 50 % param cut.

## Accept criteria

- **Strong accept** (val_loss ≤ **2.984**, tok/s regression ≤ 10 %): would flip the prior that tying costs real quality at this scale. Would motivate a proper MoR-style follow-up (exp/12).
- **Informative reject** (**2.984 < val_loss ≤ 3.050**, tok/s OK): pure tying keeps the model viable but clearly costs quality. Worth documenting but not promoting. MLA + LoopLLM stacking (exp/12) becomes dubious — if LoopLLM alone costs 0.05 and MLA alone costs 0.016, stacking would be in the +0.06–0.08 range, well past accept.
- **Hard reject** (val_loss > **3.100** OR tok/s regression > 15 %): pure tying at this scale is a dead end for fixed-tokens training. Skip exp/12.
- **Kill-early** (step ≈ 2 000, 1 B tokens): stop if val loss > **3.60** (baseline step-2000 was 3.4729; 0.13 slack for the capacity-shortfall warmup transient).

## Implementation notes

- **Single-Block storage.** `transformer.h` is an `nn.ModuleList` of length 1, holding the shared `Block`. Parameter keys become `transformer.h.0.attn.*` etc. — matching the untied naming at position 0. The existing zero-init rule (suffix match on `.c_proj.weight` / `.w_down.weight`) hits exactly one weight of each kind under tying.
- **Residual-init scaling (non-zero-init path).** The Radford `1/sqrt(2*n_layer)` scale is *still* correct under tying: the residual stream receives `n_layer` updates, and the scale ensures the cumulative variance stays sensible. The fact that all updates go through the *same* transform doesn't change the residual-stream analysis.
- **Optimizer / LR schedule.** No changes. AdamW's single parameter group covers the shared block weights — each gradient update accumulates `n_layer` sets of gradients (one from each iteration's backward path) into the same parameter, effectively giving Muon-style "concentrated" gradient signal per weight.
- **torch.compile.** Unchanged — the forward-loop refactor is a normal Python `for _ in range(...)` over an `nn.Module.__call__`. Dynamo traces it the same as the untied path.
- **Checkpoint sizes.** Will be roughly `0.37 × v0.3 size` = ~0.53 GB per `.pt` file (vs v0.3's 1.42 GB). Rolling checkpoints (`step_16000.pt`, `step_18000.pt`) remain enabled.
- **HF weight-load test.** Unaffected — `weight_tied` defaults to `False`; the faithful GPT-2 config and `tests/test_hf_weight_load.py` are unchanged.

## Result

| metric                           | baseline (v0.3) | exp/11 LoopLLM | Δ |
|----------------------------------|----------------:|---------------:|---:|
| val loss @ 1 B tokens            | 3.4729          |                |   |
| val loss @ 5 B tokens            | 3.0780          |                |   |
| val loss @ 10 B tokens (best)    | 2.9641          |                |   |
| val loss (200-batch held-out)    | 2.9694          |                |   |
| HellaSwag acc (1 000 examples)   | 0.3780          | *optional*     |   |
| tokens / s median                | 178 217         |                |   |
| wall-clock 1 epoch               | 15 h 39 min     |                |   |
| model params                     | 123.7 M         | 45.7 M         | **−63 %** |
| time to val loss 3.50            | 98 min          |                |   |
| time to val loss 3.10            | 418 min         |                |   |

- **Seeds:** single seed (0). If the result lands inside the 2.984–3.050 "informative reject" band, add seeds 1 & 2 before finalising.
- **Loss curves:** attach `report_assets/loss_curve.png` on completion.

## Verdict

**TBD** — fill in on run completion.
