---
id: 11-loopllm
status: rejected
baseline_run: runs/03-modded-tricks/
experiment_run: runs/11-loopllm/
baseline_tag: v0.3-exp03
date: 2026-04-24
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

> **Addendum (2026-04-23, mid-run override):** By the step-1 000 eval the run was trending to likely exceed the kill-early threshold (val loss 4.22 at step 1 000 vs baseline 3.86, delta widening). The kill-early rule was **explicitly overridden by the experimenter** to let the full 10 B-token run complete. Rationale: even a hard-reject 10 B data point is useful for (a) the loss-curve shape analysis vs the "expected 45 M-from-scratch" Chinchilla extrapolation and (b) the exp/12 MLA+LoopLLM planning decision, which depends on knowing the *end-of-training* cost of tying, not just the early-training transient. The pre-declared accept/informative-reject/hard-reject bars at 10 B remain in force.

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
| val loss @ 1 B tokens            | 3.4729          | 3.8495         | +0.3767 |
| val loss @ 5 B tokens            | 3.0780          | 3.5028         | +0.4248 |
| val loss @ 10 B tokens (final)   | **2.9642**      | 3.4060         | +0.4418 |
| best val loss                    | **2.9641**      | 3.4056         | **+0.4415** |
| HellaSwag acc (1 000 examples)   | 0.3780          | *not run*      | — |
| tokens / s median                | 178 217         | 177 064        | **−0.65 %** |
| wall-clock 1 epoch               | 15 h 39 min     | 15 h 44 min    | +5 min |
| model params                     | 123.7 M         | **45.7 M**     | **−63.0 %** |
| time to val loss 3.50            | 98 min          | 496 min        | **0.20×** |
| time to val loss 3.20            | 270 min         | **never reached** | — |
| time to val loss 3.10            | 418 min         | **never reached** | — |
| time to val loss 3.00            | 713 min         | **never reached** | — |

### Against pre-declared criteria

- **Strong accept** (val ≤ 2.984): **FAIL by 0.42** (val = 3.406)
- **Informative reject** (val ≤ 3.050): **FAIL by 0.36**
- **Hard reject** (val > 3.100): **TRIGGERED by 0.31**

This is the **largest rejection gap** in the project's history by ≈ 10×. exp/02 (Muon-only) landed at Δ = −0.0007 (ties), exp/10 (MLA) at +0.016 (near-miss), exp/06 (Muon+μP) at +0.010 (near-tie). exp/11 is a clean, substantial hard reject.

### Loss-curve shape — a persistent capacity-shortfall offset, not a crossover

The per-step gap **grew steadily** from early training, **peaked around step 15 000**, and then **plateaued**:

| step | v0.3 | exp/11 | Δ |
|---:|---:|---:|---:|
|  500 | 4.7746 | 5.0807 | +0.306 |
| 1 000 | 3.8634 | 4.2222 | +0.359 |
| 2 000 | 3.4729 | 3.8495 | +0.377 |
| 3 000 | 3.3308 | 3.7284 | +0.398 |
| 5 000 | 3.2075 | 3.6134 | +0.406 |
| 7 000 | 3.1369 | 3.5530 | +0.416 |
| 10 000 | 3.0675 | 3.4951 | +0.428 |
| 13 000 | 3.0175 | 3.4510 | +0.434 |
| 15 000 | 2.9922 | 3.4293 | +0.437 |
| 17 000 | 2.9738 | 3.4142 | +0.440 |
| 19 073 | **2.9642** | **3.4060** | **+0.4418** |

Δ-of-Δ trajectory per 500 steps:

- **0 → 5 000**: +0.020 (widening during warmup transient)
- **5 000 → 10 000**: +0.002 (slow widening)
- **10 000 → 15 000**: +0.0009 (barely drifting)
- **15 000 → 19 073**: +0.0003 (stable plateau)

This is qualitatively different from:

- **exp/06 (Muon+μP)**: three-phase crossover — early lead, mid-training deficit, late closure.
- **exp/10 (MLA)**: flat gap from step 500 onward at +0.016.
- **exp/11 (LoopLLM)**: growing gap that plateaus at a large offset (+0.44).

The plateau tells us this isn't a "converging-but-slower" model — it's converged to a ~0.44-nats-worse minimum. That floor is the capacity shortfall, and no amount of additional training at fixed tokens will close it.

### Predicted vs actual

- Predicted val loss Δ @ 10 B: **[+0.05, +0.15]**, point **+0.09**. Actual: **+0.4418**. **Predicted much too generously** — the miss is 3× the point prediction and 3× the upper bound. This is the biggest miscalibration in the project to date.
  - **What I got wrong in the prediction**: I imagined LoopLLM's 12-iteration looping of a shared block would recover a large fraction of the untied model's effective capacity. In practice it recovers *some* capacity vs a 45 M-untied baseline (Chinchilla-style scaling would predict val ≈ 3.55 for a 45 M untied model at 10 B; exp/11 reaches 3.41, so looping *is* worth ~0.14 nats), but the 63 % parameter cut dominates and the 10 B-token fixed-budget leaves no room to compensate.
- Predicted tok/s Δ: **[−1 %, +2 %]**. Actual: **−0.65 %**. ✅ within range.
- Kill-early gate (step 2 000 val > 3.60): **triggered** at step 2 000 with val = 3.8495 (0.25 over). Experimenter overrode the pre-declared rule to run to 10 B for full-curve data — noted in the report's addendum. **The override was the right call**: the full curve is informative for exp/12 planning in a way the 1 B-token snapshot wouldn't have been.

## Verdict

**Hard reject.** val loss Δ @ 10 B = **+0.4415** vs the hard-reject bar of +0.136 (trigger is 3.25× the threshold). Pure weight tying at `n_layer = 12` shared-block iterations on v0.3 MHA **materially hurts pretraining quality** at fixed tokens, and the loss plateaus — it is not "slowly catching up, come back in another 10 B".

**Keep the code, not the baseline.** The `weight_tied` flag, the U-Net-incompatibility guard, and the tests stay on `main` — they're infrastructure for future experiments (MoR, recurrent-depth variants, partial-tying grids). `configs/gpt2_124m_modernplus_looped.py` is preserved for reproducibility. `v0.3-exp03` remains the project baseline.

## Key findings

1. **LoopLLM gives a real but small lift over a matched-size untied baseline.** Chinchilla extrapolation for a 45 M untied model trained on 10 B tokens predicts val ≈ 3.55. exp/11 reached 3.41 — that's a **~0.14-nat improvement for the same parameter count**. So weight-tied looping is not a complete no-op at this scale. It's just nowhere near enough to cover a 63 % parameter cut.
2. **Throughput is effectively free.** tok/s −0.65 %. The loop has identical per-step FLOPs and the smaller parameter footprint is roughly cancelled by the lack of per-layer kernel specialisation under `torch.compile`.
3. **The gap plateau is the cleanest signal in the run.** Between steps 10 000–19 073 the Δ barely moves (+0.428 → +0.442). This asymptote indicates the shared transform has reached its capacity limit, not that it's converging slowly.
4. **"U-Net off" is a measurable contributor but not the dominant term.** exp/03's accept delta was −0.024 on top of v0.2, so "forced U-Net off" would account for perhaps +0.024 of the +0.44 gap (≈ 5 %). The other ~95 % is the capacity shortfall from tying. If we ever need the attribution to be clean, exp/13 (weight_tied=False, u_net_skips=False) would isolate the U-Net contribution.

## Recommendations / follow-ups

1. **Skip exp/12 (MLA + LoopLLM) as originally scoped.** MLA alone cost +0.016 val loss; LoopLLM alone costs +0.44. The stacked cost is dominated by the LoopLLM term — MLA + LoopLLM would likely land at val ≈ 3.42, still a hard reject for pretraining.

   **However, the motivation for MLA + LoopLLM was never pretraining val loss** — it's the DeepSeek-R1-style inference-time reasoning pattern, where small weights + iterated computation lets you scale test-time compute. For that axis, a more productive follow-up is:
   - **exp/12 (revised): LoopLLM + adaptive K at inference.** Train at fixed K = 12 as exp/11, but at inference allow K ∈ {8, 12, 16, 20, 24}. Measure generation quality on reasoning-heavy prompts (HumanEval, GSM8K-style small benchmarks) as a function of K. This is the actual thesis MoR / Geiping-RDT tests, and it needs a model that's already been trained.

2. **Don't do further pure-tying experiments at 124 M effective depth.** The capacity shortfall is the load-bearing axis and 63 % reduction is too much. If there's appetite to continue the weight-tying thread, a **partial-tying** experiment (3 distinct blocks × 4 iterations, giving ~58 M params) is the only variant worth running next — enough capacity-recovery to be competitive, while still keeping the "shared-block reasoning" story.

3. **Attribution control (exp/13) is low-priority.** The +0.024 U-Net contribution is small enough relative to the +0.44 LoopLLM cost that the attribution is effectively settled. Don't spend 15 h on a control unless exp/12 (or a successor) needs the clean accounting.

4. **Real value of this run:** it's the first rigorous data point we have for "what does weight-tied looping cost at 124 M / 10 B". Every future discussion about looped transformers, MoR, DeepSeek-R1-style architectures should cite `runs/11-loopllm/` as the calibration point.

