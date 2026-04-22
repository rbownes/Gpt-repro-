---
id: 06-muon-mup
status: rejected
baseline_run: runs/03-modded-tricks/
experiment_run: runs/06-muon-mup/
baseline_tag: v0.3-exp03
date: 2026-04-22
author: rjbownes
seeds: [0]
---

# Experiment 06 — MuonAdamW + μP plumbing on v0.3-exp03

## Previous baseline

- **Config:** `configs/gpt2_124m_modernplus.py` @ git tag `v0.3-exp03` (commit `e70512c`).
- **Arch:** modern block (RoPE + RMSNorm + QK-Norm) + modded-nanogpt tricks (ReLU² MLP, zero-init out-projections, U-Net skips, logit_softcap=30).
- **Optimizer:** AdamW (β = 0.9 / 0.95, wd = 0.1, peak LR 6e-4, cosine decay to 0.1×, 715 warmup steps).
- **Data:** FineWeb-Edu-10B (~10 B tokens, 19,073 steps @ 524,288 tok/step, effective batch 512 seqs).
- **Baseline metrics (seed 0):**
  - val loss @ 1 B / 5 B / 10 B tokens: **3.4729 / 3.0780 / 2.9641** (best=**2.9694** on 200-batch held-out)
  - HellaSwag acc (1 000 examples): **0.3780**
  - tokens/s median: **178 215** · wall-clock 1 epoch: **15 h 39 min**
  - time to val loss 3.00: **713 min**

## The change

Two simultaneous additions on top of v0.3, both optimiser-side:

1. **MuonAdamW** replaces AdamW. 2-D block matrices (attn `c_attn`/`c_proj`, MLP `c_fc`/`c_proj`) go through **Muon** (nanochat port: Polar-Express orthogonalisation + NorMuon variance reduction + cautious weight decay, 5 inner iterations). Embeddings (`wte`, tied `lm_head`), norm weights, and all biases stay on **fused AdamW**.
2. **μP plumbing.** `record_base_shapes` / `apply_mup` / `mup_lr_scale` are wired in. At base_width = target_width (124M), every `width_mult = 1.0`, so this is mathematically a no-op in this run — but the machinery is in place for a future 350M / 20M-proxy μTransfer run without any further code change.

Data, token budget, warmup, cosine-decay schedule, grad-accum, effective batch (512 sequences), grad-clip, weight-decay, and all architecture flags are **unchanged** vs. v0.3.

- **Diff:** branch `exp/06-muon-mup` (off `main`, which is at the exp/03-modded-tricks tip, commit `71ae6ad`).
- **Files touched:**
  - `src/gpt_repro/optim_muon.py` (new, ~200 lines)
  - `src/gpt_repro/mup.py` (new, ~130 lines)
  - `src/gpt_repro/optim.py` (refactor: `lr_at_step` → `lr_frac_at_step`, Muon-branch in `build_optimizer`, μP LR scale hook)
  - `src/gpt_repro/train.py` (new `TrainConfig` fields; μP apply step; per-group `base_lr * frac` LR update)
  - `configs/gpt2_124m_modernplus_muon_mup.py` (new)
  - `tests/test_model_shapes.py` (new tests: `test_muon_adamw_overfit_tiny_batch`, `test_muon_param_grouping_shapes`)
  - `tests/test_mup_plumbing.py` (new)
- **Hyperparameters introduced:**
  - `optimizer`: `"adamw"` → `"muon_adamw"`
  - `muon_lr` = 0.02 (nanochat default, per-shape `√(d_major / d_minor)` scaled inside the kernel)
  - `muon_momentum` = 0.95, `muon_ns_steps` = 5, `muon_beta2` = 0.9
  - `use_mup` = True, `mup_base_shapes_path` = None (self-base)
- **Hyperparameters unchanged from v0.3:** `peak_lr=6e-4` (for the AdamW group), `beta1/beta2/eps=(0.9, 0.95, 1e-8)`, `weight_decay=0.1`, `grad_clip=1.0`, `warmup_steps=715`, `min_lr_ratio=0.1`, `total_steps=19_073`.

### Why μP is bundled with Muon in a single experiment (scope note)

The project convention is "one diff, one experiment". We're landing two diffs in this run because μP at base width is provably a no-op (every `width_mult = 1.0` ⇒ every `mup_lr_scale = 1.0`; param-group `base_lr` values are byte-identical to the non-μP path). A dedicated "μP" experiment at 124M would be measuring zero signal. μP pays off when we scale — landing the plumbing now makes the next proxy-sweep or 350M run a config-only change.

The **Muon side** of this diff is the load-bearing change and carries all predicted effect.

## Why it might improve

- **Reference:** [Keller Jordan — Muon blog](https://kellerjordan.github.io/posts/muon/); [Polar-Express Sign Method (Amsel et al. 2024)](https://arxiv.org/pdf/2505.16932); [NorMuon variance reduction (2025)](https://arxiv.org/pdf/2510.05491); [Yang et al. 2022 — μTransfer (Tensor Programs V)](https://arxiv.org/abs/2203.03466).
- **Mechanism.** exp/02 rejected "plain" Muon on top of v0.2 because AdamW caught up by 10 B tokens (Δ val = −0.0007, within noise) — though Muon reached val 3.1 ~14 % faster along the way. v0.3 bundles zero-init out-projections and U-Net skips, both of which change the optimisation landscape: the residual stream starts as the identity, and cross-depth skips make early layers' representations directly visible to late layers. Muon's orthogonalised updates should carry further through this altered landscape — the exp/03 report itself calls this follow-up out: "Revisit Muon on top of v0.3 — AdamW's catch-up may no longer happen." Additionally the nanochat port is richer than exp/02's Muon: Polar Express (faster convergence than Newton-Schulz at 5 steps) plus NorMuon variance reduction (per-neuron adaptive scale).
- **Autoresearch provisional data.** At a 300-second fixed-wall-clock budget with effective batch 32 (different regime), our autoresearch loop measured **−0.050 bpb (val_bpb 1.2729 → 1.2235)** from the exact same MuonAdamW port vs. a best-tuned AdamW. That's the *time-to-target* axis where exp/02 also saw Muon win. The 10 B run measures *end-of-training* quality, which is a harder bar.
- **Predicted effect (written BEFORE running):**
  - val loss Δ @ 10 B tokens: **−0.010 to −0.030** (point: **−0.020** → target val ≈ 2.944). Smaller than exp/03's −0.024 because exp/02's null result at 10 B tokens is the prior; the zero-init + U-Net architectural change is the only reason to expect it to flip.
  - tok/s Δ: **−3 % to −8 %** (autoresearch measured −6 %). Polar Express adds 5 inner matmuls per Muon group per optim step; NorMuon adds a reduction. Offset by fused AdamW on non-matrix params.
  - HellaSwag Δ: **±1.5 pp** (noise floor of 1 000-sample eval). No strong prior.
- **Accept criteria:** val loss @ 10 B ≤ **2.944** (Δ ≤ −0.020) **AND** tok/s regression ≤ **10 %** (≥ 160 k tok/s).
- **Reject:** val loss Δ < 0.005 (Muon ties AdamW at 10 B again), **or** tok/s regression > 15 %, **or** training instability (loss spikes, NaN, grad-norm divergence absent from v0.3).
- **Kill-early** (step ≈ 2 000, 1 B tokens): stop if val loss > **3.52** (baseline + 0.05) or if first-eval (step 500) val_loss exceeds `runs/03-modded-tricks/metrics.jsonl`'s step-500 value by more than **0.10** — likely indicates an optimiser-port bug.

## Implementation notes

- **MuonAdamW port source.** `src/gpt_repro/optim_muon.py` is a line-for-line port of `autoresearch/train.py` lines 396–578 (MuonAdamW + fused kernels + Polar-Express coefficients), adapted to read hyperparameters from `TrainConfig` fields instead of module-level constants.
- **Parameter grouping.** `wte` (tied with `lm_head`), `wpe` (if present), all 1-D tensors (RMSNorm weights, biases, QK-Norm weights) → single AdamW group (98 params on v0.3 arch). All 2-D block matrices → one Muon group per unique shape. On v0.3 modernplus (12 layers, n_embd=768, ReLU² MLP) this produces 4 Muon groups: `(768,768)`, `(768,3072)`, `(2304,768)`, `(3072,768)`, each with 12 members. Muon's fused kernel stacks members along a new leading axis, so all members of a group must share a shape.
- **Per-shape LR scaling.** Muon applies a `max(1, d_major/d_minor)**0.5` LR boost inside `_muon_step_fused`. This is the nanochat convention and matches the Polar-Express analysis. No separate per-group LR override in the config.
- **Unified LR schedule.** `optim.lr_frac_at_step(step, …) → float ∈ [0, 1]` is the shared schedule; each group writes `group["lr"] = group["base_lr"] * frac` every step. The AdamW group's `base_lr` = `cfg.peak_lr` (= 6e-4); each Muon group's `base_lr` = `cfg.muon_lr` (= 0.02). This keeps warmup + cosine decay shape identical to v0.3 while Muon and AdamW operate at their respective absolute scales.
- **μP wiring.** `apply_mup(model, base_shapes)` runs between model construction and optimiser build. At `mup_base_shapes_path=None`, base shapes are recorded from the current model itself, guaranteeing every `mup_width_mult = 1.0` ⇒ every `mup_lr_scale = 1.0`. The optimiser builder multiplies `base_lr` by the group-mean `mup_lr_scale`, which equals 1.0 here; numerically identical to `use_mup=False`.
- **compile.** `cfg.compile_mode = "default"`. We do **not** use `"reduce-overhead"` (CUDA graphs) in this run because U-Net skips share the `wte` output across blocks and broke CUDA-graph output tracking in autoresearch (see `autoresearch/results.tsv` row `COMPILE_MODE max-autotune`). The Muon fused kernel is itself `@torch.compile(dynamic=False, fullgraph=True)` — that compilation is scoped to the optimiser step and does not interact with the forward-graph shared-tensor problem.
- **0-D CPU hyperparam tensors.** Muon and AdamW fused kernels take LR / beta / wd as pre-allocated 0-D CPU tensors (`torch.tensor(0.0)`) whose values get `.fill_()`-ed each step. This lets the schedule change LR without invalidating the compiled graph.
- **Checkpointing.** `MuonAdamW.state_dict()` and `load_state_dict` go through PyTorch's standard `torch.optim.Optimizer` machinery, which handles the per-parameter `momentum_buffer` and `second_momentum_buffer` tensors without special code. Resume should work; will be verified on the first `ckpt_every=2000` save.
- **HF weight-load test.** Unaffected — this experiment is optimiser-only, architecture is byte-identical to v0.3. `tests/test_hf_weight_load.py` continues to exercise only the faithful GPT-2 config.

## Result

| metric                           | baseline (v0.3) | exp/06   | Δ |
|----------------------------------|----------------:|---------:|---:|
| val loss @ 1 B tokens (step 2 000)   | 3.4729          | **3.4205** | **−0.0523** |
| val loss @ 5 B tokens (step 9 500)   | **3.0780**      | 3.1677     | +0.0897 |
| val loss @ 10 B tokens (best val)    | **2.9641**      | 2.9737     | +0.0096 |
| val loss @ 10 B tokens (final step)  | **2.9642**      | 2.9742     | +0.0101 |
| HellaSwag acc (1 000 examples)       | 0.3780          | *not run*  | — |
| tokens / s median                    | 178 217         | 177 902    | −315 (**−0.18 %**) |
| wall-clock 1 epoch                   | 15 h 39 min     | 15 h 40 min | +1 min |
| time to val loss 3.500               | 98.4 min        | **73.9 min** | **1.33× faster** |
| time to val loss 3.200               | **270 min**     | 418 min    | 0.65× |
| time to val loss 3.100               | **418 min**     | 615 min    | 0.68× |
| time to val loss 3.000               | **713 min**     | 838 min    | 0.85× |
| time to val loss 2.975               | **836 min**     | 936 min    | 0.89× |

- **Seeds:** single seed (0). Result is far from the accept bar (Δ must be ≤ −0.020; actual Δ = +0.010) so no additional seeds warranted — the effect direction is unambiguous even under noise.
- **HellaSwag:** the paper-battery eval wasn't wired into this run. Can be computed post-hoc from `runs/06-muon-mup/best_val.pt` if needed; skipping for now because the val-loss result already decides the verdict.

### Three-phase trajectory — the core finding

The Δ between exp/06 and exp/03 is **not monotone**; it's a three-phase curve, which is the essential pattern this experiment documents:

| phase             | steps        | tokens        | behaviour |
|-------------------|-------------:|--------------:|-----------|
| Muon dominant     | 500 – 2 500  | 0.26 – 1.31 B | Peak Muon lead of **−0.644** at step 500. At 1 B tokens, Muon is 0.052 ahead. |
| AdamW overtakes   | 2 500 – 8 500 | 1.31 – 4.46 B | Crossover at step 3 000 (1.57 B). Gap grows to **+0.092** at step 8 500 — Muon's worst point. |
| Muon catches up   | 8 500 – 19 073 | 4.46 – 10.0 B | Gap closes linearly; exp/06 ends at Δ = +0.010 (essentially tied, same shape as exp/02 at exp/02's scale). |

Practically, Muon wins the first ~1.5 B tokens of training by a **big** margin (1.33× faster to val loss 3.5), loses the middle ~3 B tokens, and nearly ties by the end.

### Throughput

tok/s regression is **negligible** (−0.18 %, 315 tok/s on a base of 178 k). Prediction was −3 % to −8 %; autoresearch had measured −6 % at batch=32. At the v0.3 effective batch of 512 seqs, Polar Express and NorMuon overhead is dwarfed by the 32 forward/backward passes in grad accumulation. This invalidates the cost argument against Muon on this hardware — Muon is essentially free on tok/s.

### μP

As designed, a no-op at base width. `apply_mup` reported `self-base (146 params); LR scaling is a no-op at base width`. Every `mup_width_mult = 1.0`, every `mup_lr_scale = 1.0`. The machinery is exercised in production code path and ready for the first non-base-width run (350 M, or a 20 M proxy for exp/08).

## Verdict

**Reject** on the pre-declared primary axis: val loss Δ @ 10 B = **+0.010**, vs the accept bar of **≤ −0.020** and the reject floor of **< 0.005**. This is the same qualitative outcome as exp/02 (plain Muon on v0.2, Δ = −0.0007), but measured against a stronger baseline (v0.3 with zero-init + U-Net + softcap): the zero-init + U-Net claim from the exp/03 follow-up section *did not* alter the Muon-vs-AdamW crossover behaviour.

**Keep the code, not the config.** `src/gpt_repro/optim_muon.py` and `src/gpt_repro/mup.py` stay on main; the config `configs/gpt2_124m_modernplus_muon_mup.py` is preserved for reproducibility but is **not** promoted to the new baseline. `v0.3-exp03` remains the baseline tag.

## Recommendations / follow-ups

1. **Do NOT re-run a vanilla Muon v4 on v0.3 — the answer is settled.** Two consecutive experiments (exp/02, exp/06) on two different architectures produced the same result: Muon ≈ AdamW at 10 B tokens, Muon ahead at ≤ 2 B tokens. The hypothesis that "v0.3's zero-init + U-Net shifts the landscape enough for Muon to pull ahead" is falsified.

2. **exp/07 — retuned Muon schedule.** The closure rate between steps 9 000 and 19 000 is suspiciously linear at ~0.0025 / 500 steps. Two plausible fixes to turn Muon into a net win:
   - **Lower `muon_lr`** from 0.02 toward 0.01. The peak lead at step 500 suggests Muon overshoots at current LR; less overshoot → less mid-training correction → smaller AdamW lead in steps 3 000 – 8 500.
   - **Warmdown that decays Muon harder than AdamW.** nanochat's recipe uses a trapezoid + linear warmdown (65 % of total steps) rather than full cosine; this biases the last third of training toward small, stable updates where Muon's orthogonalisation bias matters less. Either per-group cosine parameters (Muon decays to `min_lr_ratio_muon=0.05`, AdamW to `min_lr_ratio_adamw=0.2`) or a scheduled transition.
   - Both are config-only; no new code.

3. **This result is a clear argument FOR short-run regimes.** At 1 B tokens, Muon wins by 0.052 val loss with no tok/s cost. If the project ever needs a "cheap development loop" (early architecture comparison, ablations), Muon is the right optimiser. **For future experiments that plan to train < 2 B tokens — always prefer Muon; for ≥ 5 B tokens — stick with AdamW until exp/07 retunes the schedule.**

4. **The 0.18 % tok/s cost is the headline non-result.** The autoresearch loop's −6 % measurement at batch=32 doesn't transfer to production batch=512. This is useful for anyone considering Muon on this hardware: cost is effectively zero at large effective batches, the only axis to evaluate on is quality.

5. **μP infrastructure is ready.** Next experiment (recommend exp/08) should be the 20 M proxy HP sweep + transfer back to 124 M, which is the actual play that μP enables. Current hand-tuned HPs on 124 M are inherited from exp/00, unlikely to be optimal.
