---
id: 06-muon-mup
status: in-progress
baseline_run: runs/03-modded-tricks/
experiment_run: runs/06-muon-mup/
baseline_tag: v0.3-exp03
date: 2026-04-21
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

| metric                          | baseline (v0.3) | exp/06 | Δ |
|---------------------------------|----------------:|-------:|---:|
| val loss @ 1 B tokens           | 3.4729          |        |   |
| val loss @ 5 B tokens           | 3.0780          |        |   |
| val loss @ 10 B tokens (best)   | 2.9641          |        |   |
| val loss (200-batch held-out)   | 2.9694          |        |   |
| HellaSwag acc (1 000 examples)  | 0.3780          |        |   |
| tokens / s median               | 178 215         |        |   |
| wall-clock 1 epoch              | 15 h 39 min     |        |   |
| time to val loss 3.10           | 418 min         |        |   |
| time to val loss 3.00           | 713 min         |        |   |

- **Seeds:** single seed (0) for this first cut. If the signal is within ±0.03 of the accept bar, add seeds 1 & 2.
- **Loss curves:** attach `report_assets/loss_curve.png` on completion.

## Verdict

**TBD** — fill in on run completion.
