---
id: 03-modded-tricks
status: accepted
baseline_run: runs/01-modern-block/
experiment_run: runs/03-modded-tricks/
baseline_tag: v0.2-exp01
date: 2026-04-20
author: rjbownes
seeds: [0]
---

# Experiment 03 — modded-nanogpt tricks (ReLU² MLP + zero-init projections + U-Net skips + logit softcap)

## Previous baseline

- **Config:** `configs/gpt2_124m_modern.py` @ commit `a8141ed`, git tag `v0.2-exp01`.
- **Arch:** modern block — RoPE + RMSNorm + SwiGLU + QK-Norm, 124 M params.
- **Optimizer:** AdamW on all params (β = 0.9 / 0.95, wd = 0.1, peak LR 6 e-4).
- **Data:** FineWeb-Edu-10B, 19 073 steps @ 524 288 tok/step.
- **Baseline metrics:**
  - val loss @ 1 B / 5 B / 10 B tokens: **3.4641 / 3.1009 / 2.9884**
  - HellaSwag (1 000 val examples): **0.3830**
  - tokens/s median: **181 732**
  - wall-clock 1 epoch: **15 h 20 min**

### Context: exp/02 was rejected

Exp/02 (Muon on hidden matmuls) did not clear the val-loss accept bar in
isolation on a 10 B-token cosine-decay run — it tied AdamW at 2.988 while
getting there ~14 – 25 % faster. We noted in that report that modded-nanogpt
bundles Muon *with* the tricks in this exp. Here we deliberately test the
*tricks only* on top of AdamW, so each ingredient's marginal contribution
is attributable separately from Muon's.

## The change

Swap four pieces of the block, all activated via new `GPTConfig` flags:

- **SwiGLU → ReLU² MLP.** Two-matrix MLP, hidden = 4 · d = 3 072,
  activation `relu(x)²`. Same parameter count as SwiGLU with `mlp_hidden=2048`
  and as the faithful GELU MLP (all three paths are 4.72 M params per layer).
- **Zero-init out-projections.** Attention's `c_proj` and MLP's
  `c_proj` (ReLU²) are initialised to zero. The GPT-2 residual-scale init
  (`std = 0.02 / √(2·n_layer)`) is replaced by literal zeros. Block outputs
  start at zero so the residual stream is "identity" at step 0.
- **U-Net skip connections.** Layer indices `i ∈ [n/2, n-1]` receive an
  additive skip from layer `n-1-i`'s post-block output. For `n_layer = 12`:
  layer 6 gets layer 5's output, layer 7 gets layer 4's, …, layer 11 gets
  layer 0's.
- **Logit softcap.** Final logits clipped via `s · tanh(logits / s)` with
  `s = 30.0`. Prevents extreme logit magnitudes that can destabilise early
  training. Matches Gemma-2 and modded-nanogpt conventions.

Data, optimizer, schedule, token budget, hardware — all identical to v0.2.
Only the arch flags change.

- **Diff:** branch `exp/03-modded-tricks`. See pending implementation commit.
- **Files touched:** `src/gpt_repro/model.py`, `configs/gpt2_124m_modernplus.py`
  (new), `tests/test_model_shapes.py`.
- **Hyperparameters introduced / changed:**
  - `mlp_type`: `"swiglu"` → `"relu2"`
  - `mlp_hidden`: `2048` → `None` (auto = 4·d = 3072 for ReLU²)
  - `zero_init_proj`: `false` → `true`
  - `u_net_skips`: `false` → `true`
  - `logit_softcap`: `None` → `30.0`

## Why it might improve

All four are pieces of the current single-GPU nanoGPT speedrun record
([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), multiple
community updates 2024 – 2026) and each has independent prior art:

- **ReLU²** ([So et al. 2021](https://arxiv.org/abs/2109.08668) "Primer";
  also in PaLM) — empirically small perplexity win vs GELU/SwiGLU at small
  scale, and slightly cheaper because no gate pre-multiply.
- **Zero-init out-projections** (community folklore, formalised in
  modded-nanogpt and [ReZero](https://arxiv.org/abs/2003.04887)) — residual
  stream is literally the embedding at step 0, so gradients flow cleanly
  through the network before any block contributes. Stabiliser that also
  tends to produce a small final-loss win.
- **U-Net skip connections** (modded-nanogpt / [Transformer-XL-like shortcuts](https://arxiv.org/abs/2304.08467))
  — cross-depth information flow lets early layers' representations
  directly inform late layers, easing optimisation and giving a small loss
  improvement at fixed depth.
- **Logit softcap** (Gemma-2; modded-nanogpt) — prevents the final layer
  from emitting logits with extreme magnitudes that the AdamW step can
  amplify; mostly a stability guarantee, with incidental small loss wins.

### Mechanism

Jointly these are a "well-behaved block and well-behaved head" package.
The residual stream starts as a clean identity (zero-init projections);
information can tunnel across half the depth (U-Net skips); the
activation is slightly cleaner than SwiGLU (ReLU²); the output head can't
blow up (softcap). None of the four is individually expected to move the
needle by more than 0.01 – 0.02 val loss; together I expect
**~0.02 – 0.05** total.

### Predicted effect (written BEFORE running)

- val loss Δ @ 10 B tokens: **−0.02 to −0.05**. Point prediction **−0.03**
  (target val ~2.958). Smaller than exp/01's modernization step because
  that did the load-bearing work; these tricks are polish.
- tokens/s Δ: **−3 % to +2 %**. U-Net skips add 6 small adds per step,
  softcap adds a tanh on the logits, ReLU² is slightly cheaper than SwiGLU
  (two matmuls instead of three at same total params). Expected near-neutral.
- HellaSwag Δ: **+0.0 to +1.5 pp**.

### Accept criteria

- **Accept if** val loss @ 10 B tokens drops by **≥ 0.02** (i.e. ≤ 2.968)
  **and** tok/s regression is **≤ 5 %** (≥ 172.6 k tok/s). Both must hold.
- **Reject if** val loss Δ < 0.005, tokens/s regression > 10 %, or
  instability (loss spikes, NaN, divergent gradient norms absent from
  baseline).
- **Kill-early:** at 1 B tokens (step ≈ 2 000), if val loss > 3.52 (baseline
  + 0.05), stop.

## Implementation notes

- **ReLU² MLP:** reuses the faithful 2-matrix MLP class with
  `F.relu(x).pow(2)` replacing `F.gelu(..., approximate="tanh")`. Residual
  projection (`c_proj`) is the one zero-initialised by the new flag.
- **Zero-init projections:** in `GPT.__init__`, after `apply(_init_weights)`
  runs, override weights with suffix `.attn.c_proj.weight` and
  `.mlp.c_proj.weight` / `.mlp.w_down.weight` to zero (not the
  `1/√(2n_layer)` scaled init the faithful baseline uses).
- **U-Net skips:** at model init, record the number of layers. In the
  forward pass, maintain a LIFO stack; for `i < n/2`, push post-block
  output; for `i ≥ n/2`, pop and add to the block's input **before**
  running that block. Only triggered when `cfg.u_net_skips = True`.
- **Logit softcap:** after `lm_head`, apply `softcap · tanh(logits / softcap)`
  iff `cfg.logit_softcap is not None`. Does not change model output path
  when `None`.
- **Faithful default preservation:** all four flags default to their
  v0.1-baseline-faithful values. `tests/test_hf_weight_load.py` must
  continue to pass because `GPT.from_pretrained_gpt2` ignores all new flags.

## Result

| metric                          | baseline (v0.2) | exp/03 | Δ |
|---------------------------------|---------------:|-------:|---:|
| val loss @ 1 B tokens           | 3.4641         | 3.4729   | +0.009 (early slowdown from zero-init) |
| val loss @ 5 B tokens           | 3.1009         | **3.0780** | **−0.023** |
| val loss @ 10 B tokens (best)   | 2.9884         | **2.9641** | **−0.024** |
| val loss (200-batch held-out)   | 2.9946         | **2.9694** | **−0.025** |
| HellaSwag acc (1 000 examples)  | 0.3830         | 0.3780   | −0.5 pp (outside predicted range) |
| tokens / s median               | 181 732        | 178 215  | **−1.9 %** |
| tokens / s mean                 | 181 240        | 177 594  | −2.0 % |
| wall-clock 1 epoch              | 15 h 20 min    | 15 h 39 min | +19 min |
| time to val loss 3.5            | 96.6 min       | 98.4 min | +2 min   (tie) |
| time to val loss 3.2            | 290 min        | **270 min**  | **−7 %** |
| time to val loss 3.10           | 483 min        | **418 min**  | **−13 %** |
| time to val loss 3.00           | 883 min (≈final) | **713 min**  | **−19 %** |
| time to val loss ≈ 2.988 (v0.2's final) | N/A | **762 min** | reaches v0.2 quality in **83 %** of v0.2 wall-clock |

### Predicted vs actual

- Predicted val loss Δ: **−0.02 to −0.05** (point **−0.03**). Actual: **−0.024**. In range, below the point prediction. ✅
- Predicted tok/s Δ: **−3 % to +2 %**. Actual: **−1.9 %**. In range. ✅
- Predicted HellaSwag Δ: **+0.0 to +1.5 pp**. Actual: **−0.5 pp**. **Miss** — slight regression rather than the small gain I expected. Still within the ~1.5 pp noise floor of a 1 000-sample eval, but I wouldn't paper over it: the bundle trades a tiny bit of HellaSwag multiple-choice skill for val-loss improvement. Candidate culprit: logit softcap at `s=30` compresses the spread of log-probs, which could make per-choice log-likelihoods closer together and dampen the confidence signal the metric relies on. An ablation (softcap vs no softcap) would tell us for sure; leaving that to follow-up.

### Loss-curve shape

Zero-init projections cost ~0.2 val loss at step 500 (4.77 vs 4.59 for v0.2)
because the residual stream is literally the token embedding for the first
few hundred steps — every block is initially a no-op. By step 2 500 exp/03
has caught up; by step 3 000 it's ahead; the gap holds steady at **−0.02**
for the rest of training. Consistent with "zero-init buys a cleaner
optimisation path that pays off by mid-training and keeps paying".

## Verdict

**Accept.** Both primary pre-declared criteria clear:

- val loss @ 10 B Δ: **−0.024** (required ≥ −0.02) ✅
- tok/s regression: **1.9 %** (required ≤ 5 %) ✅
- no training instability; smooth cosine decay; zero NaN / spike events.
- HellaSwag: −0.5 pp (secondary, just below the pre-declared range, within noise).

**Advance `v0.3-exp03` tag** at this commit. Future experiments fork from
**`v0.3-exp03`** and treat modern block + modded tricks as the new baseline.

Configs preserved at `configs/gpt2_124m.py` (faithful),
`configs/gpt2_124m_modern.py` (v0.2), and
`configs/gpt2_124m_modernplus.py` (v0.3). All three remain valid, run-to-completion
recipes; new experiments fork the last one.

### Follow-up candidates

- **Ablate the logit softcap.** If it's responsible for the HellaSwag dip,
  dropping it gives free HellaSwag accuracy at no loss cost.
- **Revisit Muon on top of v0.3** (exp/04). Exp/02 rejected Muon on top of
  v0.2 because AdamW caught up by 10 B tokens. On top of v0.3, zero-init +
  U-Net skips may change the optimisation landscape enough that Muon's
  orthogonal updates carry further — worth re-running.
- **FP8 matmul via TransformerEngine (roadmap #04)** is now the most valuable
  single-axis experiment for tok/s; v0.3 gave +2.4 % combined modern-block
  quality at −2 % tok/s, and an FP8 boost could put throughput back above
  the original faithful baseline.

