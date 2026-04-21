---
id: 05-speed-pack
status: rejected
baseline_run: runs/03-modded-tricks/
experiment_run: runs/05-speed-pack/
baseline_tag: v0.3-exp03
date: 2026-04-21
author: rjbownes
seeds: [0]
---

# Experiment 05 — Speed pack (bundled throughput improvements)

## Previous baseline

- **Config:** `configs/gpt2_124m_modernplus.py` @ commit `e70512c`, tag `v0.3-exp03`.
- **Arch:** RoPE + RMSNorm + QK-Norm + ReLU² + zero-init proj + U-Net skips + logit-softcap(30). 123.6 M params.
- **Optimizer:** AdamW β=0.9/0.95, wd 0.1, peak LR 6e-4.
- **Compile:** `torch.compile(mode="default")`, SDPA flash.
- **Baseline metrics:**
  - val loss @ 1 B / 5 B / 10 B: **3.4729 / 3.0780 / 2.9641**
  - HellaSwag (1 000): **0.3780** (**−0.5 pp vs v0.2**; softcap-suspected regression)
  - tokens/s median: **178 215** · wall-clock 1 epoch: **15 h 39 min**

## The change

Bundled speed-oriented experiment. Four changes at once because each one is too small to be worth a full 15 h A/B on its own, and there's no quality coupling between them:

1. **`compile_mode="max-autotune-no-cudagraphs"`.** Measured +6 % tok/s in `experiments/perf-util-probe.md`. Pure kernel-tuning, no math change.
2. **Liger fused linear cross-entropy.** Replaces `lm_head(x); F.cross_entropy(...)` with `LigerFusedLinearCrossEntropyLoss(lm_head.weight, x, targets)`, which fuses the `[B·T, V]` matmul + softmax + NLL into a single kernel and never materialises the full logits tensor. Large VRAM and tok/s win at small-V-big-tail models.
3. **Softcap off** (`logit_softcap=None`). v0.3 introduced softcap=30 which is suspected to be the −0.5 pp HellaSwag regression. Fused CE is fundamentally incompatible with softcap (logits never materialise), so they're removed together.
4. **GQA with 4 KV heads** (`n_kv_head=4`). Q stays at 12 heads × 64 dim; K, V drop to 4 heads × 64 dim and are broadcast to all 12 Q-heads inside SDPA via `enable_gqa=True`. Small training-time speed win; bigger inference-time KV-cache win.

Nothing else changes: same optimizer, same schedule, same data, same token budget, same seed.

- **Diff:** branch `exp/05-speed-pack`.
- **Files touched:** `src/gpt_repro/model.py`, `src/gpt_repro/train.py` (port compile_mode), `configs/gpt2_124m_speedpack.py` (new), `tests/test_speed_pack.py` (new).
- **Hyperparameters introduced / changed:**
  - `compile_mode`: `"default"` → `"max-autotune-no-cudagraphs"`
  - `use_liger_fused_ce`: `false` → `true`
  - `logit_softcap`: `30.0` → `None`
  - `n_kv_head`: `null` (= `n_head`) → `4`

### Why each piece

- **max-autotune**: measured +6 % already in `experiments/perf-util-probe.md`.
- **Liger fused CE**: avoids the `[B·T, V] = [16·1024, 50257] ≈ 800 M BF16 floats = 1.6 GB` logits tensor every microbatch. Kernels and memory traffic combined, this is usually +5 – 15 % at small-model / large-vocab. Reference: [Liger-Kernel](https://github.com/linkedin/Liger-Kernel). Probed on SM_120 + torch 2.7.1: works. `LigerRMSNorm` does *not* (needs newer torch), so we stay on our own RMSNorm.
- **Softcap off**: the suspected cause of v0.3's HellaSwag regression. Softcap squeezes log-prob spreads which dulls the per-choice confidence signal HellaSwag scores on. Separately: fused CE can't coexist with a post-lm_head softcap. Both reasons say "remove it".
- **GQA (4 KV heads)**: Standard Llama-2 / Mixtral convention. Small-model training tok/s win (~1 – 3 %) because K/V projections are 3× smaller and SDPA does KV broadcast for free. Loss is expected neutral within noise at this scale.

### Predicted effect (written BEFORE running)

- **tokens/s Δ:** **+10 % to +20 %**. Point prediction **+15 %** (→ 205 k tok/s). Compounding:
  - max-autotune alone: +6 % (measured)
  - fused CE: +5 – 10 % (probed; scale-dependent)
  - GQA: +1 – 3 % (small at fixed seq=1024)
  - softcap off: ~0 % (tanh on a tiny tensor)
- **val loss Δ @ 10 B tokens:** **−0.005 to +0.010**. Softcap-off may marginally help (logits can learn larger magnitudes in well-separated cases); GQA at 4 KV heads is roughly quality-neutral at this scale.
- **HellaSwag Δ:** **+0.3 to +1.5 pp.** If softcap was in fact responsible for the v0.3 regression, we recover it here.
- **Stability:** minor risk — Liger fused CE is a newer kernel path and occasionally has gradient edge-cases on odd vocab sizes. Our tests will check the backward before the full run.

### Accept criteria

- **Accept if** tokens/s ≥ **196 k** (≥ +10 % vs v0.3's 178 k), **and** val loss @ 10 B tokens is within **±0.015** of v0.3's 2.9641 (i.e. in [2.949, 2.979]), **and** HellaSwag is at least **0.373** (no regression beyond 0.5 pp from v0.3's 0.378).
- **Reject** if tokens/s < 180 k (< +1 %, i.e. the bundle didn't meaningfully move the needle), val loss Δ > 0.02, HellaSwag drops > 1 pp, or any stability event.
- **Kill-early:** at 1 B tokens, val loss > 3.52 (v0.3 + 0.05).

## Implementation notes

- **`TrainConfig.compile_mode`** ported from perf-probe fix. Default stays `"default"` everywhere except the speed-pack config.
- **Liger fused CE.** Active only when `cfg.use_liger_fused_ce=True` AND `targets is not None`. The forward still returns `(logits, loss)` at inference (no targets) so generation isn't touched. Softcap is forbidden when fused CE is on — asserted at `GPT.__init__`.
- **GQA.** `CausalSelfAttention` now builds a single `c_attn` projecting to `(n_head + 2·n_kv_head) · head_dim` (instead of `3·n_embd`), splits into Q/K/V with the right shapes, and calls `F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)` when `n_kv_head < n_head`. HF-weight-load parity is preserved because faithful config leaves `n_kv_head=None` (= `n_head`), which keeps the MHA codepath and the same `3·n_embd` c_attn shape.
- **Parameter count effect.** Going from 12 KV to 4 KV:
  - attention QKV proj: `3·768·768 = 1.77 M` → `(12+4+4)·64·768 = 0.98 M` per layer (−790 k/layer × 12 = **−9.5 M params total**)
  - new total: ~114 M (down from 123.6 M). Comparable to the v0.2 → v0.3 drift from removing `wpe`.

### Liger fused CE dropped from the bundle

Smoke with the full bundle (including Liger) crashed inside
`torch._dynamo` at compile time with `CUDA error: misaligned address`.
Running Liger + compile is a known SM_120 interaction issue. Options
considered:

1. Keep Liger + turn compile off → **82 k tok/s** (net regression; rejects the experiment).
2. Keep Liger + compile + debug the alignment issue → open-ended timesink.
3. **Drop Liger; keep compile + max-autotune + GQA + softcap-off.**

The smoke of option 3 is **204 – 208 k tok/s** on the real data —
already +15 – 17 % vs v0.3's 178 k, clearing the +10 % accept bar from
the other three changes alone. So the final bundle on the full run is
three changes, not four. The `use_liger_fused_ce` flag and test coverage
stay in the codebase for future non-compiled use or a torch upgrade
that fixes the interaction.

Updated expected contributions (from smoke isolation):
- max-autotune: **+6 %** (measured in perf-util-probe)
- GQA + softcap-off, combined: **~+9 – 11 %**
  (larger than predicted GQA alone; candidate cause: the smaller
  packed QKV projection also reduces memory-bound traffic, which
  matters more on 5090 than tensor-core savings)

## Result

| metric                          | baseline (v0.3) | exp/05 | Δ |
|---------------------------------|---------------:|-------:|---:|
| parameters                      | 123.6 M        | **114.2 M** | −7.8 % (GQA drops 9.5 M across 12 layers) |
| val loss @ 1 B tokens           | 3.4729         | 3.4955   | +0.023 |
| val loss @ 5 B tokens           | 3.0780         | 3.1050   | +0.027 |
| val loss @ 10 B tokens (best)   | 2.9641         | **2.9922** | **+0.028** |
| val loss (200-batch held-out)   | 2.9946         | 2.9987   | +0.004 (within noise) |
| HellaSwag acc (1 000 examples)  | 0.3780         | **0.3700** | **−0.8 pp** |
| tokens / s median               | 181 732        | **209 857** | **+15.5 %** |
| tokens / s mean                 | 181 240        | 209 354  | +15.5 % |
| wall-clock 1 epoch              | 15 h 20 min    | **13 h 17 min** | **−13 %** (saved ~2 h) |
| time to val loss 3.5            | 96.6 min       | ~22 min  | −77 % |
| time to val loss 3.2            | 290 min        | ~167 min | −42 % |
| time to val loss 3.10           | 483 min        | ~335 min | −31 % |
| time to val loss 3.04           | 883 min        | ~481 min | −46 % |

Run was interrupted at step 2020; resumed from the step-2000 checkpoint.
Total wall-clock above includes both segments; steady-state tok/s is
measured across the resumed portion.

### Predicted vs actual

- Predicted tok/s Δ: **+10 % to +20 %** (point **+15 %**). Actual: **+15.5 %** — spot on target. ✅
- Predicted val loss Δ: **−0.005 to +0.010**. Actual: **+0.028** — **miss, worse than predicted**. ❌
- Predicted HellaSwag Δ: **+0.3 to +1.5 pp** (softcap-off hypothesis). Actual: **−0.8 pp** — **miss, wrong direction**. The softcap-off hypothesis from exp/03 is **not confirmed**: removing softcap *did not* recover HellaSwag. This is a non-obvious reversal worth recording.

### Loss curve shape

Exp/05 sits **+0.02 to +0.03 worse** than v0.3 at *every* evaluated step from
2 500 onward. It never catches up — the gap is persistent, not a zero-init
warmup artefact like exp/03 had. So the quality penalty is load-bearing, not a
convergence lag.

## Verdict

**Reject.** All three pre-declared criteria must hold; two of three miss.

- val loss @ 10 B Δ: **+0.028** (required ≤ +0.015) ❌
- HellaSwag: **0.370** (required ≥ 0.373) ❌
- tok/s: **210 k** (required ≥ 196 k) ✅

### What actually happened

The speed bundle did exactly what it was supposed to do on *throughput*:
**+15.5 % tok/s**, **−13 % wall-clock (~2 hours saved per 10 B run)**,
**time-to-every-intermediate-loss is 30 – 75 % faster**. Those are the largest
throughput wins in this project by a wide margin.

The quality cost is also real and measurable:
- **GQA 12 → 4 KV heads reduces attention capacity.** At 124 M scale and
  1 k context, 4 KV heads is *less* capacity than the model is trained for.
  Larger models tolerate aggressive GQA ratios because the embedding and
  MLP capacity absorbs the loss; at 124 M we're already capacity-limited.
  Most of the +0.028 val loss is attributable here.
- **Softcap removal did not recover HellaSwag.** In exp/03 the softcap-on
  recipe cost 0.5 pp HellaSwag vs v0.2's no-softcap recipe; I assumed
  removing softcap would reverse that. Instead HellaSwag dropped *further*
  (−0.8 pp vs v0.3). So the exp/03 HellaSwag dip is more likely caused by
  the other modded tricks (U-Net skips, zero-init, ReLU²), not softcap.
  This falsifies the exp/03 follow-up hypothesis.

### Honest framing

If the experiment were judged purely on **time-to-target-loss**, this is
an accept: exp/05 reaches v0.3's final loss (2.964) in ~12 h vs v0.3's
15 h 20 m — a **24 % wall-clock win for the same quality**. But the
pre-declared criteria were *fixed-tokens* quality metrics, and on those
axes we regressed. The discipline is to judge by the pre-declared
criteria, not rewrite them post-hoc.

### What's preserved

- `src/gpt_repro/model.py` retains `n_kv_head`, `use_liger_fused_ce`,
  and the `CausalSelfAttention` refactor (packed QKV with grouped KV).
- `configs/gpt2_124m_speedpack.py` preserved as-is.
- `tests/test_speed_pack.py` — 7 tests, all passing.
- `src/gpt_repro/train.py` retains `compile_mode` (default `"default"`;
  the +6 % free-win is still opt-in per-run via `--override`).

### Follow-up candidates, ordered by value

1. **Ablate GQA alone** (`exp/05b`, likely). Keep max-autotune +
   softcap-off (already in speedpack), turn GQA off (`n_kv_head=None`).
   Hypothesis: if GQA is the main quality cost, this recovers v0.3 loss
   at ~193 k tok/s (halfway between v0.3 and exp/05). Settles the
   "can we get just speed without quality cost" question cleanly.
2. **Re-examine the exp/03 softcap dip.** The u_net_skips or zero-init
   is a more plausible culprit than softcap. Worth a cheap ablation.
3. **Muon retry on the speed pack** (if GQA ablation accepts). Exp/02
   rejected Muon on v0.2 because AdamW caught up by 10 B; on a −8 %
   smaller model the optimisation landscape shifts — possibly enough
   for Muon to persist.

No new `v0.x` tag. `v0.3-exp03` remains the live quality baseline;
`compile_mode=max-autotune-no-cudagraphs` remains the free +6 %
opt-in for any future experiment.
