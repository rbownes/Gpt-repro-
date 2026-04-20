---
id: 04-fp8
status: rejected
baseline_run: runs/03-modded-tricks/
experiment_run: runs/04-fp8-smoke/ (no full run; rejected from smoke data alone)
baseline_tag: v0.3-exp03
date: 2026-04-20
author: rjbownes
seeds: [0]
---

# Experiment 04 — FP8 matmul via TransformerEngine

## Previous baseline

- **Config:** `configs/gpt2_124m_modernplus.py` @ commit `e70512c`, git tag `v0.3-exp03`.
- **Arch:** v0.3 modernplus block — RoPE + RMSNorm + QK-Norm + ReLU² + zero-init proj + U-Net skips + logit softcap. 123.6 M params.
- **Optimizer:** AdamW β = 0.9 / 0.95, wd 0.1, peak LR 6 e-4.
- **Data:** FineWeb-Edu-10B, 19 073 steps @ 524 288 tok/step, seed 0.
- **Precision:** BF16 autocast, `torch.compile(mode="default")`, SDPA flash.
- **Baseline metrics (single seed):**
  - val loss @ 1 B / 5 B / 10 B: **3.4729 / 3.0780 / 2.9641**
  - HellaSwag (1 000): **0.3780**
  - tokens/s median: **178 215**
  - MFU (at 6N FLOPs/token): **~31 % of 419 TFLOPS BF16 peak**
  - wall-clock 1 epoch: **15 h 39 min**

The perf probe (`experiments/perf-util-probe.md`) established that we're
compute-bound at this scale, that memory-bound ops (RMSNorm, ReLU², softmax)
dominate per-step time in addition to the matmul itself, and that the
single clean next lever is FP8. This experiment tests that directly.

## The change

Swap every `nn.Linear` in the hidden-state path (attention `c_attn`,
attention `c_proj`, MLP `c_fc`, MLP `c_proj`) for `transformer_engine.pytorch.Linear`
and wrap forward in `te.fp8_autocast` with the **DelayedScaling HYBRID**
recipe (E4M3 forward / E5M2 backward, amax history length 16, "max" reduce).
The `lm_head` stays on `nn.Linear` (tied to `wte` embedding; swapping it
would break the tie and also needs an FP8-sensitive softmax downstream).

Nothing else changes: same model architecture, same optimizer, same
schedule, same token budget, same data, same compile mode, same seed.

- **Diff:** branch `exp/04-fp8`. Implementation pending.
- **Files touched:** `src/gpt_repro/model.py`, `configs/gpt2_124m_fp8.py`
  (new), `tests/test_fp8.py` (new), `pyproject.toml` ([fp8] extra already
  present; documenting the `--no-build-isolation` install step).
- **Hyperparameters introduced / changed:**
  - `use_fp8`: `false` → `true`
  - `fp8_recipe`: `None` → `"delayed_hybrid"`
  - FP8 config constants: `fp8_amax_history_len = 16`, `fp8_amax_compute_algo = "max"`

### Recipe choice

TE 2.13 on SM_120 supports **DelayedScaling** (works) and `Float8CurrentScaling`
(should work). It **rejects MXFP8BlockScaling** with an explicit assert:

> `MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet.`

So block-scaled FP8 — the typical datacenter-Blackwell recipe — isn't
available. DelayedScaling with HYBRID is the robust choice at this scale.

## Why it might improve

Reference: [NVIDIA TransformerEngine
docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/),
[FP8 Formats for Deep Learning (Micikevicius et al. 2022)](https://arxiv.org/abs/2209.05433),
[nanochat FP8 discussion](https://github.com/karpathy/nanochat/discussions/382).

**Mechanism.** FP8 halves the bytes moved per GEMM input and doubles
tensor-core peak throughput on Blackwell. Our perf probe established:
- 31 % MFU at 419 TFLOPS BF16 peak = 130 TFLOPS actual.
- 5090 FP8 dense peak is **~838 TFLOPS** (2× BF16).
- Not all ops are GEMMs — norms, softmax, activations, lm_head are still
  BF16 (or FP32 internally). So we can't get a full 2× speedup, but a
  meaningful chunk of per-step time IS those GEMMs.

Empirical reference points: nanochat on Blackwell reports ~1.3 – 1.5 × tok/s
with FP8 vs BF16 at the 120 – 300 M scale. Modded-nanogpt's FP8 head-only
experiment adds ~10 %. A full-matmul FP8 on our 124 M should land in the
**+20 to +60 %** band.

### Predicted effect (written BEFORE running)

- **tokens/s Δ:** **+20 % to +60 %**. Point prediction **+30 %** (231 k tok/s).
- **val loss Δ @ 10 B tokens:** **−0.01 to +0.01** (within noise of BF16).
  FP8 is lossy but DelayedScaling is designed to match BF16 loss at this
  model scale on well-behaved pretraining data.
- **HellaSwag Δ:** **±1 pp** (within noise).
- **Stability:** minor risk of activation outliers blowing the amax
  history in early training. DelayedScaling's warmup usually absorbs this;
  if it doesn't we'd see NaN/inf grad spikes.

### Accept criteria

- **Accept if** tokens/s ≥ **214 k** (≥ +20 % over v0.3's 178 k), **and**
  val loss @ 10 B tokens is within **±0.015** of v0.3's 2.9641 (i.e. in
  [2.949, 2.979]). Both must hold.
- **Reject if** tokens/s < 190 k (< +7 %), val loss Δ > 0.02, or any
  training instability (NaN amax, gradient blow-up, compile failure we
  can't work around).
- **Kill-early:** if val loss at 1 B tokens is more than **0.05** worse
  than v0.3's 3.4729 (i.e. > 3.52), stop.

## Implementation notes

- **Config flag gating.** `GPTConfig.use_fp8` defaults to `False`, so
  faithful / v0.2 / v0.3 configs are unchanged and `test_hf_weight_load`
  keeps passing.
- **Module swap.** A `make_linear(cfg, in, out, bias)` helper returns
  `te.Linear` when `use_fp8=True`, `nn.Linear` otherwise. Applied to
  attention (`c_attn`, `c_proj`) and MLP (`c_fc`, `c_proj`). `lm_head`
  stays `nn.Linear` (tied-embedding constraint and FP8-sensitive softmax).
- **FP8 autocast.** Wrap the block loop in `te.fp8_autocast(enabled=use_fp8,
  fp8_recipe=recipe)`. The recipe is constructed once at `GPT.__init__`
  and reused.
- **Parameter discovery.** Our Muon param-splitter in `optim.py` keys on
  suffix strings like `c_attn.weight`; TE's `te.Linear.weight` has the
  same name, so the splitter still works correctly if Muon is ever
  re-enabled on top of FP8.
- **Compile compatibility.** Probed: `te.Linear` composes with
  `torch.compile` in the simple forward case. Full grad-accum + backward
  + tied-embed is the real test; if it crashes, fall back to
  `compile=False` and re-evaluate — tok/s may still win thanks to FP8's
  raw GEMM advantage.
- **Init.** TE's `te.Linear` uses standard PyTorch init, so our
  `apply(_init_weights)` override still sets every linear to N(0, 0.02),
  and the residual-scale / zero-init passes still catch `c_proj.weight`.

## Result — from smoke (no full run, see verdict)

All smoke runs: same v0.3-modernplus arch, 15 steps after compile warmup,
real FineWeb-Edu data, only `use_fp8` and `compile_mode` varying.

| config | steady-state tok/s | Δ vs BF16 default |
|---|---:|---:|
| BF16 + `compile="default"` (= v0.3 path) | **174 k** | — |
| **FP8** + `compile="default"` | 152 k | **−13 %** |
| FP8 + `compile_mode="max-autotune-no-cudagraphs"` | 152 k | −13 % |
| FP8 + `compile=False` | 94 k | −46 % |

FP8's loss curve in the smoke run tracks BF16 closely (steps 1-15 both
drop from ~14.2 → ~9.6), so there's no evidence of FP8 hurting
convergence at this scale. But because the **primary accept criterion is
tokens/s ≥ 214 k (+20 %)** and all FP8 variants land at 152 k (−13 %),
running the full 15 h at negative throughput would only confirm a loss
number on a slower run — not worth the wall-clock.

## Why FP8 loses on this hardware/scale

Three compounding reasons, none of which are fixable with a config flag:

1. **TE per-call Python overhead dominates.** 4 linears × 12 layers ×
   32 grad-accum microbatches = **1 536 `te.Linear.forward` calls per
   optimiser step**. Each does FP8 quant/dequant, amax history update,
   and Python-level bookkeeping. At d = 768 / hidden = 3 072 the GEMMs
   themselves are small enough that the bookkeeping cost per call is
   comparable to the GEMM cost. At 7 B+ the GEMMs are ~60× bigger but
   the overhead is constant, so large models see the expected win.

2. **`te.fp8_autocast` is a compile-graph boundary.** The context
   manager lives in Python and breaks `torch.compile`'s inductor traces
   across the `fp8_autocast` call. We lose BF16's fusion benefits
   (element-wise ops folded into neighbouring GEMMs) without fully
   recovering them with FP8-native fusion. This is why
   `max-autotune-no-cudagraphs` — which got us +6 % on BF16 (see
   `experiments/perf-util-probe.md`) — gives nothing on FP8.

3. **SM_120 FP8 tensor-core path is worse than SM_100.** MXFP8 block
   scaling is the recipe that produces the big wins on H200/B200
   (SM_100). TE 2.13 explicitly rejects it on SM_120 with
   `AssertionError: MXFP8 (for all gemm layouts) is not supported on
   12.0+ architectures yet`. Consumer Blackwell gets DelayedScaling
   only, which on H-series is already slower than MXFP8. On
   SM_120 the raw FP8 GEMM throughput advantage over BF16 is smaller
   than the datasheet suggests once you actually plumb it through TE.

## Verdict

**Reject.** Pre-declared primary criterion was tokens/s ≥ 214 k (+20 %).
Measured: 152 k (−13 %). Secondary criterion (val loss within ±0.015)
was not evaluated because the throughput regression alone makes the
full run a waste of GPU time — 15 hours of training to confirm a slower
path is slower. Loss curves in the smoke match BF16, so there's no
evidence FP8 broke convergence; the change simply doesn't help.

### What's preserved, and why

- `src/gpt_repro/model.py` keeps the `use_fp8`, `fp8_recipe`, and
  `make_linear` helpers. All disabled by default.
- `configs/gpt2_124m_fp8.py` preserved — flips one flag on top of v0.3.
- `tests/test_fp8.py` — 4 FP8-specific tests (shape, argmax agreement,
  backward finite, te.Linear is the right class). All pass.
- The `[fp8]` pyproject extra stays — needed for TE's NCCL-linked build,
  and the install instructions in the v0.3 README now cover the
  `--no-build-isolation` step.

### Follow-up candidates, in order of expected payoff

1. **Revisit at 350 M or 1.5 B scale.** Same code path; larger matmuls
   should close the TE-overhead gap. If we ever do the 350 M rung, FP8
   is the first throughput knob to pull before anything else.
2. **Try `torchao`'s `float8_dynamic_quant` path** (now merged). Gives
   a PyTorch-native FP8 without TE's per-call Python overhead. Would be
   exp/04b if we want to chase throughput more.
3. **Wait for TE SM_120 MXFP8 support.** Tracked in TE repo; when
   available, retry this experiment verbatim.

No new `v0.x` tag. `v0.3-exp03` remains the live baseline; exp/05
branches from there.
