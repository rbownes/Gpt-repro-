---
id: 00-baseline
status: accepted
baseline_run: runs/baseline/
experiment_run: runs/baseline/
baseline_tag: v0.1-baseline
date: 2026-04-18
author: rjbownes
seeds: [0]
---

# Experiment 00 — Faithful GPT-2 124M baseline

## Previous baseline

None. This *is* the baseline: a strict Radford et al. 2019 GPT-2 small
reproduction on FineWeb-Edu-10B. Every future experiment is measured against
the numbers in this report.

## The change

- **Configuration:** `configs/gpt2_124m.py` @ commit `626509c`.
- **Architecture:** 12 layers × 768 hidden × 12 heads, 1024 context, 124 M params.
  LayerNorm (pre-norm), learned positional embeddings, GELU (`tanh` approximation),
  tied input/output embeddings, residual-projection init scaled by `1/√(2·n_layer)`.
- **Data:** FineWeb-Edu-10B (`HuggingFaceFW/fineweb-edu` `sample-10BT`),
  tokenised with tiktoken `gpt2`, EOT-separated, sharded as uint16 `.bin`
  (1 × 100 M-token val shard + 99 × ~99.5 M-token train shards, 9.85 B train tokens).
- **Optimizer / schedule:** AdamW β = (0.9, 0.95), wd 0.1, eps 1e-8;
  peak LR 6 e-4, linear warmup 715 steps, cosine decay to 0.1 × peak,
  grad clip 1.0, effective batch 524 288 tokens/step, 19 073 steps (~10 B tokens).
- **Precision / kernels:** BF16 autocast, `torch.compile`, SDPA flash backend
  (see README "Hardware notes" for why flash beats cuDNN under compile on SM_120).
- **Hardware:** single RTX 5090 (SM_120, 32 GB), AMD 9950X3D, 60 GB RAM.

## Why it might improve

N/A — this is the reference. See [`experiments/README.md`](../README.md) for the
roadmap of improvements to layer on top of this run.

## Implementation notes

- `tests/test_hf_weight_load.py` passes: our module's logits match HF
  `GPT2LMHeadModel` on real prompts within 1 e-3 tolerance. Architecture is
  verified faithful.
- **Kernel-backend story on SM_120** — noted in detail in the README; short
  version: `sdpa_flash` + `torch.compile` was picked over the plan's original
  `sdpa_cudnn` default because cuDNN SDPA crashes inductor on SM_120 under
  torch 2.7.1 (stride assert). `sdpa_flash` composed cleanly and measured
  ~190 k tok/s sustained, well above the 150 k target.
- **Smoke path is part of CI discipline** — the 9-test `pytest` suite plus a
  debug config on a synthetic tiny shard is what I run before every
  experiment branch.

## Result

### Loss vs tokens (compute efficiency curve)

| milestone              | step   | val loss |
|------------------------|-------:|---------:|
| step 500 (first eval)  |  500   | 5.2042   |
| 1 B tokens             |  2 000 | 3.5984   |
| 5 B tokens             |  9 500 | 3.1669   |
| 10 B tokens (end)      | 19 073 | **3.0407** |

### End-of-training numbers

| metric                              | value           |
|-------------------------------------|----------------:|
| val loss @ 10 B tokens (best)       | **3.0402**      |
| val loss (held-out, 200-batch eval) | 3.0473          |
| HellaSwag acc (1 000 val examples)  | **0.3680**      |
| tokens/s median (steady state)      | 190 004         |
| tokens/s mean (steady state)        | 189 345         |
| wall-clock, 1 epoch (10 B tokens)   | 14 h 41 min (52 863 s) |
| time to val loss 3.5                | ~140 min        |
| time to val loss 3.2                | ~394 min        |
| time to val loss 3.1                | ~601 min        |

HellaSwag of 36.8 % is well above the paper's 28.9 % baseline for GPT-2 124M.
Most likely drivers:
1. **FineWeb-Edu is cleaner than WebText** — the whole point of step 1 in the
   improvements roadmap. The paper's 28.9 % number is on WebText; 10 B tokens
   of FineWeb-Edu is a different (and better) training distribution.
2. Scored on 1 000 of ~10 k val examples, with mean-per-token log-likelihood
   (equivalent to acc_norm). Some per-run noise expected.
3. Possible HellaSwag-format sensitivity (activity-label prefix vs raw `ctx`).

### Not yet reported

- **WikiText-103 PPL** and **LAMBADA accuracy** need an `lm-eval` adapter for
  our module (it's not a HF checkpoint). Deferred — the val-loss curve and
  HellaSwag are enough to unlock the experiment pipeline. The `[eval]` extra
  is installed and ready; an adapter will be added in an early experiment
  branch so later runs can all report these.

## Verdict

**Accept.** Baseline is faithful to the paper architecture, trained end-to-end
without incident, converges smoothly (no loss spikes, no NaN restarts), and
the resulting val loss of 3.04 is within one contour of the plan target
(2.85 – 3.00). A second epoch would likely hit 2.85 – 2.90 but is not
necessary — the job of `00-baseline` is to produce an honest reference, not
to push absolute quality.

**Next:** tag `v0.1-baseline` at commit `626509c` and branch the first
improvement (`exp/01-modern-block`: RoPE + RMSNorm + SwiGLU + QK-Norm).
