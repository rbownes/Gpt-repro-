# Perf probe — "why is my GPU utilisation at 50 %?"

Date: 2026-04-20 (between exp/03 and the next experiment)
Branch: `exp/03-modded-tricks` (v0.3 config, AdamW, BF16, `torch.compile` default)

## Short answer

**Your GPU utilisation is not at 50 %.** `nvidia-smi`'s `utilization.gpu` sat
at **99 – 100 %** across the whole training run. The 50 % you saw is almost
certainly `utilization.memory` — 55 – 57 % consistently — which measures
**memory-controller busy fraction**, not "fraction of peak throughput".

The more load-bearing metric is **Model FLOPs Utilisation (MFU)**: what
fraction of the GPU's peak tensor-core throughput we're actually using.
At 174 k tok/s on our 124 M-param model, we're at **~31 % MFU**
(128.6 TFLOPS actual / 419 TFLOPS peak BF16 dense on the 5090). That
is normal for a 124 M model on a 32 GB card — the matmul shapes
(`d_model=768`, `hidden=3072`) are too small to saturate tensor cores,
and memory-bound ops (RMSNorm, ReLU², softmax, residuals) dominate per-step
time. Big models (≥ 7 B) routinely reach 45 – 55 % MFU for structural
reasons, not because we're doing anything wrong.

## Measurements (single RTX 5090, SM_120, driver 595.58.03, torch 2.7.1+cu128)

### Idle
- `utilization.gpu = 10 %` (from Hyprland desktop)
- VRAM: 1.2 / 32 GB

### Training (v0.3 modernplus config, real FineWeb-Edu shards)
- `utilization.gpu = 99 – 100 %` (sampled every 1 s for 30 s steady state)
- `utilization.memory = 55 – 57 %`
- Power: **545 – 570 W / 600 W cap** (~ 92 % of TGP)
- SM clock: **2 880 – 2 895 MHz** (boost)
- VRAM: 14.6 / 32 GB

## Bench table (30-step slices on real data, steady-state tok/s)

| # | Change vs v0.3 baseline | tok/s | MFU | Δ vs baseline |
|---|---|---:|---:|---:|
| 0 | **baseline** — `torch.compile(mode="default")`, grad_accum=32, mb=16 | 173 – 174 k | 31.0 % | — |
| 1 | On-device loss accumulation (remove `loss.item()` from inner loop) | 173 – 176 k | 31.2 % | noise |
| 2 | `micro_batch = 32`, `grad_accum = 16` (same effective batch) | 175 k | 31.3 % | +0.5 % |
| 3 | `compile_mode = "max-autotune-no-cudagraphs"` | **184 k** | **33.0 %** | **+6 %** |
| 4 | Fix 3 + fix 2 combined | 184 k | 33.0 % | (no stack with fix 2) |

Fix 3 is the only change that moved the needle. The step-1 tok/s in the log
(30 – 40 k / 130 – 140 k) is compile-warmup noise; every row above uses steady
state starting at step 10.

## What we tried that didn't help

- **Removing `loss.item()` inside the grad-accum loop.** Theoretical cost
  is 32 GPU → CPU syncs per outer step; in practice PyTorch's async
  dispatch already hides most of that, so we saw 0 – 1 % change. Kept
  the fix anyway because it's strictly better (one sync per step, not 32),
  but it's not load-bearing.
- **`micro_batch=32, grad_accum=16`.** Larger matmuls should give bigger
  tensor-core blocks, but at d=768 we're still in the "small" regime
  whether the batch is 16 or 32. `micro_batch=64` OOMs on the lm_head.
- **`compile_mode="reduce-overhead"`.** Enables CUDA graphs. Crashes with
  "accessing tensor output of CUDAGraphs that has been overwritten" because
  our tied-embedding + grad-accumulation pattern breaks CUDA graph
  invariants. Tried `torch.compiler.cudagraph_mark_step_begin()` — same
  crash on `loss.backward()`. Kept the `mark_step_begin` call behind the
  config guard in case this works on a future torch release, but marked
  the mode as unsupported with our tied-embedding setup.
- **`compile_mode="max-autotune"` (with cudagraphs).** Same CUDA graph
  issue as reduce-overhead. The `-no-cudagraphs` variant (fix 3 above) is
  what we land on.

## What would actually move MFU from 33 % → 50 %+

- **FP8 matmul** (roadmap #04). Halves the effective compute cost of every
  GEMM. We're compute-bound, so this is the single biggest lever. Literal
  expected win on 5090: +50 – 100 % tok/s.
- **Fused attention + norm + residual kernel.** Most of the non-matmul
  time is in RMSNorm + residual + softmax — all memory-bound. A fused
  kernel (e.g. `NVIDIA/TransformerEngine`, Liger Kernels) removes the
  memory round-trips. Usually +10 – 20 % on small models.
- **Packed sequences / variable-length attention.** Our batches are all
  full 1024-length. If we packed multiple documents per sequence we'd
  amortise the attention cost, but that's a data-pipeline change. Out of
  scope for a quick perf pass.

## Action taken

1. Added `compile_mode` to `TrainConfig` (default `"default"` — existing
   runs unchanged).
2. Replaced the scalar `loss_accum` with a GPU-side accumulator and a
   single `.item()` per outer step. Correctness identical.
3. Did **not** change `configs/gpt2_124m_modernplus.py` — keeping v0.3
   reproducible at 178 k tok/s (the number in the accepted report).
   Future experiments opt into `compile_mode=max-autotune-no-cudagraphs`
   via config override for a free ~6 % throughput.

## Recommendation

The "50 % GPU utilisation" reading was the memory-controller metric, not a
real bottleneck. The real metric (MFU) is 31 – 33 % which is normal for
124 M on a 5090. The single cheap win is `compile_mode="max-autotune-no-cudagraphs"`
(+6 % tok/s, one extra minute of compile time). The meaningful next step
is the **FP8 experiment (roadmap #04)** — that's where 50 %+ MFU lives on
this hardware.
