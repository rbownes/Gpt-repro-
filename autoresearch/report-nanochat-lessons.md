# What we learned from porting nanochat's ideas

Branch: `autoresearch/speed-2026-04-21`
Starting best before nanochat trials: val_bpb **1.2729**
Best after nanochat trials: val_bpb **1.2235** (−0.049, new best)

Three trials ported from `karpathy/nanochat`:

| # | Port | Result | Δ val_bpb | Δ tok/s |
|---|---|---|---|---|
| A | Tier-1 speed bundle (`gc.freeze`, `PYTORCH_ALLOC_CONF=expandable_segments`, double-buffered prefetch overlap) | discard | +0.007 | −2.6 % |
| B | FP8 tensorwise (nanochat-style `torch._scaled_mm`, 48 Linears converted) | discard (crash → slow) | +0.183 | −53 % |
| **C** | **MuonAdamW (Muon on 2-D block matrices, AdamW on embeddings/1D)** | **keep** | **−0.050** | **−6 %** |

## What each trial taught us

### Trial A — small speed tweaks didn't help
Bundled three ostensibly-free wins that nanochat applies at the top of `base_train.py`:

- `gc.disable() + gc.freeze()` after step 1 (claim: ~500 ms stalls avoided).
- `PYTORCH_ALLOC_CONF=expandable_segments:True` (allocator fragmentation).
- Double-buffered pinned tensors + prefetching the next batch *after* `.backward()` so the memmap→pinned CPU copy overlaps GPU compute.

Net result: **tok/s dropped 2.6 %**. The memmap→pinned copy in our loader is already <1 ms; turning it into a double-buffered prefetch just added an extra `next_batch()` call per grad-accum iteration and an extra pinned allocation. The loader was already well-overlapped by `non_blocking=True` + the implicit CUDA stream sync on first tensor use. gc.freeze alone might help in isolation, but bundled with the loader rework the signal was drowned.

**Lesson.** Don't port speed tricks unless you've identified the specific stall they solve. nanochat runs multi-hour jobs where a 500 ms gc stall matters; we run 300 s jobs with ~1760 steps, and our loop was already tight. "Our loader was already fine" isn't something you can see from staring at nanochat — you have to measure.

### Trial B — FP8 was a compatibility disaster
Ported `nanochat/fp8.py`'s minimal tensorwise FP8: one `autograd.Function` with two GEMMs calling `torch._scaled_mm`, decorated `@torch._dynamo.allow_in_graph`, no TransformerEngine tensor-subclass machinery.

Two failures:

1. **First attempt**: `@torch._dynamo.allow_in_graph` on the `autograd.Function` didn't prevent inductor from tracing `.abs().max()` inside `_to_fp8`. Crashed with `convert FlexibleLayout to FixedLayout first` during inductor lowering. This is a known PyTorch inductor/dynamo interaction bug on certain versions — nanochat's approach works on their pinned PyTorch but not ours.

2. **Second attempt (`@torch.compiler.disable` on `Float8Linear.forward`)**: Ran, but produced 48 graph breaks per forward pass (one per Linear). tok/s collapsed from 193 k → 91 k. val_bpb +0.18 — mostly from seeing half as many tokens, with some genuine FP8-precision loss on top.

**Lesson.** FP8 is binding: it only pays off when a single compiled graph can fuse FP8 quantisation with surrounding ops. The moment you introduce graph breaks, FP8's matmul speedup is wiped out by kernel-launch overhead on the scaffolding (amax, scale, cast). On SM_120 with our PyTorch version, the clean path is broken; the fallback path is worse than BF16. Consistent with the prior "FP8 via TE at 124M = −13 %" finding in this repo's experiment history.

### Trial C — MuonAdamW is the real win (−0.050 bpb)
Ported `nanochat/optim.py`'s single-GPU `MuonAdamW`: Polar-Express orthogonalisation + NorMuon variance reduction fused into one `torch.compile`'d step, AdamW (also fused) for embeddings and 1-D params.

Param groups built by shape:
- `wte.weight` (tied with `lm_head`) → AdamW with our tuned `(0.9, 0.9995)`.
- RMSNorm weights + all `.bias` → AdamW.
- The 4 distinct block-matrix shapes `(2304,768)`, `(768,768)`, `(3072,768)`, `(768,3072)` → 4 Muon groups, one per shape (Muon requires uniform shape within a group).

tok/s dropped 193 k → 181 k (−6 %). That's the Newton-Schulz/Polar-Express iterations (5 inner matmul steps per Muon group per optim step). Fewer optim steps in the same 300 s (1625 vs 1765). But **each step is worth more**: Muon's orthogonalised update is the single-shot equivalent of many AdamW steps.

Kept as-is with nanochat's out-of-the-box Muon hyperparams (`lr=0.02`, `momentum=0.95`, `beta2=0.9`, `ns_steps=5`). Zero retuning.

**Lesson.** Our prior 51 trials had been doing coordinate descent on AdamW hyperparams and batch size. The bounded gain from that direction was ≈−0.03 bpb per change; we were in a diminishing-returns regime. Swapping the *optimiser family* unlocked a step-change (−0.050 in one trial) precisely because we were close to AdamW's Pareto frontier and Muon's frontier sits lower.

## Transferable meta-lessons

1. **Optimiser beats kernels.** At 124 M / 300 s / sm_120, we can't win on matmul precision (FP8 is blocked by compile compat) or kernel fusion (no big stalls to eliminate). The sample-efficiency side of the ledger has more headroom. MuonAdamW is still the dominant single improvement available after batch-size tuning.
2. **"Drop-in" ports rarely are.** Three of three nanochat ideas needed adaptation. The FP8 path needed a completely different compile strategy (and still doesn't work). The speed bundle needed per-tweak attribution. Muon needed param-group construction plus an LR-frac schedule to support per-group base LRs.
3. **Port the *reasoning*, not the *code*.** nanochat's defaults are tuned for long Chinchilla-ratio training; ours is a 300 s cliff. We kept our measured `BETA2=0.9995` on the AdamW groups rather than nanochat's `0.96` — that was the right call, otherwise we'd have thrown away our own prior work.
4. **Cheap-looking tricks can have negative value.** Calling `next(train_loader)` 1 extra time per grad-accum iter sounds harmless but cost us 2.6 % tok/s. Always measure.
5. **Compile compat is a hidden dimension.** A clean FP8 port works on Hopper + nanochat's PyTorch; it crashes on Blackwell + ours. You can't tell this from the code.

## Total progression so far

| Point in time | val_bpb |
|---|---|
| Phase-A baseline (`29abdcf`, 2026-04-18) | 1.9297 |
| Running best at this session's start | 1.8053 |
| After 51-trial hyperparam sweep | 1.2729 |
| **After MuonAdamW port** | **1.2235** |

Total run-to-date improvement: **−0.706 bpb vs baseline, −0.58 bpb this session**.

## Open follow-ups (not pursued in this batch)

- **Retune Muon LR** for our 300 s regime. nanochat's `lr=0.02, momentum=0.95, beta2=0.9` was used verbatim — likely has room.
- **Retune AdamW LR separately from Muon** for the embedding+1D group (currently both scale off the same `lr_frac` schedule).
- **Re-check `MIN_LR_RATIO` / `WARMUP_FRAC`** under MuonAdamW — the optima may have shifted again, as they did at the previous batch-size change.
- **`gc.freeze` in isolation** (without the prefetch rework) — possibly worth 1 %.
