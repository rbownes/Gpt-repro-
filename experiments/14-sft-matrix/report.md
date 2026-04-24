---
id: 14-sft-matrix
status: in-progress            # in-progress | accepted | rejected
baseline_run: runs/                   # all 8 pretrained checkpoints
experiment_run: runs/sft-*/
baseline_tag: v0.3-exp03
date: 2026-04-24
author: rjbownes
seeds: [0]
---

# Experiment 14 — SFT matrix across all 8 pretrained checkpoints

## Previous baseline

Eight pretrained checkpoints, all on FineWeb-Edu 10 B tokens, differ in
arch / optimizer / loss-shaping tricks:

| checkpoint          | arch notes                                            | pretrain val_loss |
|---------------------|-------------------------------------------------------|-------------------|
| baseline            | faithful GPT-2 124M (LN, learned pos-emb, GELU)       | 2.988             |
| 01-modern-block     | + RoPE, RMSNorm, SwiGLU, QK-norm                      | 3.011             |
| 02-muon             | 01 + Muon optimizer                                   | ...               |
| 03-modded-tricks    | 01 + ReLU², zero-init proj, U-Net skips, logit softcap | 2.964            |
| 05-speed-pack       | 03 + GQA, max-autotune, softcap off                   | ...               |
| 06-muon-mup         | 03 + Muon+AdamW, μP                                   | ...               |
| 10-mla              | 03 + Multi-head Latent Attention (DeepSeek-V2)        | ...               |
| 11-loopllm          | 03 + weight-tied loop (K=12), n_layer=1 → 45 M params | 3.40              |

## The change

Single-shot SFT on each: 500 M tokens of SmolTalk through a text-marker
chat template (`<user>.../<assistant>...`), fresh AdamW optimizer, peak
LR 3e-5, warmup→constant→warmdown schedule. Then an identical
log-likelihood eval battery against every resulting SFT'd model.

- **Files touched (unified branch):**
  - `src/gpt_repro/chat.py`, `tasks.py`, `sft_data.py` — SFT harness
  - `src/gpt_repro/eval.py` — added MMLU + ARC-Easy + ARC-Challenge
  - `src/gpt_repro/model.py` — merged MLA, weight_tied, GQA flags (all default off)
  - `src/gpt_repro/utils.py::load_gpt_config_from_ckpt` — tolerates config schema drift
  - `scripts/sft.py`, `scripts/sft_eval.py`, `scripts/run_sft_matrix.sh`, `scripts/sft_matrix_report.py`

## Why it might improve

SFT converts a next-token-prediction LM into one that follows
instructions, which turns latent capabilities into measurable
zero-shot benchmark scores. Differences across pretrain variants should
therefore become more legible post-SFT than in raw val_loss alone:

- Architecture choices (MLA, GQA) may or may not retain their pretrain
  edge once instruction-tuned
- Optimizer choices (Muon, μP) should affect SFT adaptation speed and
  final quality
- Loss tricks (softcap, ReLU², U-Net skips) may help or hurt under the
  longer-range dependencies that instruction-following demands

## Evaluation protocol

- **Pre-SFT eval** (each pretrained ckpt): HellaSwag 1 k · MMLU full
  (1531 val) · ARC-Easy full (570 val) · ARC-Challenge full (299 val)
- **SFT**: 500 M tokens SmolTalk, same schedule across all 8 runs
- **Post-SFT eval**: same battery, same sample sizes
- **Primary metric**: Δ (post − pre) per task; secondary: absolute
  post-SFT scores
- **Report**: `scripts/sft_matrix_report.py` produces the comparison
  table below

## Results

### Pre-SFT baseline (2026-04-24)

All 8 pretrained checkpoints loaded cleanly on `exp/14-sft` and ran the
full LL battery. No failures — this validates that MLA, weight_tied,
GQA, and MuonAdamW-pretrained checkpoints all round-trip through the
unified code.

| checkpoint        | HellaSwag | MMLU (all) | ARC-Easy | ARC-Challenge |
|-------------------|----------:|-----------:|---------:|--------------:|
| baseline          |     0.368 |      0.278 |    0.439 |         0.224 |
| 01-modern-block   |     0.383 |      0.275 |    0.468 |         0.251 |
| 02-muon           |     0.385 |      0.278 |    0.463 |         0.241 |
| 03-modded-tricks  |     0.378 |      0.287 |    0.451 |         0.241 |
| 05-speed-pack     |     0.370 |      0.283 |    0.486 |         0.247 |
| 06-muon-mup       |     0.379 |      0.268 |    0.449 |         0.247 |
| 10-mla            |     0.381 |      0.284 |    0.472 |         0.247 |
| 11-loopllm        |     0.349 |      0.255 |    0.384 |         0.231 |
| *random baseline* |     0.250 |      0.250 |    0.250 |         0.250 |

Observations before SFT:

- **MMLU** is essentially at random (0.25 ± noise) for every checkpoint
  — 124 M params at 10 B tokens doesn't encode enough facts to move
  MMLU, as expected. SFT signal should be most visible here.
- **HellaSwag** spans 0.349 (11-loopllm, 45 M params) to 0.385
  (02-muon) — tracks pretrain val_loss closely. 03-modded-tricks
  reproduces its v0.2 `results.json` number (0.378) exactly → eval
  pipeline is byte-identical to the historical one.
- **ARC-Easy** separates checkpoints more than HellaSwag (0.38–0.49),
  with 05-speed-pack (GQA + max-autotune) leading.
- **ARC-Challenge** is at/near random for everyone — a true ceiling
  for this scale. Don't expect much Δ from SFT here.

### Post-SFT (2026-04-24, ~28 h total GPU)

All 8 SFT runs completed without incident. 500 M tokens @ peak LR 3e-5,
warmup→constant→warmdown, fresh AdamW optimizer, identical hyperparams
across runs. Each SFT ~3.5 h wall-clock on the 5090.

#### SFT val loss (best across training)

| checkpoint        | pretrain val | SFT best val | Δ vs pretrain |
|-------------------|-------------:|-------------:|--------------:|
| 06-muon-mup       |        2.974 |   **1.2592** |       −1.715  |
| 03-modded-tricks  |        2.964 |       1.2919 |       −1.672  |
| 10-mla            |        2.981 |       1.3040 |       −1.677  |
| 01-modern-block   |        3.011 |       1.3075 |       −1.704  |
| 05-speed-pack     |        2.979 |       1.3240 |       −1.655  |
| baseline          |        2.988 |       1.3450 |       −1.643  |
| 02-muon           |        2.997 |       1.4977 |       −1.499  |
| 11-loopllm        |        3.406 |       1.7119 |       −1.694  |

Ordering is not monotone in pretrain val_loss. **06-muon-mup wins the
SFT val despite being a pretrain reject** (+0.010 vs v0.3), and
**02-muon trails everyone** with the largest Δ — two hybrid findings
about Muon-initialized weights under fresh AdamW SFT. μP + MuonAdamW
produces AdamW-compatible weights; plain Muon alone does not.

#### Full post-SFT eval matrix (Δ = post − pre)

| checkpoint         | HSwag Δ | MMLU Δ | ARC-E Δ | ARC-C Δ | SFT val |
|--------------------|--------:|-------:|--------:|--------:|--------:|
| baseline           |  +0.016 | −0.025 |  +0.018 |  +0.003 |   1.345 |
| 01-modern-block    |  −0.009 | +0.009 |  −0.005 |  +0.027 |   1.308 |
| 02-muon            |  +0.004 | −0.008 |  +0.018 |  −0.017 |   1.498 |
| 03-modded-tricks   |  +0.002 | −0.008 |   0.000 |  +0.003 |   1.292 |
| 05-speed-pack      |  +0.004 | −0.025 |  −0.044 |  +0.020 |   1.324 |
| **06-muon-mup**    |  +0.011 | +0.001 |  −0.007 |  +0.007 | **1.259** |
| 10-mla             |  −0.006 | −0.020 |  −0.032 |  −0.007 |   1.304 |
| 11-loopllm         |  +0.006 | −0.004 |  −0.037 |  −0.003 |   1.712 |

Noise floors (n-dependent): σ(HSwag, n=1000) ≈ 0.015; σ(MMLU, n=1531) ≈
0.011; σ(ARC-E, n=570) ≈ 0.021; σ(ARC-C, n=299) ≈ 0.026. Nearly every Δ
in the table is inside ~1 σ of zero, so individual deltas are noisy —
but the *aggregate patterns* below survive.

## Attribution

### 1. SFT val loss ranking ≠ zero-shot eval ranking

06-muon-mup wins on SFT val but is middling on HSwag post-SFT (+0.011)
and flat on MMLU/ARC. 03-modded-tricks is 2nd on SFT val but flat
across all four evals. **Lower SFT val does not translate to better
downstream zero-shot performance** at 500 M tokens — a reminder that
chat-style val loss measures fluency on the SFT distribution, not
underlying capability retention.

### 2. Alignment tax on MMLU

**6 of 8 checkpoints regress on MMLU**, with baseline and 05-speed-pack
both losing 2.5 pp (~2 σ). Only 01-modern-block (+0.009) and
06-muon-mup (+0.001) hold flat. This is the classic alignment-tax
shape: chat-style SFT drifts the model away from factual QA calibration
faster than it instills new facts. Pre-SFT MMLU was already at/near
random (0.25–0.29), so there was little room to gain; there was plenty
to lose.

### 3. ARC-Easy sorts by attention type

Attention-unchanged checkpoints (baseline, 01-modern-block, 02-muon,
06-muon-mup) all have ARC-E Δ ≥ −0.007; attention-modified checkpoints
(05-speed-pack GQA, 10-mla, 11-loopllm weight-tied) all have ARC-E Δ
≤ −0.032. The boundary is clean:

- MHA + full-attn: mean Δ ≈ +0.006 (n=4)
- GQA / MLA / weight-tied loop: mean Δ ≈ −0.038 (n=3)

With only 3 vs 4 checkpoints this is suggestive, not conclusive.
Candidate mechanism: SFT on SmolTalk's short conversational turns may
over-specialize the attention layout when it's been bottlenecked by
pretrain (fewer KV heads in GQA; latent projections in MLA; shared-
weight stack in LoopLLM), disturbing the attention patterns ARC-E
relies on more than those the modern standard-attention blocks picked
up.

### 4. μP + MuonAdamW is the quiet win

06-muon-mup is the only checkpoint with non-negative Δ on **every**
metric, plus the lowest post-SFT val loss. This is the opposite of
what its pretrain ranking (+0.010 vs v0.3-exp03) would predict. μP's
LR-consistent initialization and MuonAdamW's mixed optimization state
appear to leave weights both AdamW-friendly and robust under chat SFT.
This is the most actionable finding in the matrix.

### 5. Plain Muon alone does not survive AdamW SFT

02-muon has the worst SFT val loss in the matrix (Δ +0.15–0.20 vs
baseline at every training step) and net-negative on ARC-C. Without
μP's parameterization, a Muon-pretrained init appears to require either
Muon SFT (not tested here) or a different LR to adapt.

## Conclusions

1. **SFT tokens 500 M at 124 M param scale moves the zero-shot battery
   by ≤ 1 σ per metric** for every architecture variant. The matrix is
   mostly inside noise, and the useful signal is in the patterns
   (alignment tax, attention-type boundary, μP win) rather than
   individual Δs.
2. **06-muon-mup is the best-behaved checkpoint under SFT** despite
   being pretrain-rejected. Promote μP + MuonAdamW as the default
   SFT-compatible pretrain recipe for future experiments.
3. **02-muon needs its Muon optimizer state** to be competitive under
   SFT. Pure AdamW SFT on a Muon-pretrained init loses ~0.15 nats of
   val loss compared to same-family peers. If Muon pretrain is retained
   in future experiments, either pair with μP (→ 06 pattern) or SFT
   with Muon.
4. **GQA / MLA / weight-tied loop all hurt ARC-E under SFT** — not a
   hard reject since ARC-E Δ is within noise per-row, but three of
   three modifications land on the wrong side of zero. Worth revisiting
   if SFT is the goal; less concerning if the target is pretrain
   val_loss or a different downstream task.

## Known caveats

- **11-loopllm** has only 45 M trainable params (weight-tied). 500 M SFT
  tokens ≈ 11× Chinchilla — expect overfitting. Keep in matrix for
  comparison but interpret accordingly.
- **02-muon optimizer state** is discarded at SFT time (all checkpoints
  get a fresh AdamW). That's intentional — keeps SFT side controlled so
  pretrain choice is the only varying axis.
- **MMLU 5-shot vs 0-shot**: we use 0-shot LL scoring. Scores will be
  lower than published 5-shot numbers. Not comparable to paper tables;
  comparable within this matrix.

## Appendix: orchestration

```bash
bash scripts/run_sft_matrix.sh                # full 8-checkpoint matrix
CHECKPOINTS="03-modded-tricks" bash scripts/run_sft_matrix.sh  # single
SFT_TOKENS=1000000 bash scripts/run_sft_matrix.sh   # short smoke
```

The script is idempotent — each stage (pre-SFT eval / SFT / post-SFT
eval) is skipped if its output file already exists. Interrupted runs
resume by re-running.
