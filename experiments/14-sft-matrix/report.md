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

### Post-SFT

*Filled in by `scripts/sft_matrix_report.py` once the SFT matrix
completes.*

TODO

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
