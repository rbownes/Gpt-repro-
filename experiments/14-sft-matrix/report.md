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

*Filled in by `scripts/sft_matrix_report.py` once the matrix run completes.*

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
