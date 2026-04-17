---
id: 00-baseline
status: in-progress
baseline_run: runs/baseline/
experiment_run: runs/baseline/
baseline_tag: v0.1-baseline
date: TBD
author: rjbownes
seeds: [0]
---

# Experiment 00 — Faithful GPT-2 124M baseline

## Previous baseline

None. This *is* the baseline: a strict Radford et al. 2019 GPT-2 small
reproduction on FineWeb-Edu-10B. Every future experiment is measured against
the numbers this run produces.

## The change

- **Configuration:** `configs/gpt2_124m.py`
- **Architecture:** 12 layers × 768 hidden × 12 heads, 1024 context, 124 M params.
  LayerNorm (pre-norm), learned positional embeddings, GELU (`tanh` approximation),
  tied input/output embeddings, residual-projection init scaled by `1/√(2·n_layer)`.
- **Data:** FineWeb-Edu-10B (`HuggingFaceFW/fineweb-edu` `sample-10BT`),
  tokenised with tiktoken `gpt2`, EOT-separated, sharded as uint16 `.bin`.
- **Optimizer / schedule:** AdamW β = (0.9, 0.95), wd 0.1, eps 1e-8;
  peak LR 6e-4, linear warmup 715 steps, cosine decay to 0.1 × peak,
  grad clip 1.0, effective batch ≈ 0.5 M tokens, total 19 073 steps.
- **Precision / kernels:** BF16 autocast, `torch.compile`, SDPA with cuDNN backend.

## Why it might improve

N/A — this is the reference. See `experiments/README.md` for the roadmap of
improvements layered on top of this run.

## Implementation notes

- Uses `tests/test_hf_weight_load.py` as the architectural correctness gate.
  Do **not** advance `v0.1-baseline` until this test passes.
- Smoke path before the full run:
  1. `pytest tests/` (non-slow tests)
  2. `uv run python scripts/train.py --config configs.debug`
  3. `pytest tests/ -m slow` (HF weight-load parity; requires network)

## Result

| metric                    | value |
|---------------------------|------:|
| val loss @ 1 B tokens     |       |
| val loss @ 5 B tokens     |       |
| val loss @ 10 B tokens    |       |
| tokens / s (median)       |       |
| time to val loss 3.0      |       |
| wall-clock 1 epoch        |       |
| WikiText-103 PPL          |       |
| LAMBADA acc               |       |
| HellaSwag acc             |       |

Target (paper-equivalent ballpark on FineWeb-Edu): val loss ≈ 2.85–3.00,
HellaSwag ≈ 29 %, WikiText-103 PPL ≲ 35.

## Verdict

Pending — fill in after the first full run. Once numbers are in the target
ballpark and `tests/test_hf_weight_load.py` passes, tag `v0.1-baseline`.
