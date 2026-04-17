# gpt-repro

Faithful GPT-2 124M reproduction on a single RTX 5090, plus a disciplined
experiment harness for layering modern improvements on top of the baseline.

## What this repo is (and isn't)

- **Is:** a paper-faithful GPT-2 small implementation (LayerNorm, learned
  positional embeddings, GELU, tied I/O embeddings) trained from scratch on
  FineWeb-Edu-10B, with a one-experiment-one-diff evaluation protocol for
  measuring architecture/optimizer/kernel improvements.
- **Isn't:** a speedrun fork. The [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
  recipe appears as a *later* experiment so its contribution is attributable.

## Quickstart

```bash
# Install (uv-managed; torch 2.7.1 + CUDA 12.8 wheels for RTX 5090 SM_120)
uv sync

# Fast smoke test: confirms everything imports and one step of training runs
uv run pytest tests/ -m "not slow"
uv run python scripts/train.py --config configs.debug

# HF parity gate (slow, needs network the first time)
uv run pytest tests/test_hf_weight_load.py -m slow

# Prepare FineWeb-Edu-10B (≈ 20 GB on disk, streams from HF Hub)
uv run python scripts/prepare_fineweb_edu.py --out data/fineweb_edu_10B

# Full 124M baseline run (single RTX 5090, ~60–90 min if kernels fire)
uv run python scripts/train.py --config configs.gpt2_124m

# Zero-shot eval once training completes
uv run python scripts/eval_zero_shot.py --ckpt runs/baseline/best_val.pt
```

## Hardware notes — RTX 5090 (SM_120, consumer Blackwell)

**Measured on this repo, 124M model, bs=16, seq=1024, BF16:**

| backend                  | torch.compile | tok/s  | notes |
|--------------------------|:-------------:|-------:|-------|
| `sdpa_flash`             |     ✅ on     | ~182 k | **current default** |
| `sdpa_cudnn`             |      off      | ~139 k | fastest naked kernel [per benchmark](https://gau-nernst.github.io/fa-5090/), but... |
| `sdpa_cudnn` + compile   |       —       |  crash | inductor stride assert on SM_120, torch 2.7.1 |

- **Default is `sdpa_flash` + `torch.compile`.** Even though cuDNN benchmarks
  faster as a [naked kernel](https://gau-nernst.github.io/fa-5090/) (~97 % of
  SOL vs flash's ~91 %), the inductor-generated fake kernel for cuDNN's
  `_scaled_dot_product_cudnn_attention` emits a stride-mismatch assert on SM_120
  under torch.compile. `sdpa_flash` composes cleanly and wins by ~30 % once
  compiled. Revisit on future torch releases; the SDPA backend is a one-line
  config swap.
- **FlashAttention-2** is available as an opt-in backend
  (`attention_backend=flash_attn_2`) via `flash-attn==2.8.0.post2` from the
  [`[flash]` extra](./pyproject.toml). Expected performance on 5090 is similar
  to `sdpa_flash`; useful as a calibration / reproducibility point.
- **FlashAttention-3 / 4 do not support SM_120** yet (FA-4 targets datacenter
  Blackwell / SM_100 only). Track Dao-AILab issues
  [#1987](https://github.com/Dao-AILab/flash-attention/issues/1987) and
  [#2307](https://github.com/Dao-AILab/flash-attention/issues/2307).

## Layout

```
configs/              training configs (one per experiment surface)
src/gpt_repro/        library: model.py, train.py, data.py, eval.py, optim.py, ...
scripts/              CLIs: train.py, prepare_fineweb_edu.py, eval_zero_shot.py
tests/                shape / loader / HF parity tests
experiments/          run log: one subdirectory per experiment, plus TEMPLATE.md
runs/                 training artifacts (gitignored)
data/                 tokenized shards (gitignored)
```

## Experiment protocol

Every improvement is run as a single-variable experiment against the frozen
baseline tag `v0.1-baseline`. Each experiment has its own
`experiments/NN-short-name/report.md` derived from
[`experiments/TEMPLATE.md`](./experiments/TEMPLATE.md), with:

- previous baseline numbers
- the change (one sentence + diff)
- hypothesis + predicted effect + accept/reject criteria (**written before running**)
- implementation notes (gotchas, compile compatibility, HF parity status)
- result (val loss at 1 / 5 / 10 B tokens, tok/s, paper-battery)
- verdict

See [`experiments/README.md`](./experiments/README.md) for the index and the
14-row roadmap.
