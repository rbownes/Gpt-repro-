---
id: NN-short-name
status: in-progress            # in-progress | accepted | rejected
baseline_run: runs/baseline/
experiment_run: runs/NN-short-name/
baseline_tag: v0.1-baseline
date: YYYY-MM-DD
author: rjbownes
seeds: [0]                     # add 1, 2 for sub-0.05 loss effects
---

# Experiment NN — <short title>

## Previous baseline

- **Config:** `configs/gpt2_124m.py` @ commit `<sha>`, git tag `v0.1-baseline`.
- **Arch:** faithful GPT-2 124M (LayerNorm, learned pos-emb, GELU).
- **Optimizer:** AdamW (β = 0.9 / 0.95, wd = 0.1, peak LR 6e-4, cosine decay, 715 warmup steps).
- **Data:** FineWeb-Edu-10B (~10 B tokens, 19,073 steps @ 0.5 M tok/step).
- **Baseline metrics** (mean over 3 seeds):
  - val loss @ 10 B tokens: **<fill>**
  - WikiText-103 PPL: **<fill>** · LAMBADA acc: **<fill>** · HellaSwag acc: **<fill>**
  - tokens/s (median): **<fill>** · time-to-val-loss-3.0: **<fill>**

## The change

One sentence: *what is being swapped*.

- **Diff:** `git diff v0.1-baseline..HEAD` — link to PR if applicable.
- **Files touched:** `src/gpt_repro/…`, `configs/…`
- **Hyperparameters introduced / changed:**
  - `<key>`: `<old>` → `<new>`

## Why it might improve

- **Reference:** [paper / blog title](url)
- **Mechanism (one paragraph):** why the change should move quality or speed.
- **Predicted effect (written BEFORE running):**
  - val loss Δ @ 10 B tokens: `<prediction>`
  - tok/s Δ: `<prediction>`
- **Accept criteria:** e.g. "val loss drops ≥ 0.03 *or* tok/s ≥ +10 %, no regression on the other axis > 5 %".
- **Reject / falsification:** e.g. "if val loss is ±0.01 and tok/s Δ < 3 %, reject as noise".

## Implementation notes

- Non-obvious design decisions (e.g. "Muon only on 2D hidden weights; embeddings, LN stay on AdamW").
- Gotchas encountered and how they were resolved (bf16 → fp32 accumulations, compile-breakers, numerical instabilities, etc.).
- Confirmed: `torch.compile` still works / fallback path documented.
- Confirmed: HF weight-load test (`tests/test_hf_weight_load.py`) still passes if applicable (skip if the arch change breaks HF parity by design, e.g. RoPE).

## Result

| metric                    | baseline | experiment | Δ |
|---------------------------|---------:|-----------:|---|
| val loss @ 1 B tokens     |          |            |   |
| val loss @ 5 B tokens     |          |            |   |
| val loss @ 10 B tokens    |          |            |   |
| tokens / s (median)       |          |            |   |
| time to val loss 3.0      |          |            |   |
| wall-clock 1 epoch        |          |            |   |
| WikiText-103 PPL          |          |            |   |
| LAMBADA acc               |          |            |   |
| HellaSwag acc             |          |            |   |

- **Seeds:** list individual seed metrics; don't hide behind mean ± std.
- **Loss curves:** attach `report_assets/loss_curve.png`.
- **Notes on variance:** mention anomalies, kill-early events, restarts.

## Verdict

**<Accept | Reject>** — one sentence matching the result against the accept/reject criteria above.

- If **accepted:** merge to `main`, update `configs/gpt2_124m.py` to make this the new baseline, advance `v0.{N+1}-baseline` tag, update `experiments/README.md` roadmap table.
- If **rejected:** leave the branch and this report with `status: rejected`. Rejected reports are as valuable as accepted ones.
