# Autoresearch — Phase A (speed)

## Goal

**Minimise `val_bpb` after a fixed 300-second training budget.** Because
training time is fixed, faster kernels → more tokens seen → lower `val_bpb`.
`val_bpb` is therefore the single scalar score; `tok_per_sec` and
`peak_vram_mb` are logged only as diagnostics. Lower is better.

## Setup

1. Confirm you are on a branch named `autoresearch/speed-<YYYY-MM-DD>`. Bail
   if on `master`, `main`, or any `v*` tag. (Phase A is destructive to state;
   it uses `git reset --hard`.)
2. Confirm data + `token_bytes.pt` are ready:
   `uv run python autoresearch/prepare.py`. If it exits non-zero, stop and
   tell the user what to run.
3. If `autoresearch/results.tsv` is a bare header (no baseline row), the
   first thing you do is run the unmodified `train.py` once to establish
   the baseline `val_bpb` and `tok_per_sec`. Log that as row 1 with status
   `baseline`.

## Experimentation

**Editable files:** `autoresearch/train.py` ONLY.

**Frozen (do not touch):**
- `autoresearch/prepare.py` — ground-truth eval path.
- Any constant marked `# FROZEN_ARCH` in `train.py` (vocab size, block size,
  n_layer, n_head, n_embd, mlp_hidden, RoPE base, RMSNorm/QK-Norm/zero-init/
  U-Net/softcap flags). These are quality decisions; moving them is out of
  scope for Phase A.
- The `evaluate_bpb` function — its math is the metric definition.

**In scope (everything else):**
- `MICRO_BATCH` and `GRAD_ACCUM` — try larger micro-batches with smaller
  grad-accum if VRAM allows.
- `COMPILE`, `COMPILE_MODE` — `max-autotune-no-cudagraphs` is the current
  default (measured +6 %). Try `reduce-overhead`, CUDA graphs variants, or
  `max-autotune` proper.
- `ATTENTION_BACKEND` — try every value. `sdpa_flash` works today; `sdpa_cudnn`
  broke previously under compile on SM_120 but might work now; `flash_attn_2`
  is available if `flash-attn==2.8.0.post2` is in the venv.
- `PEAK_LR`, `WARMUP_FRAC`, `MIN_LR_RATIO`, `BETA1`, `BETA2`, `EPS`,
  `WEIGHT_DECAY`, `GRAD_CLIP` — tune for the fixed 300-s regime.
- Any reimplementation of the training loop, loss accumulation, data loading,
  etc. Examples: pinned-memory H2D prefetch, async CUDA streams, fused
  optimizer step via `torch.compile`.
- Kernel routing for specific ops (RMSNorm, cross-entropy, GELU/ReLU²) via
  Liger-Kernel or similar — but note Liger + compile is known-broken on
  SM_120 (see "What NOT to repeat" below).

**Hard constraints:**
- Must remain a **single file** under 1000 lines.
- No new pyproject dependencies. (liger-kernel, flash-attn, transformer-engine
  are already installable via the repo's extras.)
- Peak VRAM ≤ 28 GB (5090 has 32 GB, leave headroom).
- The final summary block must still print `val_bpb:`, `tok_per_sec:`,
  `peak_vram_mb:`, `num_steps:`, `training_seconds:`, `total_seconds:` — the
  loop greps these.

**What NOT to repeat** (learned from this repo's full-run experiments, do not
burn 5 minutes re-discovering):
- **FP8 via TransformerEngine at 124 M**: rejected. TE per-call Python
  overhead dominates at `d_model=768`; measured −13 % tok/s vs BF16 on
  SM_120. See `experiments/04-fp8/report.md`. Revisit only at ≥ 350 M.
- **Liger fused CE + `torch.compile`**: crashes with `CUDA error: misaligned
  address` on SM_120. Either disable compile (−46 % tok/s in exp/05) or
  skip Liger.
- **GQA (`N_KV_HEAD` < `N_HEAD`)**: costs val_bpb at 124 M (+0.028 in exp/05).
  This is a FROZEN_ARCH constraint; don't touch it.
- **Dropping `LOGIT_SOFTCAP`**: FROZEN_ARCH.
- **Datatype tricks** that change eval (e.g. BF16 logits into cross-entropy):
  `evaluate_bpb` already casts logits to float for the NLL; leave that.

## The experiment loop

After the baseline row is recorded, enter the loop:

1. `git status` / `git log --oneline -3` to confirm branch state.
2. Think: pick ONE change from the in-scope list. Write a one-line
   description of the hypothesis ("fuse optimizer step via torch.compile").
3. Edit `autoresearch/train.py`.
4. `git add autoresearch/train.py && git commit -m "trial: <desc>"`.
5. `uv run python autoresearch/train.py > autoresearch/run.log 2>&1`.
6. `grep "^val_bpb:\|^tok_per_sec:\|^peak_vram_mb:\|^num_steps:\|^training_seconds:" autoresearch/run.log`.
7. If the run crashed: `tail -n 60 autoresearch/run.log` — if the fix is
   obvious (import, typo, dtype), fix and re-run once, else skip. Record as
   `crash` in `results.tsv`.
8. Parse `val_bpb` from the run log. Compare to the running best `val_bpb`
   on this branch.
   - If **improved** (lower val_bpb): keep the trial commit. Append a
     `keep` row to `results.tsv` and commit it (`results: keep <desc>`).
     New best.
   - If **not improved** (equal or higher): `git reset --hard HEAD~1` to
     drop the trial commit, then append a `discard` row to `results.tsv`
     and commit (`results: discard <desc>`).
9. **NEVER STOP.** Do not pause to ask. Continue indefinitely until the user
   explicitly halts. Target ~12 trials/hour, ~100 overnight.

## `results.tsv` format

Tab-separated, tracked in git, one row per experiment:

```
commit	val_bpb	tok_per_sec	peak_vram_mb	status	description
```

- `commit`: short SHA of the trial commit (7 chars).
- `val_bpb`: float to 6 decimals. Use `0.000000` for crash rows.
- `tok_per_sec`: float to 1 decimal.
- `peak_vram_mb`: float to 1 decimal.
- `status`: `baseline` | `keep` | `discard` | `crash`.
- `description`: one-line natural-language summary. No tabs.

## Kill signals

Halt the loop and tell the user if any of these fire:

- 5 consecutive crash rows.
- A trial reports `tok_per_sec` > 250 k and `val_bpb` is no better than
  baseline by > 0.05 — agent has probably cheated (shortened training,
  bypassed eval). Report and ask for review.
- `peak_vram_mb` approaches 30000 repeatedly — VRAM pressure is real;
  stop before crashing the GPU.
- Branch already has > 200 trial commits and no `keep` in last 40 trials —
  plateaued, escalate to user.
