# Autoresearch (gpt-repro adaptation)

Karpathy-style autoresearch loop for this project: an AI agent iteratively
edits `autoresearch/train.py` under a 5-minute-per-trial budget, keeping only
changes that improve the scalar metric. This is a **research lab notebook**
— it lives on its own branch and never merges to `master`. Winning changes
are hand-promoted to `experiments/NN-*` full-run experiments under the
pre-declared-hypothesis discipline used elsewhere in this repo.

## Kickoff

Start a new run by telling Claude Code:

> *Have a look at `autoresearch/program.md` and kick off a new autoresearch
> run on this branch.*

The agent will:
1. Read `program.md` + `train.py` + `prepare.py` + current `results.tsv`.
2. Verify the active branch is `autoresearch/speed-<date>` (Phase A) or
   `autoresearch/quality-<date>` (Phase B).
3. Run `uv run python autoresearch/prepare.py` once to confirm data + cache
   `token_bytes.pt`.
4. Run the unmodified `train.py` to establish baseline `val_bpb` + `tok_per_sec`.
5. Enter the edit → commit → run → grep → keep/reset loop. Never stops.

## Phases

**Phase A — speed.** 300-s wall-clock budget per trial, `val_bpb` as the
scalar score. In-scope: kernels, compile modes, data-loader, optimizer
fusion, attention backend. FROZEN_ARCH constants in `train.py` cannot be
touched — architecture decisions belong to Phase B. See `program.md`.

**Phase B — quality.** 1 B-token budget per trial (~80 min at 210 k tok/s),
primary metric `val_bpb` + secondary `hellaswag_acc`. In-scope: architecture
variants, optimizer, training schedule. Run by branching off the Phase A
tip commit (so all speed wins carry forward) and rewriting `program.md`
for the quality brief.

## Files

- `train.py` — single editable file, ~500 lines. Only file the agent changes.
- `prepare.py` — frozen. Verifies shards and builds `token_bytes.pt` cache.
- `program.md` — natural-language spec. Human edits between phases. Agent
  reads but never writes.
- `results.tsv` — per-trial log. Tracked in git. One row per trial.
- `run.log` — stdout of last trial. Gitignored. Parsed by grep for results.
- `README.md` — this file.

## Single-run smoke

Before entering the loop, you can run one trial manually to confirm
everything works:

```
uv run python autoresearch/prepare.py
uv run python autoresearch/train.py > autoresearch/run.log 2>&1
grep "^val_bpb:\|^tok_per_sec:\|^peak_vram_mb:" autoresearch/run.log
```

Expected: ~300-s wall-clock, prints a structured summary block at the end,
no CUDA errors.

## Promotion to full-run experiment

If an autoresearch branch produces a clearly better `train.py`, don't merge
it. Instead:

1. Read the `train.py` tip-of-branch; identify the meaningful changes.
2. Port them into `src/gpt_repro/` behind flags, following the existing
   experiment discipline (pre-run hypothesis, accept/reject criteria, full
   10 B-token run).
3. Document that experiment in a new `experiments/NN-*/report.md`.
4. The autoresearch branch stays untouched — it's the search log.

## Why `train.py` isn't just the project code

Karpathy's design is deliberate: **one file, one metric, 5-minute trials**.
A multi-file setup would give the agent too many moving pieces and too much
context to read per trial. Flattening forces tight focus. The price is
mild code duplication — we accept it.
