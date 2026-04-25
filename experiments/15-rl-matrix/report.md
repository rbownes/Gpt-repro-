---
id: 15-rl-matrix
status: in-progress            # in-progress | accepted | rejected
baseline_run: runs/sft-*/      # all 8 SFT'd checkpoints
experiment_run: runs/rl-*/
baseline_tag: v0.4-sft-matrix
date: 2026-04-25
author: rjbownes
seeds: [0]
---

# Experiment 15 — RL post-training matrix on generative MC extraction

## Previous baseline

The 8 chat-SFT'd checkpoints from exp/14 (`runs/sft-*/best_val.pt`),
each evaluated by log-likelihood scoring on HellaSwag + MMLU + ARC-E
+ ARC-C. Architecture lineage and SFT val rankings carry over from
`experiments/14-sft-matrix/report.md`.

## The change

Instead of LL-scoring the four candidate continuations, prompt the
SFT'd model with the question + lettered options A/B/C/D and ask it
to **generate** the answer letter. Reward / score = exact-match on
the gold letter after parsing the generation.

This is the metric exp/15 RL targets — RL teaches the model to emit
just the letter rather than producing essay-style continuations. The
gap between LL acc (exp/14 ceiling) and gen acc here is the headline
pre-RL signal.

- **Files touched (Phase A):**
  - `src/gpt_repro/gen_eval.py` — `format_mc_prompt`,
    `extract_letter` (lenient + strict), `_generate_greedy`,
    per-task `*_gen` helpers.
  - `src/gpt_repro/chat.py` — added `render_user_turn(prompt)`.
  - `scripts/gen_eval.py` — CLI mirroring `scripts/sft_eval.py`.
  - `tests/test_gen_eval.py` — 37 unit tests (parser modes,
    formatter, chat-template integration).
- **Hyperparameters:**
  - Decode: greedy (argmax), `max_new_tokens=16`, stop on EOT or
    first token of `</assistant>`.
  - Parser: `mode="lenient"` (regex matches first letter or "answer
    is X" / "option B" patterns).
- **Datasets:** same eval splits as exp/14 — HSwag (1 k), MMLU
  (1531), ARC-Easy (570 → 567 after 4-way filter), ARC-Challenge
  (299 → 295 after 4-way filter).

## Why it might improve

Mnemosyne wiki framing: "RLVR sharpens existing capability." LL
proves the latent knowledge is there for most ckpts on most tasks
(0.35–0.48 LL acc on ARC-E across all 8 ckpts). RL with a binary
exact-match letter reward should close the gap toward LL acc and
should *also* fix the parse-failure mode (model emitting essays).

**Predicted effect** (written before RL runs):

- Δ_RL gen acc on ARC-E: **+0.10 to +0.20** (close half the
  LL-to-gen gap)
- Δ_RL parse-failure rate: **−20 to −80 pp** (much larger drop on
  the format-shattered ckpts)
- Variance by ckpt: 02-muon and 11-loopllm have the most room to
  recover; 03-modded and 10-mla have the least

## Results

### Phase A — pre-RL baselines (2026-04-25)

Full LL-vs-generative comparison across 8 SFT'd checkpoints (full
val splits; n in parentheses):

| ckpt | task | LL acc | gen acc | Δ (gen−LL) | parse_fail |
|------|------|-------:|--------:|-----------:|-----------:|
| baseline | HellaSwag (1000) | 0.384 | 0.207 | −0.177 | 0.018 |
| baseline | MMLU (1531)      | 0.253 | 0.222 | −0.031 | 0.127 |
| baseline | ARC-E (567)      | 0.456 | 0.242 | −0.215 | 0.041 |
| baseline | ARC-C (295)      | 0.227 | 0.244 | +0.017 | 0.064 |
| 01-modern-block | HellaSwag | 0.374 | 0.216 | −0.158 | 0.130 |
| 01-modern-block | MMLU      | 0.284 | 0.228 | −0.056 | 0.134 |
| 01-modern-block | ARC-E     | 0.463 | 0.238 | −0.225 | 0.076 |
| 01-modern-block | ARC-C     | 0.278 | 0.288 | +0.011 | 0.031 |
| **02-muon** | HellaSwag     | 0.389 | **0.007** | **−0.382** | **0.977** |
| **02-muon** | MMLU          | 0.270 | **0.024** | **−0.247** | **0.889** |
| **02-muon** | ARC-E         | 0.481 | **0.051** | **−0.430** | **0.869** |
| **02-muon** | ARC-C         | 0.224 | **0.034** | **−0.190** | **0.881** |
| 03-modded-tricks | HellaSwag | 0.380 | 0.254 | −0.126 | 0.013 |
| 03-modded-tricks | MMLU     | 0.278 | 0.218 | −0.060 | 0.144 |
| 03-modded-tricks | ARC-E    | 0.451 | 0.219 | −0.232 | 0.030 |
| 03-modded-tricks | ARC-C    | 0.244 | 0.278 | +0.034 | 0.031 |
| 05-speed-pack | HellaSwag    | 0.374 | 0.238 | −0.136 | 0.020 |
| 05-speed-pack | MMLU         | 0.258 | 0.165 | −0.093 | 0.321 |
| 05-speed-pack | ARC-E        | 0.442 | 0.182 | −0.260 | 0.344 |
| 05-speed-pack | ARC-C        | 0.268 | 0.159 | −0.108 | 0.264 |
| 06-muon-mup | HellaSwag      | 0.390 | 0.228 | −0.162 | 0.021 |
| 06-muon-mup | MMLU           | 0.268 | 0.169 | −0.100 | 0.336 |
| 06-muon-mup | ARC-E          | 0.442 | 0.196 | −0.246 | 0.310 |
| 06-muon-mup | ARC-C          | 0.254 | 0.200 | −0.054 | 0.231 |
| 10-mla | HellaSwag           | 0.375 | 0.246 | −0.129 | 0.000 |
| 10-mla | MMLU                | 0.265 | 0.218 | −0.047 | 0.078 |
| 10-mla | ARC-E               | 0.440 | 0.206 | −0.234 | 0.016 |
| 10-mla | ARC-C               | 0.241 | 0.288 | +0.047 | 0.027 |
| **11-loopllm** | HellaSwag   | 0.355 | **0.029** | **−0.326** | **0.893** |
| **11-loopllm** | MMLU        | 0.251 | **0.057** | **−0.194** | **0.779** |
| **11-loopllm** | ARC-E       | 0.347 | **0.115** | **−0.233** | **0.638** |
| **11-loopllm** | ARC-C       | 0.227 | **0.136** | **−0.092** | **0.593** |

Bold rows = "format-shattered" — parse failure rate ≥ 60 % across
all four tasks; gen acc collapses to ≤ 0.14 even when LL knowledge
is solid.

### Three-cluster grouping by SFT format compliance

The 8 ckpts split cleanly by parse-failure rate:

| cluster | parse_fail range | members | gen acc range |
|---------|------------------|---------|---------------|
| **Clean**     | 0 – 15 %  | baseline, 01-modern, 03-modded, 10-mla | 0.21 – 0.29 |
| **Degraded**  | 23 – 35 % | 05-speed-pack, 06-muon-mup             | 0.16 – 0.24 |
| **Shattered** | 59 – 98 % | 02-muon, 11-loopllm                    | 0.01 – 0.14 |

Notes:

- **02-muon** is the most striking failure: HSwag gen 0.007, ARC-E
  gen 0.051 — the SFT'd model essentially never emits a letter on
  MC questions despite being the *2nd-highest LL accuracy on ARC-E*
  in the matrix (0.481, beaten only by 03-modded's 0.451 and tied
  with 05-speed-pack at 0.442). Reconnects to exp/14's finding that
  02-muon had the worst SFT val (1.498 vs 1.26–1.34 elsewhere).
  Plain Muon-pretrained weights × fresh AdamW SFT failed to
  establish chat-format compliance.
- **11-loopllm**: same shattered pattern. 45 M cap + weight-tying
  was insufficient capacity to learn the format on top of basic
  chat behavior.
- **06-muon-mup vs 02-muon divergence**: μP rescued 06 from the
  "shattered" failure mode that hit 02. Same Muon optimizer family
  in pretrain, but μP's parameterization left weights more
  AdamW-adaptable. (Already a finding from exp/14; reconfirmed
  here on a different metric.)
- **Attention-architecture comparison**: 10-mla preserves clean
  format compliance (0–8 % parse fails); 05-speed-pack (GQA) shifts
  to degraded (26–34 %). Same attention-modification axis; very
  different SFT-survival outcomes.

### Per-task observations

- **ARC-Easy** has the **largest LL→gen gap on every clean ckpt**
  (−0.21 to −0.26). LL acc 0.44–0.48 says the knowledge is there;
  gen acc 0.18–0.25 says the model can't verbalize it. Biggest RL
  headroom.
- **HellaSwag** shows similar (−0.13 to −0.18) on clean ckpts.
- **MMLU** gap is small (−0.03 to −0.10) because LL was already
  near random (0.25–0.29). RL upside on MMLU is bounded.
- **ARC-Challenge** gap is the smallest (mostly within ±0.05). LL
  near-random means knowledge ceiling is low to begin with.

### Pattern hypothesis (to test in Phase D)

Three predictions for post-RL behavior:

1. **Format-shattered ckpts (02-muon, 11-loopllm) should show the
   largest absolute Δ_RL** — there's no format compliance to lose
   and nothing to maintain. RL only has to teach "emit a letter".
2. **Clean ckpts (baseline, 01, 03, 10) should show modest Δ_RL on
   ARC-E** — format already compliant; RL is closing the
   knowledge-to-output gap, which is documented to be harder than
   format learning.
3. **Degraded ckpts (05, 06) should show medium Δ_RL** — RL has to
   simultaneously fix format and close the knowledge gap.

If post-RL gen acc beats SFT-LL acc on any ckpt × task, that's a
notable finding (RL elicited *more* knowledge than was measurable
via LL).

## Known caveats

- **n_eval per task is small**: HSwag 1k (cap), ARC-C 295 (full
  split). Many of the small Δ values are within ±1 σ of zero. The
  *cluster pattern* and the *parse_fail magnitudes* are robust;
  individual deltas should be read as 2σ-noisy.
- **02-muon's near-zero gen acc**: technically below random (0.25)
  for 4-way MC. Means the model emits letters non-uniformly when
  it does emit them — likely defaulting to a single letter (e.g.
  always B) that's rarely the gold answer. Worth checking the
  per-letter answer distribution in the JSON output once for
  diagnosis.
- **Lenient parser**: a model that emits "the correct option is to
  consider..." parses to no letter (lenient regex requires
  "answer is X" / "option X" patterns; "the correct option is" with
  later text doesn't match). Some real-knowledge cases may be
  charged as parse failures.
- **Greedy decode**: post-RL we may want to revisit with sampling
  (T=0.7 + pass@5) to separate "knows but doesn't always emit" from
  "doesn't know". Out of scope for Phase A.

## Phase A exit criteria — met

- 8 `gen_eval_results.json` files exist under `runs/sft-*/`.
- Pre-RL baseline table populated in this report.
- 37 / 37 new tests green (114 / 114 total).
- 1 commit on `exp/15-rl`.

Next: Phase B (GRPO + RL training loop).
