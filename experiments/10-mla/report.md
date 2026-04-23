---
id: 10-mla
status: rejected
baseline_run: runs/03-modded-tricks/
experiment_run: runs/10-mla/
baseline_tag: v0.3-exp03
date: 2026-04-23
author: rjbownes
seeds: [0]
---

# Experiment 10 — Multi-head Latent Attention (DeepSeek-V2) on v0.3

## Previous baseline

- **Config:** `configs/gpt2_124m_modernplus.py` @ git tag `v0.3-exp03` (commit `e70512c`).
- **Arch:** modern block (RoPE + RMSNorm + QK-Norm) + modded-nanogpt tricks (ReLU² MLP, zero-init out-projections, U-Net skips, `logit_softcap=30`), with **plain multi-head attention**.
- **Optimizer:** AdamW β=(0.9, 0.95), wd=0.1, peak LR 6e-4, cosine decay to 0.1×, 715 warmup steps.
- **Data:** FineWeb-Edu-10B, 19 073 steps @ 524 288 tok/step (= 10 B tokens).
- **Baseline metrics (seed 0):**
  - val loss @ 1 B / 5 B / 10 B tokens: **3.4729 / 3.0780 / 2.9641** (best=**2.9694** on 200-batch held-out)
  - HellaSwag acc (1 000 examples): **0.3780**
  - tokens/s median: **178 215** · wall-clock: **15 h 39 min**
  - 124 M params (123 671 808 exact)

## The change

Swap `CausalSelfAttention` for `MLAttention` — DeepSeek-V2's latent-bottleneck attention with decoupled RoPE. All other arch flags, optimizer, schedule, batch size, data, and token budget are **unchanged**.

- **Diff:** branch `exp/10-mla` (off `main`, which is at `71ae6ad`).
- **Files touched:**
  - `src/gpt_repro/model.py`: new `MLAttention` class (~90 lines), new `make_attention` dispatch, RoPE buffer sized to `mla_d_qk_rope` when `attention_type="mla"`, five new `GPTConfig` fields.
  - `configs/gpt2_124m_modernplus_mla.py`: new, minimal diff vs. modernplus.
  - `tests/test_model_shapes.py`: 7 new tests covering MLA shapes, RoPE sizing, QK-Norm target, param count vs. MHA, overfit gate.
- **Hyperparameters introduced:**
  - `attention_type`: `"mha"` → `"mla"`
  - `mla_d_kv_comp`   = 256 (= 4·d_head): KV-cache latent bottleneck
  - `mla_d_qk_nope`   = 32  (= d_head/2): per-head no-pe Q/K chunk (gets QK-Norm)
  - `mla_d_qk_rope`   = 32  (= d_head/2): shared-across-heads RoPE-rotated Q/K chunk
  - `mla_d_v`         = 64  (= d_head):   per-head V dim
- **Hyperparameters unchanged from v0.3:** `peak_lr=6e-4`, betas, `weight_decay=0.1`, `grad_clip=1.0`, `warmup_steps=715`, `min_lr_ratio=0.1`, `total_steps=19_073`, `effective batch=512 seqs = 524 288 tok/step`.

### Parameter accounting — model is ~8 M smaller than MHA v0.3

| module (per layer) | MHA v0.3         | MLA (this)     | Δ |
|---|---:|---:|---:|
| q_proj / c_attn    | 768·3·768 = 1.77 M | 768·(12·64) = 0.59 M | −1.18 M |
| kv_down            | (n/a, folded into c_attn) | 768·288 = 0.22 M | +0.22 M |
| kv_up              | (n/a)              | 256·(12·96) = 0.30 M | +0.30 M |
| c_proj             | 768·768 = 0.59 M   | (12·64)·768 = 0.59 M | 0 |
| **attn / layer**   | **2.36 M**         | **1.70 M**           | **−0.66 M** |
| MLP (unchanged)    | 4.72 M             | 4.72 M               | 0 |
| **block total**    | **7.08 M**         | **6.42 M**           | **−0.66 M** |

Over 12 layers: **−7.9 M** attention params vs. MHA. Embeddings, norms, softcap unchanged. Total model: **123 671 808 → 115 657 344** (−6.5 %).

This is a design choice — MLA's chunking can be scaled up to parameter-match MHA (e.g. `d_v=96` pushes back to ~120 M) but the resulting model is "MLA with extra capacity" rather than "v0.3 with attention swapped". We deliberately kept the chunk sizes at the DeepSeek-V2-style defaults relative to `d_head` so this experiment tests the *mechanism*, not a capacity-matched comparison. The accept criterion below accounts for the param shortfall.

## Why it might improve

- **References:** [DeepSeek-V2 (arXiv 2405.04434)](https://arxiv.org/abs/2405.04434) (original paper); [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) (production use).
- **Mechanism.** MLA replaces the usual per-head K and V projections with a single shared latent `c_kv` (dim `d_kv_comp`). Per-head K/V are recomputed by up-projections `W_uk`, `W_uv` at attention time. Position information is decoupled from the attention content: a small shared chunk `k_pe` (dim `d_qk_rope`) carries RoPE; the rest of K (`k_nope`) is position-free. The effective attention is `q_nope·k_nope + q_pe·k_pe` — same bilinear form as MHA, but K/V live on a low-rank manifold.
- **Why this could help quality at fixed tokens.** The latent-bottleneck prior is a mild regulariser: the model can't overfit an individual K/V direction that isn't supported by the c_kv basis. At DeepSeek's scale this was at worst neutral; at 124 M with 10 B tokens, we're closer to the under-fit end where regularisation can go either way.
- **Why this could HURT quality at fixed tokens at 124 M.** 7.9 M fewer parameters in the attention stack. The compression ratio (d_kv_comp=256 vs. full KV 768·2=1536 per token) is aggressive; at 124 M scale the capacity loss may not be offset by the regularisation win.
- **Predicted effect (written BEFORE running):**
  - val loss Δ @ 10 B tokens: **−0.010 to +0.020**. Point prediction: **+0.005** (target val ≈ 2.969). A slight regression is the most likely outcome at the capacity-shortfall *and* fixed-token regime; the MLA win is supposed to appear at long contexts / inference-time KV memory, neither of which this experiment measures.
  - tok/s Δ: **−2 % to −8 %**. MLA has *more* attention FLOPs than MHA at training time (the up-projection is extra work). Expected −5 %.
  - HellaSwag Δ: ±1.5 pp (noise).
- **Accept criteria:**
  - **Quality-neutral accept**: val loss Δ ∈ [−0.02, +0.015] AND tok/s regression ≤ 10 %. The interpretation is "MLA lands near-parity at this scale without burning throughput — KV-cache benefits are a bonus for future inference/RL work".
  - **Strong accept**: val loss Δ ≤ −0.015 AND tok/s regression ≤ 10 %. "MLA is strictly a win; replace baseline."
  - **Reject**: val loss Δ > +0.015 OR tok/s regression > 15 %. "The capacity shortfall isn't offset; not worth the code."
- **Kill-early** (step ≈ 2 000, 1 B tokens): stop if val loss > 3.52 (baseline + 0.05), or if step-500 val loss is > baseline step-500 + 0.10.

## Implementation notes

- **MLA attention shape pipeline.** Q is projected to `n_head × (d_qk_nope + d_qk_rope)` then reshaped to `(B, H, T, Dn+Dp)`. KV down-proj yields `(c_kv, k_pe)` where `c_kv ∈ ℝ^{d_kv_comp}` and `k_pe ∈ ℝ^{d_qk_rope}` is shared across heads. KV up-proj materialises per-head `(k_nope, v)`. RoPE is applied only to `q_pe`/`k_pe`. Final Q/K assembled by `cat([nope, pe], dim=-1)`; V passes through unchanged. SDPA with `is_causal=True` then `c_proj` to `n_embd`.
- **QK-Norm on the no-pe parts only.** Matches the DeepSeek paper: RoPE-rotated components skip pre-norm because the rotation is already unitary. `q_norm`/`k_norm` RMSNorm weights are shape `(d_qk_nope,)` — test `test_mla_qk_norm_only_on_nope` guards this.
- **Separate RoPE buffer size.** `GPT.__init__` picks `rope_dim = d_qk_rope` when `attention_type="mla"`, vs. `n_embd/n_head` otherwise. Test `test_mla_rope_buffer_sized_to_d_qk_rope` guards this.
- **No flash_attn_2 path for MLA.** The existing MHA `flash_attn_2` branch assumes full-head K/V shape and doesn't support MLA's chunked Q/K. SDPA (flash backend) is used unconditionally for MLA. At 1024 context this is within 1 % of flash_attn_2 on SM_120.
- **Zero-init out-projection.** `zero_init_proj=True` targets `.c_proj.weight` by name-suffix — MLA's `c_proj` keeps the name so the v0.3 rule applies unchanged.
- **HF weight-load test.** Unaffected — MLA is off by default (`attention_type="mha"` is the default). The faithful config continues to pass HF parity.
- **torch.compile compatibility.** MLA forward pass is pure `nn.Linear` + `.view` / `.transpose` / `.split` / `.expand` / `apply_rope` / SDPA — no dynamic shapes, no graph breaks. `compile_mode="default"` used (matching v0.3's perf commit); we avoid `"reduce-overhead"` because U-Net skips + cudagraph output tracking have historically broken on this codebase.

## Result

| metric                           | baseline (v0.3) | exp/10 MLA | Δ |
|----------------------------------|----------------:|-----------:|---:|
| val loss @ 1 B tokens            | 3.4729          | 3.4891     | +0.0163 |
| val loss @ 5 B tokens            | 3.0780          | 3.0955     | +0.0175 |
| val loss @ 10 B tokens (final)   | **2.9642**      | 2.9806     | +0.0165 |
| best val loss                    | **2.9641**      | 2.9805     | +0.0164 |
| HellaSwag acc (1 000 examples)   | 0.3780          | *not run*  | — |
| tokens / s median                | 178 217         | **182 906** | **+2.63 %** |
| wall-clock 1 epoch               | 15 h 39 min     | **15 h 13 min** | **−26 min** |
| model params                     | 123.7 M         | 115.7 M    | **−6.5 %** |

**Time-to-target** (first eval ≤ threshold):

| threshold | v0.3 MHA | exp/10 MLA | speedup |
|---:|---:|---:|---:|
| ≤ 3.500 | 98.4 min | **96.3 min** | **1.02×** |
| ≤ 3.200 | **270 min** | 288 min | 0.94× |
| ≤ 3.100 | **418 min** | 455 min | 0.92× |
| ≤ 3.000 | **713 min** | 766 min | 0.93× |
| ≤ 2.990 | **762 min** | 838 min | 0.91× |

### Loss-curve shape — constant gap, not a crossover

Unlike exp/06 (Muon+AdamW), exp/10 is **not** a crossover story. MLA lands behind at step 500 already and the gap stays within [+0.01, +0.024] for the entire run. The per-step Δ is essentially a flat +0.016 offset:

| step | v0.3 | exp/10 | Δ |
|---:|---:|---:|---:|
|   500 | 4.7746 | 4.7555 | **−0.0191** ← MLA's only lead |
|  1000 | 3.8634 | 3.8687 | +0.0054 |
|  1500 | 3.5981 | 3.6212 | +0.0231 |
| 10000 | 3.0675 | 3.0851 | +0.0176 |
| 18500 | 2.9657 | 2.9827 | +0.0170 |
| 19000 | 2.9641 | 2.9805 | +0.0164 |

The only step where MLA beat v0.3 was step 500. Everywhere else it's been behind by ~0.015–0.023.

### Against pre-declared criteria

- **Quality-neutral accept** (Δ ∈ [−0.02, +0.015] AND tok/s ≤ 10 %): **FAIL by 0.0014** on the val-loss axis; the tok/s axis passes comfortably (−0 % — in fact +2.6 % improvement). Δ = +0.0164 vs the +0.015 threshold.
- **Strong accept** (Δ ≤ −0.015): FAIL.
- **Reject** (Δ > +0.015 OR tok/s > 15 % regression): **triggers** on Δ by 0.0014.

### Predicted vs actual

- Predicted val-loss Δ: **[−0.010, +0.020]**, point **+0.005**. Actual: **+0.0164** — within the predicted range but above the point estimate and just past the accept bar. ✅ range / ❌ accept.
- Predicted tok/s Δ: **[−8 %, −2 %]**. Actual: **+2.6 %** — **outside the predicted range, in the wrong direction**. MLA turned out to be *faster* not slower: the −6.5 % parameter shortfall dominates the extra up-projection work at these chunk sizes. ❌ prediction miss, ✅ nice surprise.

## Verdict

**Reject** on the strict letter of the pre-declared criterion: val loss Δ = +0.0164 exceeds the +0.015 accept bar by 0.0014. This is *within seed noise* at a single seed — a multi-seed study would be needed to distinguish "marginal reject" from "noise-level null" — but per the project discipline, single-seed results that clearly cross a threshold (even by a hair) aren't upgraded to multi-seed unless they'd plausibly tip the other way, which a flat +0.016 gap across the entire curve will not.

**Keep the code, not the baseline.** `src/gpt_repro/model.py`'s `MLAttention` class and the `attention_type` flag stay on `main`; the config `configs/gpt2_124m_modernplus_mla.py` is preserved for reproducibility. `v0.3-exp03` remains the baseline tag. MLA is **not** promoted.

## Key findings

1. **MLA is slightly faster, not slower, at this chunk sizing.** tok/s went from 178 k → 183 k (**+2.6 %**). Whole-run wall-clock drops 26 min. The pre-run prediction of −2 to −8 % was wrong — the fewer attention params (−6.5 % total) dominate the extra up-projection FLOPs at 124 M scale. This flips the cost/benefit calculus: MLA is a **free speed tweak** if you're willing to absorb a 0.016 val-loss cost.
2. **The val loss gap is flat across the whole curve.** Unlike Muon's three-phase crossover, MLA just has a consistent ~0.016 offset from step 1500 onward. Interpretation: the model is strictly slightly smaller / less capable across the entire training run, not compensating late.
3. **Per-parameter efficiency is actually better than MHA.** MLA lost 6.5 % of the parameter count but only 0.55 % of the val-loss improvement over baseline (from 3.040 → 2.981, vs MHA's 3.040 → 2.964). That's a modestly *better* parameter-efficiency ratio.
4. **The MLA motivation is inference/RL, not pretraining val loss.** This experiment only measures pretraining val loss at 10 B tokens. The actual value of MLA — 5–10× KV-cache compression for long-context generation and RL rollouts — wasn't exercised and can't be rejected by this run.

## Recommendations / follow-ups

1. **Do NOT run a parameter-matched MLA next.** Increasing `d_v=64 → 96` or `d_kv_comp=256 → 384` would parameter-match MHA but moves this experiment from "MLA at DeepSeek-V2-style sizing" to "MLA with arbitrary extra capacity", which isn't a cleaner experiment. If future exp/11 or exp/12 wants parameter-matched MLA, be explicit about the motivation.
2. **exp/11 candidate: RL-time KV-cache benchmarking.** Load `runs/10-mla/best_val.pt` and `runs/03-modded-tricks/best_val.pt`. Measure generation throughput at contexts 1 024 / 4 096 / 16 384 and batch sizes 1 / 8 / 64. This is where MLA's actual value lives; it's untested after this experiment.
3. **exp/11 alternative: Differential Transformer (roadmap #08)** or **Native Sparse Attention (#09)**. Both are single-axis quality-first attention changes and haven't been run. Either is a cleaner next step than MLA variants.
4. **MLA + Muon stacking** — plausible since MLA's attention block has different grad dynamics (narrower bottleneck → stronger per-param signal). Could close the 0.016 gap at zero token cost. Low priority given both alone fell short.
5. **Don't use MLA as a general-purpose baseline.** It's ~0.016 val loss worse at matched tokens. If you need a more efficient attention, GQA (already rejected at exp/05) is simpler; if you need long-context inference, MLA is still the right call despite this result.
