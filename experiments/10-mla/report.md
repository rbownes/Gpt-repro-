---
id: 10-mla
status: in-progress
baseline_run: runs/03-modded-tricks/
experiment_run: runs/10-mla/
baseline_tag: v0.3-exp03
date: 2026-04-22
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

| metric                          | baseline (v0.3) | exp/10 | Δ |
|---------------------------------|---------------:|-------:|---:|
| val loss @ 1 B tokens           | 3.4729         |        |   |
| val loss @ 5 B tokens           | 3.0780         |        |   |
| val loss @ 10 B tokens (best)   | 2.9641         |        |   |
| HellaSwag acc (1 000 examples)  | 0.3780         |        |   |
| tokens / s median               | 178 217        |        |   |
| wall-clock 1 epoch              | 15 h 39 min    |        |   |
| peak VRAM                       | ~14 GB         |        |   |
| model params                    | 123.7 M        | 115.7 M | **−6.5 %** |

- **Seeds:** single seed (0). If result is within ±0.02 of the accept threshold, add seeds 1 & 2 before deciding.
- **Loss curves:** attach `report_assets/loss_curve.png` on completion.

## Verdict

**TBD** — fill in on run completion.
