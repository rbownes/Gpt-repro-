"""GRPO loss for exp/15.

Implements the smallest-possible-correct DeepSeekMath / R1-style GRPO:
group-relative advantages, PPO-clipped policy gradient *without* a
value head, plus a per-token KL penalty against a frozen reference
policy using DeepSeek's k3 estimator.

Inputs are a flat list of `Rollout` objects already collected by
`rollout.generate_group(...)` — `len(rollouts)` must be divisible by
`group_size` (G), with rollouts grouped by *prompt* (consecutive G
rollouts in the list share a prompt for advantage normalisation).

Per-token loss formulation:

    advantage_g = (r - r_g.mean()) / (r_g.std() + ε)            # per group
    ratio_t     = exp(logp_new_t − logp_old_t)
    L_pol_t     = −min(ratio_t · adv, clip(ratio_t, 1±ε_clip) · adv)
    δ_kl_t      = logp_ref_t − logp_new_t
    L_kl_t      = exp(δ_kl_t) − δ_kl_t − 1                      # k3 estimator
    loss        = mean_t(L_pol_t) + kl_coef · mean_t(L_kl_t)

Mean is over generated-token positions only (prompt and pad tokens
are masked out). Groups with zero reward variance contribute nothing
to the policy gradient (advantage zeroed) but still pull through the
KL term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gpt_repro.rollout import Rollout
from gpt_repro.tokenizer import EOT_ID


def compute_grpo_loss(
    policy: torch.nn.Module,
    rollouts: list[Rollout],
    *,
    group_size: int,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
    amp_dtype: torch.dtype = torch.bfloat16,
    var_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute the GRPO loss; returns dict with `loss` (with grad) plus
    detached diagnostics (`policy_loss`, `kl`, `mean_reward`,
    `mean_advantage`, `n_tokens`, `clip_frac`).

    `len(rollouts)` must be divisible by `group_size`. Rollouts are
    expected to be ordered so that consecutive groups of `group_size`
    came from the same prompt (this is what `generate_group` produces).
    """
    assert len(rollouts) > 0, "compute_grpo_loss: no rollouts"
    assert len(rollouts) % group_size == 0, (
        f"len(rollouts)={len(rollouts)} not divisible by group_size={group_size}"
    )

    device = next(policy.parameters()).device
    B = len(rollouts)

    # ---- Group-relative advantages -------------------------------------
    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32, device=device)
    rewards_g = rewards.view(-1, group_size)
    means = rewards_g.mean(dim=1, keepdim=True)
    stds = rewards_g.std(dim=1, keepdim=True)
    adv_g = (rewards_g - means) / (stds + var_eps)
    # Zero out groups with no signal.
    zero_var = stds < var_eps
    adv_g = torch.where(zero_var, torch.zeros_like(adv_g), adv_g)
    adv = adv_g.view(-1)                     # (B,)

    # ---- Build padded batch --------------------------------------------
    seqs = [r.prompt_ids + r.gen_ids for r in rollouts]
    prompt_lens = [len(r.prompt_ids) for r in rollouts]
    gen_lens = [len(r.gen_ids) for r in rollouts]
    max_len = max(len(s) for s in seqs)

    inp_full = torch.full((B, max_len), EOT_ID, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        inp_full[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    # forward over inp[:, :-1] → predict tgt = inp[:, 1:].
    # Pass `tgt` so the model returns full-sequence logits (without targets
    # it returns only the last-position logit — the inference fast-path).
    # `.contiguous()` because B>1 col-slicing yields a non-contiguous view
    # and `model.forward` does `targets.view(-1)` internally.
    inp = inp_full[:, :-1].contiguous()
    tgt = inp_full[:, 1:].contiguous()
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        logits, _ = policy(inp, tgt)
    # Compute log P(tgt | inp) at every position WITHOUT materialising the
    # full (B, T-1, V) log-softmax tensor — that intermediate is several GB
    # at our prompt lengths and trips OOM. logsumexp is max-shifted so it's
    # stable in bf16; we only upcast the scalar-per-position result.
    gathered = logits.gather(-1, tgt[:, :, None]).squeeze(-1)  # (B, T-1)
    lse = torch.logsumexp(logits, dim=-1)                       # (B, T-1)
    logp_new = (gathered - lse).float()                         # (B, T-1) fp32

    # ---- Mask + scatter old/ref logps ----------------------------------
    T_minus_1 = max_len - 1
    mask = torch.zeros((B, T_minus_1), dtype=torch.float32, device=device)
    logp_old = torch.zeros_like(logp_new)
    logp_ref = torch.zeros_like(logp_new)
    for i, r in enumerate(rollouts):
        # tgt[i, j] = inp_full[i, j+1]. A generated token sits at
        # inp_full[i, prompt_lens[i] .. prompt_lens[i] + gen_lens[i] - 1],
        # which corresponds to tgt index range [prompt_lens[i] - 1,
        # prompt_lens[i] + gen_lens[i] - 1) — i.e. j ∈ [pL-1, pL+gL-2].
        start = prompt_lens[i] - 1
        end = prompt_lens[i] + gen_lens[i] - 1
        mask[i, start:end] = 1.0
        logp_old[i, start:end] = r.logp_old.to(device, dtype=torch.float32)
        logp_ref[i, start:end] = r.logp_ref.to(device, dtype=torch.float32)

    # ---- Per-token PPO-clipped policy loss -----------------------------
    delta = logp_new - logp_old
    ratio = torch.exp(delta)
    adv_per_tok = adv[:, None].expand_as(ratio)              # (B, T-1)
    unclipped = ratio * adv_per_tok
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_per_tok
    pg_per_tok = -torch.minimum(unclipped, clipped)

    # ---- Per-token KL (k3) ---------------------------------------------
    kl_delta = logp_ref - logp_new
    kl_per_tok = torch.exp(kl_delta) - kl_delta - 1.0

    # ---- Reduce ---------------------------------------------------------
    n_tokens = mask.sum().clamp_min(1.0)
    pg_loss = (pg_per_tok * mask).sum() / n_tokens
    kl_loss = (kl_per_tok * mask).sum() / n_tokens
    loss = pg_loss + kl_coef * kl_loss

    # Diagnostic: fraction of generated tokens where the clipped term was
    # the binding choice (i.e. ratio fell outside [1-ε, 1+ε]).
    clip_frac = (((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float() * mask).sum() / n_tokens

    return {
        "loss": loss,
        "policy_loss": pg_loss.detach(),
        "kl": kl_loss.detach(),
        "mean_reward": rewards.mean().detach(),
        "mean_advantage": adv.detach().abs().mean(),
        "n_tokens": n_tokens.detach(),
        "clip_frac": clip_frac.detach(),
    }
