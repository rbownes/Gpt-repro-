"""Rollout collection for GRPO.

For each prompt we sample G generations under the *old* policy
(no-grad, temperature sampling), then in a separate forward pass score
those same `(prompt + gen)` sequences under both the policy (no-grad
log-probs at sample time) and the frozen reference model. The
training-time forward (with grad) lives in `grpo.py`.

Per-token log-prob convention: `logp[i]` is the log probability of the
*generated* token at position i, given everything before it. Length
equals the number of *generated* tokens (not the full sequence).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gpt_repro.tokenizer import EOT_ID


@dataclass
class Rollout:
    """A single sampled rollout.

    Attributes:
        prompt_ids:  the prompt tokens (rendered chat prefix).
        gen_ids:     the generated tokens after the prompt.
        logp_old:    per-token log P(gen[i] | prompt + gen[:i]) under the
                     policy at sample time. Length = len(gen_ids).
        logp_ref:    same shape as `logp_old`, but under the frozen ref.
        reward:      scalar; computed externally from the decoded text.
        source:      task label for logging (e.g. "arc_easy").
    """

    prompt_ids: list[int]
    gen_ids: list[int]
    logp_old: torch.Tensor          # (T_gen,) on CPU, no grad
    logp_ref: torch.Tensor          # (T_gen,) on CPU, no grad
    reward: float
    source: str = ""


@torch.no_grad()
def _sample_one(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    amp_dtype: torch.dtype,
    stop_token_ids: tuple[int, ...],
) -> list[int]:
    """Sample a single rollout under temperature; return generated tokens only."""
    device = next(model.parameters()).device
    block_size = model.cfg.block_size if hasattr(model, "cfg") else 1024
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        ctx = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        with torch.autocast(device_type=ctx.device.type, dtype=amp_dtype):
            logits, _ = model(ctx)
        last = logits[0, -1, :].float() / max(temperature, 1e-8)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(last, min(top_k, last.size(-1)))
            last = torch.where(last < v[-1], torch.full_like(last, -float("inf")), last)
        probs = F.softmax(last, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)
        if next_id in stop_token_ids:
            break
        idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return generated


@torch.no_grad()
def _score_rollout(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    *,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Per-token log P(gen[i] | prompt + gen[:i]) under `model`.

    Single forward over the full `(prompt + gen)` sequence, then index
    into the rows that are predicting `gen` tokens. Length = len(gen_ids).
    """
    if not gen_ids:
        return torch.empty(0)
    device = next(model.parameters()).device
    full = torch.tensor([prompt_ids + gen_ids], dtype=torch.long, device=device)
    inp = full[:, :-1]
    tgt = full[:, 1:]
    with torch.autocast(device_type=inp.device.type, dtype=amp_dtype):
        logits, _ = model(inp)
    logp = F.log_softmax(logits.float(), dim=-1)            # (1, T-1, V)
    n_gen = len(gen_ids)
    gen_tgt = tgt[0, -n_gen:]                                # (n_gen,)
    gen_logp_full = logp[0, -n_gen:, :]                      # (n_gen, V)
    return gen_logp_full.gather(-1, gen_tgt[:, None]).squeeze(-1).cpu()


def generate_group(
    policy: torch.nn.Module,
    ref: torch.nn.Module,
    prompt_ids: list[int],
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    stop_token_ids: tuple[int, ...] = (EOT_ID,),
) -> list[Rollout]:
    """Sample `n_samples` generations from one prompt under `policy`,
    score them under both `policy` (logp_old) and `ref` (logp_ref).

    Reward is left at 0.0 — the caller fills it in after decoding the
    text and computing whatever task-specific reward applies.
    """
    rollouts: list[Rollout] = []
    for _ in range(n_samples):
        gen_ids = _sample_one(
            policy, prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            amp_dtype=amp_dtype,
            stop_token_ids=stop_token_ids,
        )
        if not gen_ids:
            # Nothing generated (e.g. immediate stop); skip rather than
            # carry a zero-length rollout through the loss.
            continue
        logp_old = _score_rollout(policy, prompt_ids, gen_ids, amp_dtype=amp_dtype)
        logp_ref = _score_rollout(ref,    prompt_ids, gen_ids, amp_dtype=amp_dtype)
        rollouts.append(Rollout(
            prompt_ids=list(prompt_ids),
            gen_ids=list(gen_ids),
            logp_old=logp_old,
            logp_ref=logp_ref,
            reward=0.0,
        ))
    return rollouts
