"""End-to-end GRPO sanity test with a synthetic verifiable task.

Proves the rollout + reward + group-advantage + loss + backward
pipeline is wired correctly without needing a real chat model or
chat template. The task: random tiny GPT emits one token; reward is
1.0 if that token equals GOOD_TOKEN (= 1), else 0.0. Group-relative
advantage discovers GOOD_TOKEN within ~50–100 GRPO steps and the
emission probability rises well above the 1/vocab uniform baseline.

Stays CPU-only and fast (~5 s on a workstation).
"""

from __future__ import annotations

import copy

import pytest
import torch

from gpt_repro.grpo import compute_grpo_loss
from gpt_repro.model import GPT, GPTConfig
from gpt_repro.rollout import Rollout, _sample_one, _score_rollout


VOCAB_SIZE = 32
GOOD_TOKEN = 1
PROMPT_TOKEN = 0


def _make_tiny_gpt(seed: int) -> GPT:
    torch.manual_seed(seed)
    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True,
        tie_embeddings=True,
        attention_backend="sdpa_math",  # CPU-friendly
    )
    return GPT(cfg)


def _collect_rollouts(
    policy: GPT,
    ref: GPT,
    *,
    prompt_ids: list[int],
    n_groups: int,
    group_size: int,
    max_new_tokens: int,
) -> list[Rollout]:
    rollouts: list[Rollout] = []
    for _ in range(n_groups):
        for _ in range(group_size):
            gen_ids = _sample_one(
                policy, prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=None,
                amp_dtype=torch.float32,
                stop_token_ids=(),  # always run to max_new_tokens
            )
            if not gen_ids:
                continue
            logp_old = _score_rollout(policy, prompt_ids, gen_ids, amp_dtype=torch.float32)
            logp_ref = _score_rollout(ref, prompt_ids, gen_ids, amp_dtype=torch.float32)
            rollouts.append(Rollout(
                prompt_ids=list(prompt_ids),
                gen_ids=list(gen_ids),
                logp_old=logp_old,
                logp_ref=logp_ref,
                reward=1.0 if gen_ids[0] == GOOD_TOKEN else 0.0,
            ))
    return rollouts


@torch.no_grad()
def _good_token_rate(policy: GPT, prompt_ids: list[int], n: int = 200) -> float:
    """Empirical P(emit GOOD_TOKEN as first generated token)."""
    hits = 0
    for _ in range(n):
        gen = _sample_one(
            policy, prompt_ids,
            max_new_tokens=1,
            temperature=1.0,
            top_k=None,
            amp_dtype=torch.float32,
            stop_token_ids=(),
        )
        if gen and gen[0] == GOOD_TOKEN:
            hits += 1
    return hits / n


def test_grpo_synthetic_emit_good_token() -> None:
    """GRPO should raise P(GOOD_TOKEN) from ~1/vocab to >= 0.5 within 60 steps.

    We track the best rate observed during training, not the final rate —
    once the model saturates (every rollout in a group hits GOOD_TOKEN),
    advantage collapses to zero, the policy gradient vanishes, and the
    KL penalty alone drifts weights back toward the random reference. In
    production runs `scripts/rl.py` has an early-stop on saturation; the
    test mirrors that by checkpointing the peak.
    """
    policy = _make_tiny_gpt(seed=0)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    prompt_ids = [PROMPT_TOKEN]
    group_size = 8

    base_rate = _good_token_rate(policy, prompt_ids, n=400)
    assert base_rate < 0.20, f"unexpectedly high pre-train rate: {base_rate:.3f}"

    best_rate = base_rate
    for step in range(60):
        rollouts = _collect_rollouts(
            policy, ref,
            prompt_ids=prompt_ids,
            n_groups=2, group_size=group_size,
            max_new_tokens=1,
        )
        if not rollouts:
            continue
        out = compute_grpo_loss(
            policy, rollouts,
            group_size=group_size,
            clip_eps=0.2,
            kl_coef=0.04,
            amp_dtype=torch.float32,
        )
        optim.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()
        assert torch.isfinite(out["loss"]), f"loss diverged at step {step}: {out['loss']}"
        if (step + 1) % 10 == 0:
            r = _good_token_rate(policy, prompt_ids, n=200)
            best_rate = max(best_rate, r)

    assert best_rate >= 0.5, (
        f"GRPO failed to learn the synthetic task: "
        f"best P(GOOD_TOKEN) {base_rate:.3f} -> {best_rate:.3f}"
    )


def test_grpo_loss_skips_zero_variance_groups() -> None:
    """All rewards equal in a group → advantage 0 → policy_loss is ~0
    (KL still applies; only policy_loss is exactly zero)."""
    policy = _make_tiny_gpt(seed=1)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    prompt_ids = [PROMPT_TOKEN]
    rollouts = _collect_rollouts(
        policy, ref,
        prompt_ids=prompt_ids,
        n_groups=1, group_size=4,
        max_new_tokens=1,
    )
    # Force all rewards equal
    for r in rollouts:
        r.reward = 0.5

    out = compute_grpo_loss(
        policy, rollouts,
        group_size=4,
        clip_eps=0.2,
        kl_coef=0.04,
        amp_dtype=torch.float32,
    )
    assert torch.isfinite(out["loss"])
    # zero-variance group → advantage zeroed → policy loss is exactly 0
    assert abs(out["policy_loss"].item()) < 1e-6, (
        f"zero-variance group should zero policy_loss, got {out['policy_loss'].item()}"
    )
    # KL stays at ~0 because policy and ref are identical at step 0 (both
    # were seeded the same and ref is a fresh deepcopy with no updates).
    assert out["kl"].abs().item() < 1e-4, (
        f"KL between identical policy and ref should be ~0, got {out['kl'].item()}"
    )


def test_grpo_loss_assertions() -> None:
    """compute_grpo_loss rejects bad inputs."""
    policy = _make_tiny_gpt(seed=2)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    rollouts = _collect_rollouts(
        policy, ref,
        prompt_ids=[PROMPT_TOKEN],
        n_groups=1, group_size=3,
        max_new_tokens=1,
    )
    # group_size doesn't divide len(rollouts)
    with pytest.raises(AssertionError):
        compute_grpo_loss(policy, rollouts, group_size=2, amp_dtype=torch.float32)
    # empty rollouts
    with pytest.raises(AssertionError):
        compute_grpo_loss(policy, [], group_size=4, amp_dtype=torch.float32)
