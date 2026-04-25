"""GRPO RL post-training of a chat-SFT'd checkpoint on generative MC.

Usage:
    uv run python scripts/rl.py \\
        --pretrain-ckpt runs/sft-03-modded-tricks/best_val.pt \\
        --run-dir runs/rl-03-modded-tricks \\
        --n-steps 500

Loads the SFT'd policy, clones it as a frozen reference, then for each
step samples G generations per prompt across N prompts, computes the
binary letter-match reward, applies GRPO loss + KL-to-ref, and AdamW
steps the policy. Early-stops if mean reward saturates above
`--saturation-threshold` for K consecutive steps (prevents the
post-saturation KL drift seen in the synthetic test).

Output:
    runs/rl-{id}/best_val.pt          — checkpoint at peak held-out reward
    runs/rl-{id}/metrics.jsonl        — per-step diagnostics
    runs/rl-{id}/config.json          — full hparams + run metadata
    runs/rl-{id}/env.json             — env fingerprint
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt_repro.chat import render_user_turn  # noqa: E402
from gpt_repro.gen_eval import format_mc_prompt, reward_mc  # noqa: E402
from gpt_repro.grpo import compute_grpo_loss  # noqa: E402
from gpt_repro.model import GPT, select_sdpa_backend_globally  # noqa: E402
from gpt_repro.rl_data import MCMixture  # noqa: E402
from gpt_repro.rollout import generate_group  # noqa: E402
from gpt_repro.tokenizer import EOT_ID, decode, get_encoding  # noqa: E402
from gpt_repro.utils import (  # noqa: E402
    JSONLLogger, autocast_dtype, device_str, ensure_dir, env_fingerprint,
    human_time, load_gpt_config_from_ckpt, save_checkpoint, seed_everything,
    tune_pytorch_globals, write_env_fingerprint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrain-ckpt", required=True, type=str,
                   help="path to SFT'd checkpoint to use as RL starting policy")
    p.add_argument("--run-dir", required=True, type=str)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0,
                   help="0 = no top-k (full vocab sampling)")
    # GRPO
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.04)
    # Optimizer
    p.add_argument("--peak-lr", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Eval
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-prompts", type=int, default=64,
                   help="held-out MC prompts to score reward on")
    # Early stop
    p.add_argument("--saturation-threshold", type=float, default=0.95,
                   help="mean reward above this for `--saturation-patience` "
                        "consecutive evals triggers early stop")
    p.add_argument("--saturation-patience", type=int, default=3)
    # Misc
    p.add_argument("--reward-strict", action="store_true",
                   help="use strict letter parser for reward (default: lenient)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--mmlu-aux-limit", type=int, default=5_000)
    p.add_argument("--hellaswag-limit", type=int, default=5_000)
    return p.parse_args()


def lr_at_step(step: int, *, peak_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then constant. RL runs are short; no warmdown."""
    if step < warmup:
        return peak_lr * (step + 1) / max(warmup, 1)
    return peak_lr


def load_policy_and_ref(ckpt_path: Path, device: str) -> tuple[GPT, GPT]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_gpt_config_from_ckpt(ck)
    if device == "cuda":
        select_sdpa_backend_globally(cfg.attention_backend)
    policy = GPT(cfg).to(device)
    missing, unexpected = policy.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict: missing={missing[:5]} unexpected={unexpected[:5]}")
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return policy, ref


@torch.no_grad()
def quick_eval_reward(
    policy: GPT, ref: GPT, examples: list, *,
    amp_dtype: torch.dtype, max_new_tokens: int, strict: bool,
) -> float:
    """Greedy eval: for each example, sample ONE generation and score reward.

    Uses temperature=1.0 + top_k=1 (effective greedy) so the eval is
    deterministic given current weights. Returns mean reward.
    """
    enc = get_encoding()
    from gpt_repro.chat import ASSISTANT_CLOSE
    from gpt_repro.rollout import _sample_one
    stop = (EOT_ID, enc.encode_ordinary(ASSISTANT_CLOSE)[0])
    rewards = []
    for ex in examples:
        prompt = format_mc_prompt(ex.question, ex.choices)
        ids = render_user_turn(prompt)
        gen = _sample_one(
            policy, ids, max_new_tokens=max_new_tokens,
            temperature=1.0, top_k=1, amp_dtype=amp_dtype, stop_token_ids=stop,
        )
        text = decode(gen) if gen else ""
        rewards.append(reward_mc(text, ex.gold_letter, strict=strict))
    return float(np.mean(rewards)) if rewards else float("nan")


def main() -> int:
    args = parse_args()
    tune_pytorch_globals()
    seed_everything(args.seed)
    device = device_str()
    amp = autocast_dtype()

    # --- Load policy + ref ---------------------------------------------
    pretrain_path = Path(args.pretrain_ckpt)
    print(f"loading SFT'd policy from {pretrain_path}")
    policy, ref = load_policy_and_ref(pretrain_path, device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: {n_params:,} params | device: {device} | amp: {amp}")

    # --- Data ----------------------------------------------------------
    print("loading MC train mixture (ARC-E + ARC-C + MMLU-aux + HSwag-train)...")
    train = MCMixture(
        seed=args.seed,
        mmlu_aux_limit=args.mmlu_aux_limit,
        hellaswag_limit=args.hellaswag_limit,
    )
    print(f"train mixture: {len(train):,} examples")
    train_iter = iter(train)
    eval_examples = train.take(args.eval_prompts)

    # --- Optimizer (decay on 2-D params, no decay on 1-D) --------------
    decay, no_decay = [], []
    for _, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    try:
        optim = torch.optim.AdamW(groups, lr=args.peak_lr,
                                   betas=(args.beta1, args.beta2),
                                   eps=args.eps, fused=True)
    except (RuntimeError, TypeError):
        optim = torch.optim.AdamW(groups, lr=args.peak_lr,
                                   betas=(args.beta1, args.beta2),
                                   eps=args.eps)

    # --- Run dir + logging ---------------------------------------------
    run_dir = ensure_dir(args.run_dir)
    cfg = policy.cfg if hasattr(policy, "cfg") else None
    (run_dir / "config.json").write_text(json.dumps({
        "rl": vars(args),
        "model": cfg.__dict__ if cfg else None,
    }, indent=2, default=str))
    write_env_fingerprint(run_dir / "env.json")
    log = JSONLLogger(run_dir / "metrics.jsonl")
    print(f"env: {env_fingerprint()}")

    # --- Stop tokens for rollout ---------------------------------------
    enc = get_encoding()
    from gpt_repro.chat import ASSISTANT_CLOSE
    stop_ids = (EOT_ID, enc.encode_ordinary(ASSISTANT_CLOSE)[0])

    # --- Train loop -----------------------------------------------------
    best_reward = -float("inf")
    sat_streak = 0
    t_start = time.monotonic()

    for step in range(args.n_steps):
        # LR schedule
        lr = lr_at_step(step, peak_lr=args.peak_lr,
                        warmup=args.warmup_steps, total=args.n_steps)
        for g in optim.param_groups:
            g["lr"] = lr

        # Collect rollouts: G samples per each of P prompts
        rollouts = []
        for _ in range(args.prompts_per_step):
            ex = next(train_iter)
            prompt = format_mc_prompt(ex.question, ex.choices)
            prompt_ids = render_user_turn(prompt)
            group = generate_group(
                policy, ref, prompt_ids,
                n_samples=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                amp_dtype=amp,
                stop_token_ids=stop_ids,
            )
            # Compute rewards for this group
            for r in group:
                text = decode(r.gen_ids)
                r.reward = reward_mc(text, ex.gold_letter, strict=args.reward_strict)
                r.source = ex.source
            # Drop the whole group if any rollout was empty (rare, kept for
            # safety) — keeps len(rollouts) divisible by group_size.
            if len(group) == args.group_size:
                rollouts.extend(group)

        if not rollouts:
            print(f"step {step}: no rollouts collected, skipping")
            continue

        # GRPO loss + step
        out = compute_grpo_loss(
            policy, rollouts,
            group_size=args.group_size,
            clip_eps=args.clip_eps,
            kl_coef=args.kl_coef,
            amp_dtype=amp,
        )
        optim.zero_grad(set_to_none=True)
        out["loss"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optim.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            log.log({
                "event": "train_step", "step": step + 1,
                "loss": float(out["loss"].item()),
                "policy_loss": float(out["policy_loss"].item()),
                "kl": float(out["kl"].item()),
                "mean_reward_train": float(out["mean_reward"].item()),
                "mean_advantage": float(out["mean_advantage"].item()),
                "clip_frac": float(out["clip_frac"].item()),
                "lr": lr,
                "n_rollouts": len(rollouts),
                "elapsed_s": time.monotonic() - t_start,
            })

        # Held-out eval
        if (step + 1) % args.eval_every == 0 or (step + 1) == args.n_steps:
            policy_was_training = policy.training
            policy.eval()
            eval_r = quick_eval_reward(
                policy, ref, eval_examples,
                amp_dtype=amp, max_new_tokens=args.max_new_tokens,
                strict=args.reward_strict,
            )
            if policy_was_training:
                policy.train()
            log.log({"event": "eval", "step": step + 1, "eval_reward": eval_r})
            elapsed = time.monotonic() - t_start
            print(
                f"step {step+1:>4}/{args.n_steps} | loss {float(out['loss']):+.4f} "
                f"| pg {float(out['policy_loss']):+.4f} | kl {float(out['kl']):.4f} "
                f"| train_r {float(out['mean_reward']):.3f} | eval_r {eval_r:.3f} "
                f"| lr {lr:.1e} | {human_time(elapsed)}"
            )
            if eval_r > best_reward:
                best_reward = eval_r
                save_checkpoint(
                    run_dir / "best_val.pt",
                    model=policy,
                    optimizer=optim,
                    step=step + 1,
                    config=cfg,
                    extra={"eval_reward": eval_r},
                )
            # Saturation early-stop
            if eval_r >= args.saturation_threshold:
                sat_streak += 1
                if sat_streak >= args.saturation_patience:
                    print(f"  [early-stop] eval_reward >= {args.saturation_threshold} "
                          f"for {sat_streak} consecutive evals — stopping")
                    break
            else:
                sat_streak = 0

    log.log({"event": "done", "best_eval_reward": best_reward,
             "elapsed_s": time.monotonic() - t_start})
    log.close()
    print(f"done. best eval reward: {best_reward:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
