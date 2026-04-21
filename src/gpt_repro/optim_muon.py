"""MuonAdamW optimizer — Muon for 2D matrix params, fused AdamW for the rest.

Ported from nanochat/optim.py (single-GPU variant) via the autoresearch branch.
Muon uses Polar Express orthogonalisation (faster than Newton-Schulz at same
iteration count) plus NorMuon variance reduction. AdamW is the standard fused
kernel with 0-D CPU hyperparameter tensors so LR / beta changes don't trigger
torch.compile recompilation.

Both fused kernels are torch.compile'd with dynamic=False, fullgraph=True so
the inner graph is captured once; changing optimizer LR per step only updates
0-D CPU scalars, which live outside the compiled region.

Parameter grouping (done in optim.py's `_build_muon_adamw`):
- Embeddings (`wte`), output head (`lm_head`, tied), all 1-D tensors
  (norms, biases) → AdamW group.
- 2-D block matrices (attn c_attn / c_proj, MLP c_fc / c_proj / w_gate / w_up /
  w_down) → one Muon group per unique shape. Muon requires all params in a
  single group to share a shape because the fused kernel stacks them.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Coefficients for Polar Express orthogonalisation (num_iters=5,
# safety_factor=2e-2, cushion=2). From https://arxiv.org/pdf/2505.16932.
_POLAR_EXPRESS_COEFFS: list[tuple[float, float, float]] = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,    # 0-D CPU tensor, int count
    lr_t: Tensor,      # 0-D CPU tensor
    beta1_t: Tensor,   # 0-D CPU tensor
    beta2_t: Tensor,   # 0-D CPU tensor
    eps_t: Tensor,     # 0-D CPU tensor
    wd_t: Tensor,      # 0-D CPU tensor
) -> None:
    """Fused decoupled AdamW: weight_decay → momentum → bias-correction → update."""
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def _muon_step_fused(
    stacked_grads: Tensor,            # (K, *shape)
    stacked_params: Tensor,           # (K, *shape)
    momentum_buffer: Tensor,          # (K, *shape)
    second_momentum_buffer: Tensor,   # factored second moment, (K, d_major, 1) or (K, 1, d_minor)
    momentum_t: Tensor,               # 0-D CPU tensor, Nesterov momentum
    lr_t: Tensor,                     # 0-D CPU tensor, effective Muon LR
    wd_t: Tensor,                     # 0-D CPU tensor, weight decay
    beta2_t: Tensor,                  # 0-D CPU tensor, variance EMA coeff
    ns_steps: int,                    # Polar Express iteration count
    red_dim: int,                     # reduction axis for variance (-1 or -2)
) -> None:
    """Fused: Nesterov momentum → Polar Express orthogonalisation → NorMuon
    variance reduction → cautious weight decay + update."""
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar Express (bf16 for stability + speed). No float16: dynamic range issues.
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):  # tall matrix
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # wide matrix
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction (per-row or per-column adaptive LR)
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_sz = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_sz
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_sz) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious decoupled weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined Muon (2-D matrices) + AdamW (embeddings + 1-D) optimiser.

    Every `param_group` dict must carry a `kind` key ∈ {'adamw', 'muon'}:

      AdamW groups: `lr`, `betas=(β₁, β₂)`, `eps`, `weight_decay`.
      Muon groups:  `lr`, `momentum`, `ns_steps`, `beta2`, `weight_decay`.
                    All params within one Muon group MUST have the same shape
                    (we stack them for the fused kernel).

    Additionally, each group may carry `base_lr` — the LR scheduler in
    `train.py` multiplies `base_lr` by a fraction in [0, 1] each step and
    writes the result to `group['lr']`.
    """

    def __init__(self, param_groups: list[dict]) -> None:
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors filled per step — stable graph across hyperparam changes.
        _zero = lambda: torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._aw = {k: _zero() for k in ("step", "lr", "beta1", "beta2", "eps", "wd")}
        self._mu = {k: _zero() for k in ("momentum", "lr", "wd", "beta2")}
        # Lazy-compiled variants, populated on first CUDA call. CPU falls back
        # to the eager functions because inductor segfaults on some bf16 CPU
        # lowerings in the Polar Express graph.
        self._adamw_fn = _adamw_step_fused
        self._muon_fn = _muon_step_fused
        self._compile_attempted = False

    def _maybe_compile_for(self, p: torch.Tensor) -> None:
        if self._compile_attempted:
            return
        self._compile_attempted = True
        if p.is_cuda:
            self._adamw_fn = torch.compile(_adamw_step_fused, dynamic=False, fullgraph=True)
            self._muon_fn = torch.compile(_muon_step_fused, dynamic=False, fullgraph=True)

    def _step_adamw(self, group: dict) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            self._maybe_compile_for(p)
            st = self.state[p]
            if not st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            self._aw["step"].fill_(st["step"])
            self._aw["lr"].fill_(group["lr"])
            self._aw["beta1"].fill_(group["betas"][0])
            self._aw["beta2"].fill_(group["betas"][1])
            self._aw["eps"].fill_(group["eps"])
            self._aw["wd"].fill_(group["weight_decay"])
            self._adamw_fn(
                p, p.grad, st["exp_avg"], st["exp_avg_sq"],
                self._aw["step"], self._aw["lr"],
                self._aw["beta1"], self._aw["beta2"],
                self._aw["eps"], self._aw["wd"],
            )

    def _step_muon(self, group: dict) -> None:
        params: list[Tensor] = group["params"]
        if not params:
            return
        p0 = params[0]
        self._maybe_compile_for(p0)
        st = self.state[p0]
        n, shape, device, dtype = len(params), p0.shape, p0.device, p0.dtype
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros(n, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in st:
            if shape[-2] >= shape[-1]:
                sm_shape = (n, shape[-2], 1)
            else:
                sm_shape = (n, 1, shape[-1])
            st["second_momentum_buffer"] = torch.zeros(sm_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        # nanochat's fan-aware Muon LR scaling: tall matrices get a √(d_major/d_minor)
        # boost so the update magnitude matches across shapes.
        lr_scaled = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        self._mu["momentum"].fill_(group["momentum"])
        self._mu["beta2"].fill_(group["beta2"])
        self._mu["lr"].fill_(lr_scaled)
        self._mu["wd"].fill_(group["weight_decay"])
        self._muon_fn(
            stacked_grads, stacked_params,
            st["momentum_buffer"], st["second_momentum_buffer"],
            self._mu["momentum"], self._mu["lr"],
            self._mu["wd"], self._mu["beta2"],
            group["ns_steps"], red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if closure is not None:
            raise NotImplementedError("MuonAdamW does not support closures")
        for g in self.param_groups:
            kind = g.get("kind", "adamw")
            if kind == "adamw":
                self._step_adamw(g)
            elif kind == "muon":
                self._step_muon(g)
            else:
                raise ValueError(f"Unknown optimizer kind: {kind!r}")
        return None
