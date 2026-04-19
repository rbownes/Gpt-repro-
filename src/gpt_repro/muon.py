"""Muon optimizer: momentum → orthogonalize via Newton-Schulz → step.

Reference:
- Keller Jordan, "Muon: A practical optimizer for transformer hidden weights"
  https://kellerjordan.github.io/posts/muon/
- Liu et al., "Muon is Scalable for LLM Training" (arXiv 2502.16982)

Only applies to 2-D weight matrices. Embeddings, norms, biases stay on AdamW.
"""

from __future__ import annotations

import torch
from torch import Tensor


# Quintic Newton-Schulz iteration coefficients tuned to push all singular
# values of the input toward 1 within ~5 iterations.
_NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Approximate `U @ V.T` where `U S V.T = G` is the SVD of G.

    Equivalent to "zero-power" (S → I) applied to the matrix G. Runs in bf16
    internally for throughput; the returned tensor is cast back to G's dtype.
    """
    assert G.ndim >= 2, f"Muon Newton-Schulz requires 2-D+ tensors; got {tuple(G.shape)}"
    a, b, c = _NS_COEFFS
    orig_dtype = G.dtype
    X = G.to(torch.bfloat16)
    # Operate on the "tall" side so X @ X.mT is the smaller matrix.
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(orig_dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2-D weight matrices.

    Args:
        params: iterable of parameters. All must have `ndim >= 2`.
        lr: peak learning rate. Typical range 0.01 – 0.05.
        momentum: heavy-ball momentum coefficient.
        nesterov: if True, apply Nesterov-style lookahead before orthogonalisation.
        ns_steps: Newton-Schulz iteration count.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                assert p.ndim == 2, (
                    f"Muon expects 2-D params; got {tuple(p.shape)}. "
                    "Use build_dual_optimizer to split embeddings/norms off to AdamW."
                )
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Scale so update magnitude is invariant to the matrix aspect ratio.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.data.add_(update, alpha=-lr * scale)
        return loss
