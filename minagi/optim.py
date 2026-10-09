"""
A meter for how much of a gradient is signal, and gradient clipping that is
fast on Apple GPUs.
"""

import torch


def _norm_mps(g):
    """The L2 norm of one gradient, by a route MPS computes quickly.

    MPS reduces a whole large tensor to one number pathologically slowly:
    torch.linalg.vector_norm over an expert's [32, 2048, 512] gradient took
    46 ms on an M5, and _foreach_norm, which clip_grad_norm_ uses, the same.
    The norms of its rows of 4,096, and then the norm of those, is the same
    number in 1.3 ms. Small tensors take the ordinary path.
    """
    n = g.numel()
    if n >= 1 << 16:
        w = 4096 if n % 4096 == 0 else (g.shape[-1] if g.dim() >= 2 else 0)
        if w:
            return torch.linalg.vector_norm(g.reshape(-1, w), dim=1,
                                            dtype=torch.float32).norm()
    return torch.linalg.vector_norm(g, dtype=torch.float32)


@torch.no_grad()
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """
    torch.nn.utils.clip_grad_norm_, the same contract: scale every gradient
    so their joint L2 norm is at most `max_norm`, return that norm before
    clipping. Where the gradients are on MPS the norm is taken by _norm_mps,
    about 35x faster there; everywhere else this is PyTorch's own function.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    if (not params or float(norm_type) != 2.0
            or any(p.grad.device.type != "mps" for p in params)):
        return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)
    grads = [p.grad for p in params]
    total = torch.linalg.vector_norm(torch.stack([_norm_mps(g) for g in grads]))
    coef = (float(max_norm) / (total + 1e-6)).clamp(max=1.0)
    by_dtype = {}
    for g in grads:
        by_dtype.setdefault(g.dtype, []).append(g)
    for dt, gs in by_dtype.items():
        torch._foreach_mul_(gs, coef.to(dt))
    return total


class GradSNR:
    """
    How much of the gradient is signal, measured every step.

    The held-out signal this project steers the learning rate by moves 0.0014
    per evaluation against 0.018 of noise, so it takes over a hundred readings
    to say anything - twenty hours at a ten-minute checkpoint. This is the
    same question asked of a quantity that is available on every step.

    Keep an average of the gradient and an average of its squared norm. If
    successive gradients agree, the average keeps its length and the ratio
    ||mean||^2 / mean(||g||^2) approaches one. If they are independent noise
    the average shrinks toward zero and so does the ratio. It is the gradient
    noise scale of McCandlish et al. 2018, in the cheapest form that answers
    the question: two scalars, no extra tensors.

    Reported, not acted on. A signal is watched for a while before anything is
    allowed to steer on it.
    """

    def __init__(self, beta=0.98):
        self.beta = beta
        self.m = None
        self.sq = 0.0
        self.n = 0

    @torch.no_grad()
    def observe(self, params):
        gs = [p.grad for p in params if p.grad is not None]
        if not gs:
            return None
        flat = torch.cat([g.detach().float().reshape(-1) for g in gs])
        self.m = flat.clone() if self.m is None else \
            self.m.mul_(self.beta).add_(flat, alpha=1 - self.beta)
        s = float((flat * flat).sum())
        self.sq = s if self.n == 0 else self.beta * self.sq + (1 - self.beta) * s
        self.n += 1
        return self.ratio()

    def ratio(self):
        """0 = pure noise, 1 = every step pointing the same way."""
        if self.m is None or self.sq <= 0 or self.n < 8:
            return None
        c = 1 - self.beta ** self.n                  # bias correction
        return float((self.m / c).pow(2).sum() / (self.sq / c))
