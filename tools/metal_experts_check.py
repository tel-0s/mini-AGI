#!/usr/bin/env python3
"""
The Metal expert kernels against the dispatch they replace, on the spot.

    python3 tools/metal_experts_check.py            # correctness, then speed
    python3 tools/metal_experts_check.py --quick    # correctness on small shapes

Correctness: the expert outputs, and the gradients for the input and all
three weight stacks, against the same arithmetic done exactly in fp32. The
bf16 dispatch the kernels replace (run_exact) is measured against the same
reference, so the two can be compared: the kernels should be no further
from it. Routing includes an expert nothing was routed to, runs that are
not multiples of 64 and an expert past its capacity.

Speed: forward + backward at the model's sizes, against run_exact under the
checkpointing the model uses.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from torch.utils.checkpoint import checkpoint              # noqa: E402

from minagi import metal_experts as ME                      # noqa: E402

DEV = torch.device("mps")


class StubPool:
    """What the kernels ask of a pool: its slots' weights in bf16, cast once."""

    def __init__(self, w1, w3, w2):
        self.w = (w1, w3, w2)
        self._c = None

    def inference_weights(self, dtype):
        key = (dtype,) + tuple(t._version for t in self.w)
        if self._c is None or self._c[0] != key:
            self._c = (key, tuple(t.detach().to(dtype) for t in self.w))
        return self._c[1]


def routing(N, E, k, cap_factor, seed):
    """Sorted assignments, as PooledMLP builds them: top-k of skewed random
    scores, slot 3 never chosen, capacity enforced by dropping."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(N, E, generator=g) + torch.linspace(1.5, -1.5, E)   # skew
    scores[:, 3] = -1e9
    idx = torch.topk(scores, k, dim=-1).indices
    flat_e = idx.reshape(-1)
    tok = torch.arange(N).repeat_interleave(k)
    order = torch.argsort(flat_e, stable=True)
    e_sorted, t_sorted = flat_e[order], tok[order]
    counts = torch.bincount(e_sorted, minlength=E)
    limit = max(1, int(cap_factor * e_sorted.numel() / E + 0.999))
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(e_sorted.numel()) - starts[e_sorted]
    keep = slot < limit
    e_sorted, t_sorted = e_sorted[keep], t_sorted[keep]
    runs = [(e, min(c, limit)) for e, c in enumerate(counts.tolist()) if c]
    return t_sorted.to(DEV), runs, int((~keep).sum())


def run_exact(src, W1, W3, W2, t_sorted, runs, dt):
    """PooledMLP's exact-size dispatch, as it is in pool.py."""
    xs = src[t_sorted].to(dt)
    w1, w3, w2 = (W.to(dt).unbind(0) for W in (W1, W3, W2))
    outs, start = [], 0
    for e, c in runs:
        xe = xs[start:start + c]
        outs.append((F.silu(xe @ w1[e].t()) * (xe @ w3[e].t())) @ w2[e].t())
        start += c
    return torch.cat(outs).to(src.dtype)


def rel(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def case(N, E, D, Fd, k, cap, seed=0):
    torch.manual_seed(seed)
    flat = torch.randn(N, D, device=DEV)
    W1 = (torch.randn(E, Fd, D, device=DEV) * D ** -0.5)
    W3 = (torch.randn(E, Fd, D, device=DEV) * D ** -0.5)
    W2 = (torch.randn(E, D, Fd, device=DEV) * Fd ** -0.5)
    t_sorted, runs, dropped = routing(N, E, k, cap, seed)
    gY = torch.randn(t_sorted.numel(), D, device=DEV)

    def grads(fn):
        x, a, b, c = (t.clone().requires_grad_() for t in (flat, W1, W3, W2))
        y = fn(x, a, b, c)
        return [y] + list(torch.autograd.grad((y.float() * gY).sum(), (x, a, b, c)))

    ref = grads(lambda x, a, b, c: run_exact(x, a, b, c, t_sorted, runs, torch.float32))
    old = grads(lambda x, a, b, c: run_exact(x, a, b, c, t_sorted, runs, torch.bfloat16))
    new = grads(lambda x, a, b, c: ME.expert_swiglu(x, t_sorted, runs, a, b, c, StubPool(a, b, c)))
    return ([rel(g, r) for g, r in zip(new, ref)], [rel(g, r) for g, r in zip(old, ref)],
            all(bool(torch.isfinite(g).all()) for g in new), dropped, len(runs))


def correctness(quick):
    shapes = [(300, 8, 128, 256, 2, 1.5), (1000, 32, 512, 2048, 8, 1.5),
              (77, 32, 512, 2048, 8, 1.5), (1, 32, 512, 2048, 8, 0)]
    if not quick:
        shapes += [(4096, 32, 512, 2048, 8, 1.5)]
    print("CORRECTNESS  relative error against exact fp32, the kernels / the bf16 path they replace")
    print(f"  {'N':>5} {'E':>3} {'D':>4} {'F':>5} {'k':>2}   "
          + " ".join(f"{n:>17}" for n in ("Y", "d input", "dW1", "dW3", "dW2")))
    bad = 0
    for N, E, D, Fd, k, cap in shapes:
        cap = cap or 1e9
        new, old, finite, dropped, used = case(N, E, D, Fd, k, cap)
        # no worse than the path it replaces, up to a little noise
        ok = finite and all(n <= max(1.25 * o, 2e-3) for n, o in zip(new, old))
        bad += not ok
        print(f"  {N:>5} {E:>3} {D:>4} {Fd:>5} {k:>2}   "
              + " ".join(f"{n:8.1e}/{o:8.1e}" for n, o in zip(new, old))
              + f"   ({used} experts used, {dropped} dropped)"
              + ("" if ok else ("   NOT FINITE" if not finite else "   WORSE")), flush=True)
    print(f"  {'all pass' if not bad else f'{bad} FAILED'}\n")
    return bad == 0


def rows():
    """Several rows on one set of weights, as the recurrence runs them: the
    weight gradients must come back summed, whether by .backward() or by
    autograd.grad, and a graph dropped without a backward must not leave
    its rows waiting."""
    N, E, D, Fd, k, R = 400, 32, 512, 2048, 8, 4
    torch.manual_seed(1)
    W = [torch.randn(E, Fd, D, device=DEV) * D ** -0.5, torch.randn(E, Fd, D, device=DEV) * D ** -0.5,
         torch.randn(E, D, Fd, device=DEV) * Fd ** -0.5]
    xs = [torch.randn(N, D, device=DEV) for _ in range(R)]
    routes = [routing(N, E, k, 1.5, 10 + r) for r in range(R)]
    gYs = [torch.randn(rt[0].numel(), D, device=DEV) for rt in routes]

    def loss(fn, a, b, c):
        return sum((fn(xs[r], rt[0], rt[1], a, b, c).float() * gYs[r]).sum()
                   for r, rt in enumerate(routes))

    exact = lambda x, t, ru, a, b, c: run_exact(x, a, b, c, t, ru, torch.float32)   # noqa: E731
    ref = [w.clone().requires_grad_() for w in W]
    loss(exact, *ref).backward()
    print("ROWS  four rows on one set of weights, relative error against exact fp32")
    ok = True
    for how in ("backward", "autograd.grad"):
        got = [w.clone().requires_grad_() for w in W]
        pool = StubPool(*got)
        kern = lambda x, t, ru, a, b, c: ME.expert_swiglu(x, t, ru, a, b, c, pool)   # noqa: E731
        if how == "backward":
            loss(kern, *got).backward()
            gs = [w.grad for w in got]
        else:
            gs = torch.autograd.grad(loss(kern, *got), got)
        errs = [rel(g, r.grad) for g, r in zip(gs, ref)]
        good = all(e < 1e-2 for e in errs)
        ok &= good
        print(f"  {how:14} dW1 {errs[0]:.1e}  dW3 {errs[1]:.1e}  dW2 {errs[2]:.1e}"
              + ("" if good else "   FAIL"))
    got = [w.clone().requires_grad_() for w in W]
    pool = StubPool(*got)
    y = loss(lambda x, t, ru, a, b, c: ME.expert_swiglu(x, t, ru, a, b, c, pool), *got)
    del y                                          # dropped: no backward
    import gc
    gc.collect()
    left = len(ME._GROUPS)
    ok &= left == 0
    print(f"  a graph dropped without a backward leaves {left} groups waiting"
          + ("" if left == 0 else "   FAIL"))
    print()
    return ok


def timed(fn, reps=7, warm=2):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def speed():
    N, E, D, Fd, k = 3300, 32, 512, 2048, 8       # one row: ~80% of a 4,096 window active
    torch.manual_seed(0)
    flat = torch.randn(N, D, device=DEV, requires_grad=True)
    W = [torch.randn(E, Fd, D, device=DEV, requires_grad=True),
         torch.randn(E, Fd, D, device=DEV, requires_grad=True),
         torch.randn(E, D, Fd, device=DEV, requires_grad=True)]
    t_sorted, runs, _ = routing(N, E, k, 1.5, 0)
    gY = torch.randn(t_sorted.numel(), D, device=DEV)
    pool = StubPool(*W)
    M = t_sorted.numel()
    flops = 2 * M * D * Fd * 3 * 3                 # 3 matmuls forward, 6 backward

    def old():
        y = checkpoint(lambda x, a, b, c: run_exact(x, a, b, c, t_sorted, runs, torch.bfloat16),
                       flat, *W, use_reentrant=False)
        y.backward(gY)

    def new():
        pool._c = None                             # the cast, once per step, counted
        y = ME.expert_swiglu(flat, t_sorted, runs, *W, pool)
        y.backward(gY)

    def new_cached():
        y = ME.expert_swiglu(flat, t_sorted, runs, *W, pool)
        y.backward(gY)

    print(f"SPEED  one row of the expert pool, forward + backward: {M} assignments, "
          f"{E} experts, {D} -> {Fd} -> {D}, ms (median)")
    # Apple GPUs raise their clock under sustained load: a few cold
    # repetitions read the kernels up to 1.5x slow. Warm up long, and take
    # the three in turn so a drift in clock lands on all of them alike.
    for _ in range(8):
        old(), new(), new_cached()
    ts = {f: [] for f in (old, new, new_cached)}
    for _ in range(12):
        for f in ts:
            ts[f].append(timed(f, reps=1, warm=0))
    to, tn, tc = (sorted(v)[len(v) // 2] for v in ts.values())
    print(f"  run_exact, checkpointed (bf16)      {to * 1e3:8.1f}")
    print(f"  Metal kernels, weights cast here    {tn * 1e3:8.1f}   {to / tn:4.1f}x")
    print(f"  Metal kernels, weights already cast {tc * 1e3:8.1f}   {to / tc:4.1f}x   "
          f"{flops / tc / 1e12:5.2f} TFLOP/s  (every row of a step after the first)")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not ME.supported(DEV, 512, 2048):
        raise SystemExit(f"the Metal expert kernels are unavailable here: {ME.why_unsupported()}")
    ok = correctness(a.quick) and rows()
    if ok and not a.quick:
        speed()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
