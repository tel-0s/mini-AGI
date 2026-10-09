#!/usr/bin/env python3
"""
The Metal attention kernel against the blocked path, on the spot.

    python3 tools/metal_attention_check.py            # correctness, then speed
    python3 tools/metal_attention_check.py --quick    # correctness on small shapes only

Correctness: the kernel's output and its gradients for q, k and v, against
blocked_attention computed in fp32, as relative norm errors. The shapes cover
what the model does - a window with no cache, a chunk after a cache - and the
edges a tile kernel can get wrong: lengths that are not a multiple of 64, a
cache longer than the chunk, several batches and heads.

Speed: forward + backward at the model's own sizes, against the blocked path.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                                # noqa: E402

from minagi import model as M                               # noqa: E402
from minagi import metal_attention as MA                    # noqa: E402

DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# what passes: a bf16 or fp16 flash kernel against an fp32 reference is off by
# rounding in its operands - well under 1% - and the model's own probe allows 5%
TOL = {"bf16": 2e-2, "fp16": 5e-3, "fp32": 1e-4}


def rel(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def case(B, H, T, P, dt, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, H, n, 64, generator=g) for n in (T, P + T, P + T))
    w = torch.randn(B, H, T, 64, generator=g)
    dev = torch.device("mps")

    ref_in = [t.to(dev).requires_grad_() for t in (q, k, v)]
    y = M.blocked_attention(*ref_in, P)
    ref = [y] + list(torch.autograd.grad((y * w.to(dev)).sum(), ref_in))

    got_in = [t.to(dev, DT[dt]).requires_grad_() for t in (q, k, v)]
    y = MA.metal_attention(*got_in, P)
    got = [y] + list(torch.autograd.grad((y.float() * w.to(dev)).sum(), got_in))
    errs = [rel(a, b) for a, b in zip(got, ref)]
    finite = all(bool(torch.isfinite(t).all()) for t in got)
    return errs, finite


def correctness(quick):
    shapes = [(1, 2, 256, 0), (1, 2, 200, 0), (1, 2, 160, 96), (1, 1, 130, 1000),
              (2, 3, 100, 37), (1, 2, 64, 0), (1, 1, 1, 63), (1, 2, 65, 0)]
    if not quick:
        shapes += [(1, 8, 4096, 0), (1, 8, 2048, 2048)]
    print("CORRECTNESS  relative error against blocked attention in fp32")
    print(f"  {'B':>2} {'H':>2} {'T':>5} {'P':>5}  {'dtype':5} "
          f"{'out':>9} {'dq':>9} {'dk':>9} {'dv':>9}")
    bad = 0
    for B, H, T, P in shapes:
        for dt in ("bf16", "fp16", "fp32"):
            errs, finite = case(B, H, T, P, dt)
            ok = finite and max(errs) < TOL[dt]
            bad += not ok
            print(f"  {B:>2} {H:>2} {T:>5} {P:>5}  {dt:5} "
                  + " ".join(f"{e:9.2e}" for e in errs)
                  + ("" if ok else ("   NOT FINITE" if not finite else "   FAIL")), flush=True)
    print(f"  {'all pass' if not bad else f'{bad} FAILED'}\n")
    return bad == 0


def timed(fn, reps=10, warm=3):
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
    dev = torch.device("mps")
    print("SPEED  forward + backward, 8 heads of 64, ms (median)")
    print(f"  {'case':28} {'dtype':5} {'blocked':>9} {'metal':>9} {'speedup':>8} {'metal TFLOP/s':>14}")
    for label, T, P in (("reading: 4096 over 4096", 4096, 0),
                        ("cached chunk: 2048 over 4096", 2048, 2048)):
        S = P + T
        # useful work: the causal part only. fwd: QK^T and PV; bwd: those
        # recomputed one each, plus dV, dP, dQ, dK - 7 matmuls of the triangle
        pairs = T * P + T * (T + 1) / 2
        flops = 7 * 2 * pairs * 64 * 8
        for dt in ("bf16", "fp32"):
            q, k, v = (torch.randn(1, 8, n, 64, device=dev, dtype=DT[dt],
                                   requires_grad=True) for n in (T, S, S))
            g = torch.randn(1, 8, T, 64, device=dev, dtype=DT[dt])

            def run(f):
                def go():
                    y = f(q, k, v, P)
                    y.backward(g)
                    q.grad = k.grad = v.grad = None
                return go
            tb = timed(run(M.blocked_attention))
            tm = timed(run(MA.metal_attention))
            print(f"  {label:28} {dt:5} {tb * 1e3:8.1f} {tm * 1e3:8.1f} {tb / tm:7.1f}x "
                  f"{flops / tm / 1e12:13.2f}", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--speed-only", action="store_true")
    a = ap.parse_args()
    if not MA.supported(torch.device("mps"), torch.bfloat16, 64):
        raise SystemExit(f"the Metal kernel is unavailable here: {MA.why_unsupported()}")
    ok = True if a.speed_only else correctness(a.quick)
    if ok and not a.quick:
        speed()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
