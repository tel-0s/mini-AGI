#!/usr/bin/env python3
"""
Where a learning step's GPU time goes, on an Apple GPU.

    python3 tools/profile_step.py                    # one step at the default window
    python3 tools/profile_step.py --steps 3 --trace step.json
    python3 tools/profile_step.py --window 2048      # the context a fresh run starts at

Builds a model of the real size from config.yaml (random weights, in a
temporary directory, deleted after), warms it up, then profiles whole
learning steps exactly as `train.py read` takes them: a forward over the
window under autocast with the pool's auxiliary and balance losses, the
backward, gradient clipping, and a fused AdamW step. Each step reads
different random text, so the pool pages experts in and out as it does when
the text changes.

Reports GPU time by phase, by module, by operation and by single shape (see
tools/gpuprof.py for how it is measured and attributed). --trace writes
every operation's GPU span as a Chrome trace, for ui.perfetto.dev.

The same steps - the same text, so the same sampled depths - are also run
unprofiled first, so the profiled GPU time can be held against what those
steps really cost. Depth is sampled per step, so steps differ; --steps 3 or
more evens that out.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                                # noqa: E402

from minagi import device as D                              # noqa: E402
from minagi.optim import clip_grad_norm_                    # noqa: E402
from minagi.precision import amp, set_compute_dtype        # noqa: E402


def build(window, experts, precision):
    import train as T
    from minagi.config import get, load
    from minagi.create import create
    c = load()
    tmp = tempfile.mkdtemp(prefix="minagi-profile-")
    create(os.path.join(tmp, "w"), seed=0, verbose=False, experts=experts,
           block=max(window, 512))
    set_compute_dtype(precision)
    dev = torch.device("mps")
    model, cfg, pool, _ = T.build_paged(os.path.join(tmp, "w"), dev, ram_capacity=experts)
    # the depth policy reading runs under, from config.yaml, as train.py sets it
    cfg.train_steps_mean = float(get(c, "model.train_steps_mean", 0.0) or 0.0)
    cfg.min_steps = max(1, min(int(get(c, "model.min_steps", 1) or 1), cfg.max_steps))
    if get(c, "model.bptt_window", None):
        cfg.bptt_window = int(get(c, "model.bptt_window"))
    if get(c, "model.halt_thresh", None):
        cfg.halt_thresh = float(get(c, "model.halt_thresh"))
    cfg.halt_freeze = bool(get(c, "model.halt_freeze", False))
    model.train()
    lr, wd = 3e-4, float(get(c, "training.wd", 0.1) or 0.1)
    trunk = [p for n, p in model.named_parameters() if not n.startswith("pool.")]
    pool_ps = [p for n, p in model.named_parameters() if n.startswith("pool.")]
    opt = torch.optim.AdamW([{"params": trunk, "weight_decay": wd, "lr": lr * 0.1},
                             {"params": pool_ps, "weight_decay": wd, "lr": lr}],
                            lr=lr, betas=(0.9, 0.95), fused=True)
    if hasattr(pool, "attach_optimiser"):
        pool.attach_optimiser(opt)
    return tmp, dev, model, cfg, opt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", type=int, default=4096,
                    help="characters one step forwards (model.context_end)")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "fp16"])
    ap.add_argument("--experts", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1, help="steps profiled")
    ap.add_argument("--top", type=int, default=25, help="rows per table")
    ap.add_argument("--trace", help="write a Chrome trace of the profiled steps here")
    ap.add_argument("--clip", type=float, default=1.0)
    a = ap.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("this profiles an Apple GPU (MPS); none here")

    from gpuprof import GPUProfile, ext               # tools/, beside this script
    ext()                                              # compile before anything is timed
    tmp, dev, model, cfg, opt = build(a.window, a.experts, a.precision)
    try:
        def step(i):
            torch.manual_seed(1000 + i)                # depth and text differ per step
            data = torch.randint(0, 256, (a.window + 1,))
            x = data[:-1].unsqueeze(0).to(dev)
            y = data[1:].unsqueeze(0).to(dev)
            opt.zero_grad(set_to_none=True)
            with amp(dev):
                _, loss = model(x, y)
            if getattr(cfg, "pool_aux", 0):
                loss = loss + cfg.pool_aux * model.pool_aux()
            if hasattr(model, "pool_balance"):
                loss = loss + model.pool_balance()
            loss.backward()
            clip_grad_norm_(model.parameters(), a.clip)
            return loss

        for i in range(a.warmup):
            step(i)
        def depth():
            return float(getattr(model, "last_steps", 0) or 0)

        D.sync(dev)
        t = time.perf_counter()
        plain_depths = []
        for i in range(a.steps):
            step(200 + i)
            opt.step()
            plain_depths.append(depth())
        D.sync(dev)
        plain = (time.perf_counter() - t) / a.steps
        print(f"UNPROFILED  {plain * 1e3:.0f} ms a step - a {a.window}-character window, "
              f"{a.precision}, mean depth {sum(plain_depths) / len(plain_depths):.1f} rows\n",
              flush=True)

        prof_depths = []
        with GPUProfile(model) as prof:
            for i in range(a.steps):
                step(200 + i)                          # the same text and depths as above
                with prof.phase("optimizer"):
                    opt.step()
                prof_depths.append(depth())
        # every step moves the weights, so halting - and depth - drifts a little
        # between the two passes; say so only when it is more than a little
        pd, ud = sum(prof_depths), sum(plain_depths)
        if abs(pd - ud) > 0.05 * max(ud, 1e-9):
            print(f"note: profiled steps ran {pd / len(prof_depths):.1f} rows deep against "
                  f"{ud / len(plain_depths):.1f} unprofiled - the comparison below is rough")
        prof.report(top=a.top, steps=a.steps)
        print(f"\nunprofiled step {plain * 1e3:.0f} ms against {prof.busy / a.steps * 1e3:.0f} ms "
              f"of GPU work: the GPU is busy {100 * min(prof.busy / a.steps / plain, 1):.0f}% "
              f"of an unprofiled step; the rest is the CPU (Python, dispatch, paging) or "
              f"waiting on it")
        if a.trace:
            prof.chrome_trace(a.trace)
            print(f"trace -> {a.trace}  (open in ui.perfetto.dev)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
