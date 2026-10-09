"""
GPU time per operation on an Apple GPU, attributed to the model's modules.

PyTorch's profiler records no GPU time on MPS. This times every operation by
Metal's own clock (gputime.mm: each command buffer's GPU start and end) and
says where in the model it came from:

  phase    forward, backward, recompute (a checkpointed region re-run inside
           the backward) or a phase the caller names, e.g. optimizer
  module   the innermost module whose forward the operation ran in. In the
           backward, the module whose forward created the autograd node being
           run: every node carries a sequence number, and each module call's
           range of them is recorded on the way through the forward
  op       the aten operation, or the Metal kernel, with its operand shapes

    from gpuprof import GPUProfile           # tools/ on the path
    with GPUProfile(model) as prof:
        step()
        with prof.phase("optimizer"):
            opt.step()
    prof.report()
    prof.chrome_trace("trace.json")          # open in ui.perfetto.dev

What it costs. Each operation gets a command buffer of its own, committed
without waiting, so the GPU time of each is exact but the step as a whole
runs slower than it does unprofiled: the GPU waits on the CPU between
buffers. The report keeps the two apart - GPU busy time is the work; the
idle time between buffers is mostly the profiler's own.
"""

import collections
import contextlib
import json
import os
import re
import threading

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

_EXT = None


def ext():
    """The gputime extension, compiled the first time it is asked for
    (a few seconds; cached by PyTorch after that)."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="minagi_gputime",
                    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "gputime.mm")],
                    extra_cflags=["-std=c++20"],
                    extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
                    verbose=False)
    return _EXT


_NO_SEQ = 2 ** 64 - 1              # AccumulateGrad and other nodes without a number
_SHORT = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16",
          torch.int64: "i64", torch.int32: "i32", torch.bool: "bool", torch.int16: "i16"}


def _sig(args):
    """Operand shapes, the way the report prints them: bf16[32,1024,512]."""
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(f"{_SHORT.get(a.dtype, str(a.dtype))}[{','.join(map(str, a.shape))}]")
        elif isinstance(a, (list, tuple)) and a and isinstance(a[0], torch.Tensor):
            out.append(f"{len(a)}x" + _sig(a[:1]))
        if len(out) == 3:
            break
    return " ".join(out)


def _on_mps(args, kwargs):
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor):
            if a.device.type == "mps":
                return True
        elif isinstance(a, (list, tuple)):
            if any(isinstance(t, torch.Tensor) and t.device.type == "mps" for t in a):
                return True
        elif isinstance(a, torch.device) and a.type == "mps":
            return True
    return False


class _Mode(TorchDispatchMode):
    def __init__(self, prof):
        super().__init__()
        self.prof = prof

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.is_view or not _on_mps(args, kwargs):
            return func(*args, **kwargs)
        tag = self.prof._op("aten", func.overloadpacket.__name__, args)
        E = self.prof._e
        E.begin(tag)
        try:
            return func(*args, **kwargs)
        finally:
            E.end(tag)


class GPUProfile:
    def __init__(self, model=None):
        self.model = model
        self._names = {id(m): (n or "<root>") for n, m in model.named_modules()} if model else {}
        self._e = ext()
        self._tls = threading.local()
        self._ops = []              # tag -> (kind, name, phase, module, sig)
        self._ranges = []           # (lo, hi, depth, module): a forward call's sequence numbers
        self._paint = None
        self._phase = None
        self._recomputing = 0
        self._probe = torch.zeros(1, requires_grad=True)
        self.records = None

    # ------------------------------------------------------------ attribution
    def _stack(self):
        st = getattr(self._tls, "stack", None)
        if st is None:
            st = self._tls.stack = []
        return st

    def _seq(self):
        """The sequence number the next autograd node on this thread gets."""
        with torch.inference_mode(False), torch.enable_grad():
            return self._probe.clone().grad_fn._sequence_nr() + 1

    def _pre(self, mod, args):
        st = self._stack()
        st.append((self._names.get(id(mod), type(mod).__name__), self._seq(), len(st)))

    def _post(self, mod, args, out):
        st = self._stack()
        if not st:
            return
        name, lo, depth = st.pop()
        if torch._C._current_autograd_node() is None:      # a forward, not a recompute
            self._ranges.append((lo, self._seq(), depth, name))
            self._paint = None

    def _module_of_node(self, node):
        seq = node._sequence_nr()
        if seq == _NO_SEQ:
            return "<gradient accumulation>"
        if self._paint is None:
            self._build_paint()
        lo, ids, names = self._paint
        i = seq - lo
        if 0 <= i < len(ids) and ids[i] >= 0:
            return names[ids[i]]
        return "<outside the model>"

    def _build_paint(self):
        """Every forward call paints its range of sequence numbers with its
        name, outer calls first, so the innermost module wins."""
        if not self._ranges:
            self._paint = (0, np.zeros(0, np.int32), [])
            return
        lo = min(r[0] for r in self._ranges)
        hi = max(r[1] for r in self._ranges)
        ids = np.full(hi - lo, -1, np.int32)
        names, index = [], {}
        for a, b, _, name in sorted(self._ranges, key=lambda r: r[2]):
            if name not in index:
                index[name] = len(names)
                names.append(name)
            ids[a - lo:b - lo] = index[name]
        self._paint = (lo, ids, names)

    def _context(self):
        node = torch._C._current_autograd_node()
        if self._phase is not None:
            return self._phase, "<" + self._phase + ">"
        if node is None:
            st = self._stack()
            return "forward", (st[-1][0] if st else "<outside the model>")
        return ("recompute" if self._recomputing else "backward"), self._module_of_node(node)

    def _op(self, kind, name, args):
        phase, module = self._context()
        self._ops.append((kind, name, phase, module, _sig(args)))
        return len(self._ops) - 1

    # ------------------------------------------------------------ hooks in, hooks out
    @contextlib.contextmanager
    def phase(self, name):
        """Time what runs inside as `name` - an optimizer step, say."""
        old, self._phase = self._phase, name
        try:
            yield
        finally:
            self._phase = old

    def _wrap_checkpoint(self, real):
        prof = self

        def checkpoint(fn, *a, **k):
            def run(*aa, **kk):
                if torch._C._current_autograd_node() is not None:
                    prof._recomputing += 1
                    try:
                        return fn(*aa, **kk)
                    finally:
                        prof._recomputing -= 1
                return fn(*aa, **kk)
            return real(run, *a, **k)
        return checkpoint

    def _wrap_launch(self, real):
        prof = self

        def launch(name, groups, bh, *args):
            tag = prof._op("metal", name, args)
            prof._e.begin(tag)
            try:
                return real(name, groups, bh, *args)
            finally:
                prof._e.end(tag)
        return launch

    def __enter__(self):
        import minagi.metal_attention as MA
        import minagi.model as MM
        import minagi.pool as MP
        torch.mps.synchronize()
        self._e.collect()                                  # nothing from before
        self._patches = [(MA, "_launch", MA._launch), (MP, "checkpoint", MP.checkpoint),
                         (MM, "checkpoint", MM.checkpoint)]
        MA._launch = self._wrap_launch(MA._launch)
        MP.checkpoint = self._wrap_checkpoint(MP.checkpoint)
        MM.checkpoint = self._wrap_checkpoint(MM.checkpoint)
        from torch.nn.modules import module as nnm
        self._hooks = [nnm.register_module_forward_pre_hook(self._pre),
                       nnm.register_module_forward_hook(self._post, always_call=True)]
        self._t0 = self._e.now()
        self._mode = _Mode(self)
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        self._mode.__exit__(*exc)
        for h in self._hooks:
            h.remove()
        for mod, attr, real in self._patches:
            setattr(mod, attr, real)
        self.records = self._e.collect()
        self._t1 = self._e.now()
        self._join()
        return False

    # ------------------------------------------------------------ results
    def _join(self):
        """One row per operation: its GPU time, from every distinct command
        buffer its tag was reported on."""
        spans = collections.defaultdict(set)
        for tag, _cpu, g0, g1 in self.records:
            if g1 > g0 > 0:
                spans[tag].add((g0, g1))
        self.rows = []
        for tag, (kind, name, phase, module, sig) in enumerate(self._ops):
            s = sorted(spans.get(tag, ()))
            self.rows.append({"kind": kind, "op": name, "phase": phase, "module": module,
                              "sig": sig, "gpu": sum(b - a for a, b in s), "spans": s})
        every = sorted(x for r in self.rows for x in r["spans"])
        busy, end = 0.0, None
        for a, b in every:                                 # union: overlap counted once
            if end is None or a >= end:
                busy += b - a
                end = b
            elif b > end:
                busy += b - end
                end = b
        self.busy = busy
        self.wall = (every[-1][1] - every[0][0]) if every else 0.0
        self.buffers = len(every)

    @staticmethod
    def group(module):
        """prelude.0.attn and prelude.1.attn are one row: prelude.*.attn."""
        return re.sub(r"\.\d+(?=\.|$)", ".*", module)

    def table(self, key, rows=None):
        """{key: [gpu seconds, count]} over the rows, largest first."""
        acc = collections.defaultdict(lambda: [0.0, 0])
        for r in rows if rows is not None else self.rows:
            k = key(r)
            acc[k][0] += r["gpu"]
            acc[k][1] += 1
        return sorted(acc.items(), key=lambda kv: -kv[1][0])

    def report(self, top=25, steps=1, file=None):
        p = lambda *a: print(*a, file=file)            # noqa: E731
        n = max(steps, 1)
        busy = max(self.busy, 1e-12)
        timed = sum(r["gpu"] for r in self.rows)
        p(f"GPU TIME  {n} step{'s' if n > 1 else ''} profiled, per step:")
        p(f"  GPU busy {self.busy / n * 1e3:8.1f} ms    {len(self.rows) / n:7.0f} operations, "
          f"{self.buffers / n:.0f} command buffers")
        p(f"  wall     {self.wall / n * 1e3:8.1f} ms    (profiled - each op committed alone, "
          f"so the GPU waits between them)")
        p("")

        def section(title, key, rows=None, limit=top, sub=None):
            p(title)
            p(f"  {'':58} {'ms/step':>9} {'%':>6} {'ops/step':>9} {'us/op':>8}")
            for k, (t, c) in self.table(key, rows)[:limit]:
                label = k if len(k) <= 58 else k[:55] + "..."
                p(f"  {label:58} {t / n * 1e3:9.2f} {100 * t / busy:5.1f}% "
                  f"{c / n:9.1f} {t / c * 1e6:8.1f}")
            p("")

        section("BY PHASE", lambda r: r["phase"])
        section("BY MODULE  (innermost module whose forward ran or created the op)",
                lambda r: self.group(r["module"]))
        section("BY MODULE AND PHASE", lambda r: f"{self.group(r['module'])}  [{r['phase']}]")
        section("BY OPERATION", lambda r: f"{r['op']}  [{r['phase']}]" +
                ("  (Metal kernel)" if r["kind"] == "metal" else ""))
        section("HEAVIEST SINGLE SHAPES  (operation, operands, module)",
                lambda r: f"{r['op']} {r['sig']}  {self.group(r['module'])}", limit=top)
        syncs = sum(1 for r in self.rows if r["op"] in ("_local_scalar_dense", "nonzero",
                                                        "masked_select", "unique"))
        p(f"host syncs per step (item(), nonzero ...): {syncs / n:.0f}")
        if abs(timed - self.busy) > 0.05 * busy:
            p(f"note: per-op times sum to {timed / n * 1e3:.1f} ms against "
              f"{self.busy / n * 1e3:.1f} busy - some command buffers were shared")

    def chrome_trace(self, path):
        """Every operation's GPU span, for ui.perfetto.dev or chrome://tracing:
        one track per phase, the module and operands in each span's details."""
        t0 = min((a for r in self.rows for a, _ in r["spans"]), default=0.0)
        ev = []
        for i, r in enumerate(self.rows):
            for a, b in r["spans"]:
                ev.append({"name": r["op"], "ph": "X", "pid": "GPU", "tid": r["phase"],
                           "ts": (a - t0) * 1e6, "dur": (b - a) * 1e6,
                           "args": {"module": r["module"], "operands": r["sig"], "op#": i}})
        with open(path, "w") as f:
            json.dump({"traceEvents": ev, "displayTimeUnit": "ms"}, f)
