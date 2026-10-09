"""
The expert pool's SwiGLU on Apple Silicon: five Metal kernels in place of a
Python loop over experts.

What it replaces is PooledMLP's exact-size dispatch (run_exact) for a
single-level pool. Per recurrent row that path cast every slot's fp32 weights
to bf16 again, ran five small operations per expert, and under checkpointing
did it all twice. Its backward was worse: slicing each expert's rows out of
the gathered input makes, for every slice, a full-size zero gradient that is
then added up - 64 adds of [assignments, d_model] per row. tools/profile_step.py
put the pool at three quarters of a learning step's GPU time on an M5, and
its matmuls at well under half of that.

Here the assignments arrive sorted by expert, as the dispatch already sorts
them, and every kernel works on 64-row tiles of one expert's run:

  forward   up    H = silu(X W1^T) * (X W3^T)          both matmuls and the
                                                       gate in one kernel
            down  Y = H W2^T
  backward  bwd_h the up-projection recomputed and dH = dY W2 alongside it,
                  giving H, dA1 = dH*b*silu'(a), dA3 = dH*silu(a) in one pass
                  - the work checkpointing redid, with nothing re-run
            dx    dX = dA1 W1 + dA3 W3
            dw    dW1 = dA1^T X, dW3 = dA3^T X, dW2 = dY^T H, each expert
                  reducing over its own rows

Every matmul is a Metal 4 Performance Primitives matmul2d - the M5's matrix
units - with bf16 operands and fp32 accumulation. The weights are cast to
bf16 once per step, not once per row (the pool's own cast-once copy,
PagedPool.inference_weights, which knows when a slot changes), and their
gradients come back in fp32. Against run_exact this keeps more precision,
not less: H, dX and every weight gradient are accumulated in fp32 and
rounded once, where run_exact rounded each matmul's output to bf16.

Nothing is kept for the backward beyond what checkpointing kept: the input
and the routing. Combining the experts' outputs, weighted by the router,
stays in PyTorch, where autograd gives the router its gradient.
"""

import os

import torch

TM, TN = 64, 128            # rows of one expert, and output columns, per tile
THREADS = 128               # four SIMD groups

SOURCE = r'''
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int TM = 64;
constant constexpr int TN = 128;

using bmat = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;

// `rows` rows of `cols`, row-major. MPP extents run innermost first. A tile
// sliced at row r0 of a matrix of r1 rows reads rows past r1 as zero, so a
// tile at the end of an expert's run never sees the next expert's rows.
inline bmat mat(device const bfloat* p, int cols, int rows) {
  return bmat((device bfloat*)p, dextents<int32_t, 2>(cols, rows));
}

// fast math is on; keep exp's argument finite
inline float sigm(float a) { return 1.0f / (1.0f + exp(-clamp(a, -80.0f, 80.0f))); }

using NT = matmul2d<matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false),
                    execution_simdgroups<4>>;
using NN_ACC = matmul2d<matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate),
                        execution_simdgroups<4>>;
using TN_ = matmul2d<matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), true, false, false),
                     execution_simdgroups<4>>;

// tiles[i] = (expert, first row, end of that expert's rows, -): one 64-row tile

kernel void moe_up(device const bfloat* X [[buffer(0)]],    // [M, D] assignments, by expert
                   device const bfloat* W1 [[buffer(1)]],   // [E, F, D]
                   device const bfloat* W3 [[buffer(2)]],   // [E, F, D]
                   device bfloat* H [[buffer(3)]],          // [M, F]
                   device const int4* tiles [[buffer(4)]],
                   constant int& D [[buffer(5)]], constant int& F [[buffer(6)]],
                   uint2 tg [[threadgroup_position_in_grid]]) {
  const int4 t = tiles[tg.y];
  const int e = t.x, r0 = t.y, r1 = t.z, f0 = int(tg.x) * TN;
  auto Xb = mat(X, D, r1).slice(0, r0);
  auto W1b = mat(W1 + size_t(e) * F * D, D, F).slice(0, f0);
  auto W3b = mat(W3 + size_t(e) * F * D, D, F).slice(0, f0);
  NT op;
  auto A1 = op.get_destination_cooperative_tensor<decltype(Xb), decltype(W1b), float>();
  auto A3 = op.get_destination_cooperative_tensor<decltype(Xb), decltype(W1b), float>();
  op.run(Xb, W1b, A1);
  op.run(Xb, W3b, A3);
  for (ushort i = 0; i < A1.get_capacity(); ++i) {
    if (!A1.is_valid_element(i)) continue;
    auto ix = A1.get_multidimensional_index(i);          // (column, row)
    const int r = r0 + ix[1];
    if (r < r1) {
      const float a = A1[i];
      H[size_t(r) * F + f0 + ix[0]] = bfloat(a * sigm(a) * A3[i]);
    }
  }
}

#define MOE_DOWN(NAME, OT)                                                                    \
kernel void NAME(device const bfloat* Hm [[buffer(0)]],    /* [M, F] */                       \
                 device const bfloat* W2 [[buffer(1)]],    /* [E, D, F] */                    \
                 device OT* Y [[buffer(2)]],               /* [M, D] */                       \
                 device const int4* tiles [[buffer(3)]],                                      \
                 constant int& D [[buffer(4)]], constant int& F [[buffer(5)]],                \
                 uint2 tg [[threadgroup_position_in_grid]]) {                                 \
  const int4 t = tiles[tg.y];                                                                 \
  const int e = t.x, r0 = t.y, r1 = t.z, n0 = int(tg.x) * TN;                                 \
  auto Hb = mat(Hm, F, r1).slice(0, r0);                                                      \
  auto W2b = mat(W2 + size_t(e) * D * F, F, D).slice(0, n0);                                  \
  NT op;                                                                                      \
  auto C = op.get_destination_cooperative_tensor<decltype(Hb), decltype(W2b), float>();     \
  op.run(Hb, W2b, C);                                                                         \
  for (ushort i = 0; i < C.get_capacity(); ++i) {                                             \
    if (!C.is_valid_element(i)) continue;                                                     \
    auto ix = C.get_multidimensional_index(i);                                                \
    const int r = r0 + ix[1];                                                                 \
    if (r < r1) Y[size_t(r) * D + n0 + ix[0]] = OT(C[i]);                                     \
  }                                                                                           \
}
MOE_DOWN(moe_down_f32, float)
MOE_DOWN(moe_down_bf16, bfloat)

kernel void moe_bwd_h(device const bfloat* X [[buffer(0)]],    // [M, D]
                      device const bfloat* dY [[buffer(1)]],   // [M, D]
                      device const bfloat* W1 [[buffer(2)]],   // [E, F, D]
                      device const bfloat* W3 [[buffer(3)]],   // [E, F, D]
                      device const bfloat* W2T [[buffer(4)]],  // [E, F, D]: W2 transposed
                      device bfloat* H [[buffer(5)]],          // [M, F]
                      device bfloat* dA1 [[buffer(6)]],        // [M, F]
                      device bfloat* dA3 [[buffer(7)]],        // [M, F]
                      device const int4* tiles [[buffer(8)]],
                      constant int& D [[buffer(9)]], constant int& F [[buffer(10)]],
                      uint2 tg [[threadgroup_position_in_grid]]) {
  const int4 t = tiles[tg.y];
  const int e = t.x, r0 = t.y, r1 = t.z, f0 = int(tg.x) * TN;
  auto Xb = mat(X, D, r1).slice(0, r0);
  auto Gb = mat(dY, D, r1).slice(0, r0);
  const size_t w = size_t(e) * F * D;
  auto W1b = mat(W1 + w, D, F).slice(0, f0);
  auto W3b = mat(W3 + w, D, F).slice(0, f0);
  auto W2Tb = mat(W2T + w, D, F).slice(0, f0);
  NT op;                                       // one op, one layout for all three
  auto A1 = op.get_destination_cooperative_tensor<decltype(Xb), decltype(W1b), float>();
  auto A3 = op.get_destination_cooperative_tensor<decltype(Xb), decltype(W1b), float>();
  auto G = op.get_destination_cooperative_tensor<decltype(Xb), decltype(W1b), float>();
  op.run(Xb, W1b, A1);
  op.run(Xb, W3b, A3);
  op.run(Gb, W2Tb, G);                         // dH = dY W2
  for (ushort i = 0; i < A1.get_capacity(); ++i) {
    if (!A1.is_valid_element(i)) continue;
    auto ix = A1.get_multidimensional_index(i);
    const int r = r0 + ix[1];
    if (r < r1) {
      const float a = A1[i], b = A3[i], g = G[i];
      const float s = sigm(a), si = a * s;
      const size_t o = size_t(r) * F + f0 + ix[0];
      H[o] = bfloat(si * b);
      dA1[o] = bfloat(g * b * s * (1.0f + a * (1.0f - s)));
      dA3[o] = bfloat(g * si);
    }
  }
}

kernel void moe_dx(device const bfloat* dA1 [[buffer(0)]],    // [M, F]
                   device const bfloat* dA3 [[buffer(1)]],    // [M, F]
                   device const bfloat* W1 [[buffer(2)]],     // [E, F, D]
                   device const bfloat* W3 [[buffer(3)]],     // [E, F, D]
                   device float* dX [[buffer(4)]],            // [M, D]
                   device const int4* tiles [[buffer(5)]],
                   constant int& D [[buffer(6)]], constant int& F [[buffer(7)]],
                   uint2 tg [[threadgroup_position_in_grid]]) {
  const int4 t = tiles[tg.y];
  const int e = t.x, r0 = t.y, r1 = t.z, n0 = int(tg.x) * TN;
  auto A1b = mat(dA1, F, r1).slice(0, r0);
  auto A3b = mat(dA3, F, r1).slice(0, r0);
  const size_t w = size_t(e) * F * D;
  auto W1b = mat(W1 + w, D, F).slice(n0, 0);    // W1[e] is [F, D]: K = F, N = D
  auto W3b = mat(W3 + w, D, F).slice(n0, 0);
  NN_ACC op;
  auto C = op.get_destination_cooperative_tensor<decltype(A1b), decltype(W1b), float>();
  for (ushort i = 0; i < C.get_capacity(); ++i)
    if (C.is_valid_element(i)) C[i] = 0.0f;
  op.run(A1b, W1b, C);
  op.run(A3b, W3b, C);
  for (ushort i = 0; i < C.get_capacity(); ++i) {
    if (!C.is_valid_element(i)) continue;
    auto ix = C.get_multidimensional_index(i);
    const int r = r0 + ix[1];
    if (r < r1) dX[size_t(r) * D + n0 + ix[0]] = C[i];
  }
}

// C[e] = A_e^T B_e for every slot e: A [M, Ma] and B [M, Nb] hold every
// assignment, by expert; groups[e] = (e, first row, end, -). A slot nothing
// was routed to gets zeros.
kernel void moe_dw(device const bfloat* A [[buffer(0)]],
                   device const bfloat* B [[buffer(1)]],
                   device float* C [[buffer(2)]],           // [E, Ma, Nb]
                   device const int4* groups [[buffer(3)]],
                   constant int& Ma [[buffer(4)]], constant int& Nb [[buffer(5)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]]) {
  const int4 g = groups[tg.z];
  const int e = g.x, r0 = g.y, r1 = g.z;
  const int m0 = int(tg.y) * TM, n0 = int(tg.x) * TN;
  device float* Ce = C + size_t(e) * Ma * Nb;
  if (r1 <= r0) {
    for (int j = tid; j < TM * TN; j += 128)
      Ce[size_t(m0 + j / TN) * Nb + n0 + j % TN] = 0.0f;
    return;
  }
  auto Ab = mat(A, Ma, r1).slice(m0, r0);       // transposed: M = Ma, K = this expert's rows
  auto Bb = mat(B, Nb, r1).slice(n0, r0);
  TN_ op;
  auto Cc = op.get_destination_cooperative_tensor<decltype(Ab), decltype(Bb), float>();
  op.run(Ab, Bb, Cc);
  for (ushort i = 0; i < Cc.get_capacity(); ++i) {
    if (!Cc.is_valid_element(i)) continue;
    auto ix = Cc.get_multidimensional_index(i);
    Ce[size_t(m0 + ix[1]) * Nb + n0 + ix[0]] = Cc[i];
  }
}
'''

_LIB = {}


def _lib():
    if "lib" not in _LIB:
        try:
            _LIB["lib"] = torch.mps.compile_shader(SOURCE)
        except Exception as e:                             # noqa: BLE001
            _LIB["lib"] = None
            _LIB["why"] = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
    return _LIB["lib"]


def why_unsupported():
    return _LIB.get("why")


def supported(device, D, F):
    """Whether these kernels can run this pool's shapes here at all. Whether
    they compute it RIGHT is pool.py's probe to decide."""
    if os.environ.get("MINAGI_METAL_EXPERTS", "1").strip().lower() in ("0", "off", "no", "false"):
        return False
    return (getattr(device, "type", str(device)) == "mps" and D % TN == 0 and F % TN == 0
            and torch.backends.mps.is_available() and _lib() is not None)


def _launch(kernel, grid, *args):
    """Every launch goes through here, so a profiler can time them
    (tools/gpuprof.py wraps it)."""
    gx, gy, gz = (tuple(grid) + (1, 1))[:3]
    getattr(_lib(), kernel)(*args, threads=(gx * THREADS, gy, gz), group_size=(THREADS, 1, 1))


def tables(runs, n_slots, device):
    """The tile table - (expert, first row, end of its rows) for each 64-row
    tile - and the group table, one row per slot, from the dispatch's runs:
    (expert, how many) in sorted order."""
    tiles, groups, start = [], [], 0
    have = {}
    for e, c in runs:
        for r in range(start, start + c, TM):
            tiles.append((e, r, start + c, 0))
        have[e] = (start, start + c)
        start += c
    for e in range(n_slots):
        a, b = have.get(e, (0, 0))
        groups.append((e, a, b, 0))
    t = torch.tensor(tiles, dtype=torch.int32).to(device)
    g = torch.tensor(groups, dtype=torch.int32).to(device)
    return t, g


_W2T = {}


def weights_bf16(pool):
    """(w1, w3, w2, w2 transposed) in bf16, cast once per step: the pool's
    cast-once copy, which it drops whenever a slot changes, and W2's
    transpose for the backward, made once per copy."""
    w1, w3, w2 = pool.inference_weights(torch.bfloat16)
    key = id(w2)
    if _W2T.get("key") != key:
        _W2T.clear()
        _W2T.update(key=key, ref=w2, t=w2.transpose(1, 2).contiguous())
    return w1, w3, w2, _W2T["t"]


class _ExpertSwiGLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, flat, W1, W3, W2, t_sorted, tiles, groups, wb):
        w1, w3, w2, w2t = wb
        E, F, D = w1.shape
        X = flat[t_sorted].to(torch.bfloat16).contiguous()
        M, nt = X.shape[0], tiles.shape[0]
        H = torch.empty(M, F, device=flat.device, dtype=torch.bfloat16)
        _launch("moe_up", (F // TN, nt), X, w1, w3, H, tiles, D, F)
        if flat.dtype == torch.float32:
            Y, k = torch.empty(M, D, device=flat.device), "moe_down_f32"
        else:
            Y, k = torch.empty(M, D, device=flat.device, dtype=torch.bfloat16), "moe_down_bf16"
        _launch(k, (D // TN, nt), H, w2, Y, tiles, D, F)
        ctx.save_for_backward(flat, t_sorted, tiles, groups)
        ctx.wb = wb
        return Y if Y.dtype == flat.dtype else Y.to(flat.dtype)

    @staticmethod
    def backward(ctx, dY):
        flat, t_sorted, tiles, groups = ctx.saved_tensors
        w1, w3, w2, w2t = ctx.wb
        E, F, D = w1.shape
        dev = flat.device
        X = flat[t_sorted].to(torch.bfloat16).contiguous()
        dYb = dY.to(torch.bfloat16).contiguous()
        M, nt = X.shape[0], tiles.shape[0]
        H, dA1, dA3 = (torch.empty(M, F, device=dev, dtype=torch.bfloat16) for _ in range(3))
        _launch("moe_bwd_h", (F // TN, nt), X, dYb, w1, w3, w2t, H, dA1, dA3, tiles, D, F)
        dX = torch.empty(M, D, device=dev)
        _launch("moe_dx", (D // TN, nt), dA1, dA3, w1, w3, dX, tiles, D, F)
        dflat = torch.zeros_like(flat).index_add_(0, t_sorted, dX.to(flat.dtype))
        dW1 = torch.empty(E, F, D, device=dev)
        dW3 = torch.empty(E, F, D, device=dev)
        dW2 = torch.empty(E, D, F, device=dev)
        _launch("moe_dw", (D // TN, F // TM, E), dA1, X, dW1, groups, F, D)
        _launch("moe_dw", (D // TN, F // TM, E), dA3, X, dW3, groups, F, D)
        _launch("moe_dw", (F // TN, D // TM, E), dYb, H, dW2, groups, D, F)
        return dflat, dW1, dW3, dW2, None, None, None, None


_OK = {}


def available(device, D, F):
    """
    Whether the kernels compute this pool's SwiGLU correctly here: they exist,
    and on a small routed problem their output and all four gradients agree
    with the same arithmetic done exactly in fp32 - to well within what the
    bf16 path they replace manages (about 0.5%). Checked once per device.
    """
    key = (str(device), D, F)
    if key in _OK:
        return _OK[key]
    ok = supported(device, D, F)
    if ok:
        try:
            ok = _probe(device, D, F) < 2e-2
        except Exception:                                  # noqa: BLE001
            ok = False
        if not ok:
            print("  experts: the Metal kernels failed their check - computing them "
                  "expert by expert", flush=True)
    _OK[key] = ok
    return ok


class _Weights:
    """A pool's one question of these kernels, for the probe."""

    def __init__(self, ws):
        self.ws = ws

    def inference_weights(self, dtype):
        return tuple(w.detach().to(dtype) for w in self.ws)


def _probe(device, D, F):
    import torch.nn.functional as Fn
    g = torch.Generator().manual_seed(0)
    E, N, k = 4, 150, 2
    flat = torch.randn(N, D, generator=g).to(device)
    W = [(torch.randn(E, F, D, generator=g) * D ** -0.5).to(device),
         (torch.randn(E, F, D, generator=g) * D ** -0.5).to(device),
         (torch.randn(E, D, F, generator=g) * F ** -0.5).to(device)]
    e = torch.randint(0, E - 1, (N * k,), generator=g)        # the last expert unused
    order = torch.argsort(e, stable=True)
    t_sorted = torch.arange(N).repeat_interleave(k)[order].to(device)
    runs = [(i, c) for i, c in enumerate(torch.bincount(e, minlength=E).tolist()) if c]
    gY = torch.randn(N * k, D, generator=g).to(device)

    def grads(fn):
        x, a, b, c = (t.clone().requires_grad_() for t in [flat] + W)
        with torch.enable_grad():
            y = fn(x, a, b, c)
            return [y] + list(torch.autograd.grad((y.float() * gY).sum(), (x, a, b, c)))

    def exact(x, a, b, c):
        xs, outs, s = x[t_sorted], [], 0
        for i, n in runs:
            xe = xs[s:s + n]
            outs.append((Fn.silu(xe @ a[i].t()) * (xe @ b[i].t())) @ c[i].t())
            s += n
        return torch.cat(outs)

    # the first call comes from inside a forward, under autocast: the
    # reference must still be exact fp32, not autocast's bf16
    with torch.inference_mode(False), torch.autocast(device.type, enabled=False):
        ref = grads(exact)
        got = grads(lambda x, a, b, c: expert_swiglu(x, t_sorted, runs, a, b, c,
                                                      _Weights((a, b, c))))
    return max(float((u.detach().float() - r.detach()).norm() / r.detach().norm())
               for u, r in zip(got, ref))


def expert_swiglu(flat, t_sorted, runs, W1, W3, W2, pool):
    """Every routed expert's SwiGLU on the assignments routed to it -
    run_exact's result, [assignments, d_model] in flat's dtype, rows in
    t_sorted's order. `runs` is (expert, how many) per expert in that order;
    W1, W3, W2 are the slots' fp32 weights, which receive the gradients."""
    tiles, groups = tables(runs, W1.shape[0], flat.device)
    return _ExpertSwiGLU.apply(flat, W1, W3, W2, t_sorted, tiles, groups, weights_bf16(pool))
