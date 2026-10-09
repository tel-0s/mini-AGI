"""
Fused causal attention for Apple Silicon: a Metal kernel, forward and backward.

PyTorch's MPS attention computes the right thing but keeps every score matrix
for the backward, and the blocked path in model.py, which keeps none, computes
in fp32 throughout - so bf16 attention there never reaches the GPU's matrix
hardware. This is the flash-attention algorithm (FlashAttention-2's split of
the backward) written for the M5 generation's matrix units, through Metal 4's
Performance Primitives (MPP): every matrix multiply is an MPP matmul2d with
bf16 (or fp16) operands and fp32 accumulation, which on an M5 runs at ~14.5
TFLOP/s against ~3.5 for the classic simdgroup_matrix path. Scores, softmax
and its statistics stay in fp32, as a CUDA flash kernel keeps them.

Semantics are blocked_attention's exactly: T queries at absolute positions
P..P+T-1 against P+T keys, causal, scale 1/sqrt(head_dim). What the forward
keeps for the backward is q, k, v, the output and one log-sum-exp per query.

Tiles are 64 queries x 64 keys, head_dim 64 - this model's shape; anything
else is reported unavailable and computed in blocks. Each threadgroup is four
SIMD groups that run every matmul together.

Numerics. Metal compiles with fast math, which may assume there are no
infinities, so none are used: a masked score is NEG, a large finite negative,
and exp2 of it underflows to zero. Softmax runs in base 2 with the scale and
log2(e) folded into one multiply; the saved log-sum-exp is in that base too.
P is rounded to the operand type for the P @ V multiply, as flash kernels do.

Backward: delta = rowsum(dO * O) is one PyTorch op; then one kernel per
query block accumulates dQ over key blocks, and one per key block accumulates
dK and dV over query blocks. No atomics, so the gradient is deterministic.
"""

import math
import os

import torch

BQ = BK = HD = 64
THREADS = 128                       # four SIMD groups of 32

SOURCE = r'''
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BQ = 64;
constant constexpr int BK = 64;
constant constexpr int HD = 64;
constant constexpr float NEG = -1.0e30f;

template <typename T> using dtensor = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
template <typename T> using ttensor = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;

// n rows of HD, row-major. MPP extents run innermost first: (columns, rows).
template <typename T>
inline dtensor<T> rows_of(device const T* p, int n) {
  return dtensor<T>((device T*)p, dextents<int32_t, 2>(HD, n));
}

using mac = matmul2d_descriptor::mode;

// ---------------------------------------------------------------- forward
template <typename T, typename PT>
inline void fwd_impl(device const T* Q, device const T* K, device const T* V,
                     device T* O, device float* L,
                     int Tq, int Sk, int P, float c, uint qb, uint bh, ushort tid,
                     threadgroup float* Ss, threadgroup PT* Ps,
                     threadgroup float* m_s, threadgroup float* l_s, threadgroup float* corr_s)
{
  const int q0 = int(qb) * BQ;
  Q += size_t(bh) * Tq * HD;  O += size_t(bh) * Tq * HD;  L += size_t(bh) * Tq;
  K += size_t(bh) * Sk * HD;  V += size_t(bh) * Sk * HD;

  auto Qt = rows_of(Q, Tq);
  auto Kt = rows_of(K, Sk);
  auto Vt = rows_of(V, Sk);
  auto Qb = Qt.slice(0, q0);
  auto Pt = ttensor<PT>(Ps, dextents<int32_t, 2>(BK, BQ));
  using KB = decltype(Kt.slice(0, 0));

  constexpr auto qk_d = matmul2d_descriptor(BQ, BK, HD, false, true, false);
  constexpr auto pv_d = matmul2d_descriptor(BQ, HD, BK, false, false, false, mac::multiply_accumulate);
  matmul2d<qk_d, execution_simdgroups<4>> qk;
  matmul2d<pv_d, execution_simdgroups<4>> pv;

  auto Oc = pv.get_destination_cooperative_tensor<decltype(Pt), KB, float>();
  for (ushort i = 0; i < Oc.get_capacity(); ++i)
    if (Oc.is_valid_element(i)) Oc[i] = 0.0f;
  if (tid < BQ) { m_s[tid] = NEG; l_s[tid] = 0.0f; }

  const int kend = min(Sk, P + q0 + BQ);          // keys any row of this block sees
  const int row = tid >> 1, part = tid & 1;       // softmax: two threads a row
  for (int k0 = 0; k0 < kend; k0 += BK) {
    auto Kb = Kt.slice(0, k0);
    auto Sc = qk.get_destination_cooperative_tensor<decltype(Qb), KB, float>();
    qk.run(Qb, Kb, Sc);
    const bool edge = (k0 + BK - 1 > P + q0) || (k0 + BK > Sk);
    for (ushort i = 0; i < Sc.get_capacity(); ++i) {
      if (!Sc.is_valid_element(i)) continue;
      auto ix = Sc.get_multidimensional_index(i);           // (key, query)
      float s = Sc[i] * c;
      if (edge) {
        int kp = k0 + ix[0];
        if (kp > P + q0 + ix[1] || kp >= Sk) s = NEG;
      }
      Ss[ix[1] * BK + ix[0]] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
      threadgroup float* sr = Ss + row * BK + part * 32;
      threadgroup PT* pr = Ps + row * BK + part * 32;
      const float m_old = m_s[row];
      float mx = NEG;
      for (int j = 0; j < 32; j++) mx = max(mx, sr[j]);
      mx = max(mx, simd_shuffle_xor(mx, 1));
      const float m_new = max(m_old, mx);
      const float corr = exp2(m_old - m_new);
      float sum = 0.0f;
      for (int j = 0; j < 32; j++) {
        float p = exp2(sr[j] - m_new);
        sum += p;
        pr[j] = PT(p);
      }
      sum += simd_shuffle_xor(sum, 1);      // both threads of the row read m_old above
      if (part == 0) {
        m_s[row] = m_new;
        l_s[row] = l_s[row] * corr + sum;
        corr_s[row] = corr;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort i = 0; i < Oc.get_capacity(); ++i) {
      if (!Oc.is_valid_element(i)) continue;
      auto ix = Oc.get_multidimensional_index(i);
      Oc[i] *= corr_s[ix[1]];
    }
    auto Vb = Vt.slice(0, k0);
    pv.run(Pt, Vb, Oc);
    // Where P is written over the scores (fp32), the next block's scores
    // must wait until every SIMD group has finished reading this P. A
    // separate P buffer needs no wait: the next write to it is behind the
    // next block's first barrier.
    if (is_same<PT, float>::value) threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (ushort i = 0; i < Oc.get_capacity(); ++i) {
    if (!Oc.is_valid_element(i)) continue;
    auto ix = Oc.get_multidimensional_index(i);
    int qi = q0 + ix[1];
    if (qi < Tq) O[size_t(qi) * HD + ix[0]] = T(Oc[i] / l_s[ix[1]]);
  }
  if (tid < BQ && q0 + int(tid) < Tq) L[q0 + tid] = m_s[tid] + log2(l_s[tid]);
}

// ---------------------------------------------------------------- dQ
template <typename T, typename PT>
inline void dq_impl(device const T* Q, device const T* K, device const T* V,
                    device const T* dO, device const float* L, device const float* Dl,
                    device T* dQ, int Tq, int Sk, int P, float c, float scale,
                    uint qb, uint bh, ushort tid,
                    threadgroup PT* dSs, threadgroup float* L_s, threadgroup float* D_s)
{
  const int q0 = int(qb) * BQ;
  Q += size_t(bh) * Tq * HD;  dO += size_t(bh) * Tq * HD;  dQ += size_t(bh) * Tq * HD;
  L += size_t(bh) * Tq;  Dl += size_t(bh) * Tq;
  K += size_t(bh) * Sk * HD;  V += size_t(bh) * Sk * HD;

  auto Qt = rows_of(Q, Tq);   auto dOt = rows_of(dO, Tq);
  auto Kt = rows_of(K, Sk);   auto Vt = rows_of(V, Sk);
  auto Qb = Qt.slice(0, q0);  auto dOb = dOt.slice(0, q0);
  auto dSt = ttensor<PT>(dSs, dextents<int32_t, 2>(BK, BQ));
  using KB = decltype(Kt.slice(0, 0));

  constexpr auto qk_d = matmul2d_descriptor(BQ, BK, HD, false, true, false);
  constexpr auto dq_d = matmul2d_descriptor(BQ, HD, BK, false, false, false, mac::multiply_accumulate);
  matmul2d<qk_d, execution_simdgroups<4>> qk;     // S = Q K^T and dP = dO V^T: one layout
  matmul2d<dq_d, execution_simdgroups<4>> dq;

  auto dQc = dq.get_destination_cooperative_tensor<decltype(dSt), KB, float>();
  for (ushort i = 0; i < dQc.get_capacity(); ++i)
    if (dQc.is_valid_element(i)) dQc[i] = 0.0f;
  if (tid < BQ) {
    int qi = q0 + tid;
    L_s[tid] = qi < Tq ? L[qi] : 0.0f;
    D_s[tid] = qi < Tq ? Dl[qi] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int kend = min(Sk, P + q0 + BQ);
  for (int k0 = 0; k0 < kend; k0 += BK) {
    auto Kb = Kt.slice(0, k0);
    auto Vb = Vt.slice(0, k0);
    auto Sc = qk.get_destination_cooperative_tensor<decltype(Qb), KB, float>();
    auto dPc = qk.get_destination_cooperative_tensor<decltype(Qb), KB, float>();
    qk.run(Qb, Kb, Sc);
    qk.run(dOb, Vb, dPc);
    const bool edge = (k0 + BK - 1 > P + q0) || (k0 + BK > Sk);
    for (ushort i = 0; i < Sc.get_capacity(); ++i) {
      if (!Sc.is_valid_element(i)) continue;
      auto ix = Sc.get_multidimensional_index(i);           // (key, query)
      float p = exp2(Sc[i] * c - L_s[ix[1]]);
      if (edge) {
        int kp = k0 + ix[0];
        if (kp > P + q0 + ix[1] || kp >= Sk) p = 0.0f;
      }
      dSs[ix[1] * BK + ix[0]] = PT(p * (dPc[i] - D_s[ix[1]]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dq.run(dSt, Kb, dQc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (ushort i = 0; i < dQc.get_capacity(); ++i) {
    if (!dQc.is_valid_element(i)) continue;
    auto ix = dQc.get_multidimensional_index(i);
    int qi = q0 + ix[1];
    if (qi < Tq) dQ[size_t(qi) * HD + ix[0]] = T(dQc[i] * scale);
  }
}

// ---------------------------------------------------------------- dK, dV
template <typename T, typename PT>
inline void dkv_impl(device const T* Q, device const T* K, device const T* V,
                     device const T* dO, device const float* L, device const float* Dl,
                     device T* dK, device T* dV, int Tq, int Sk, int P, float c, float scale,
                     uint kb, uint bh, ushort tid,
                     threadgroup PT* Xs, threadgroup float* L_s, threadgroup float* D_s)
{
  const int k0 = int(kb) * BK;
  Q += size_t(bh) * Tq * HD;  dO += size_t(bh) * Tq * HD;
  L += size_t(bh) * Tq;  Dl += size_t(bh) * Tq;
  K += size_t(bh) * Sk * HD;  V += size_t(bh) * Sk * HD;
  dK += size_t(bh) * Sk * HD;  dV += size_t(bh) * Sk * HD;

  auto Qt = rows_of(Q, Tq);   auto dOt = rows_of(dO, Tq);
  auto Kt = rows_of(K, Sk);   auto Vt = rows_of(V, Sk);
  auto Kb = Kt.slice(0, k0);  auto Vb = Vt.slice(0, k0);
  auto Xt = ttensor<PT>(Xs, dextents<int32_t, 2>(BQ, BK));   // [key, query]
  using QB = decltype(Qt.slice(0, 0));

  constexpr auto st_d = matmul2d_descriptor(BK, BQ, HD, false, true, false);
  constexpr auto acc_d = matmul2d_descriptor(BK, HD, BQ, false, false, false, mac::multiply_accumulate);
  matmul2d<st_d, execution_simdgroups<4>> st;     // S^T = K Q^T and dP^T = V dO^T
  matmul2d<acc_d, execution_simdgroups<4>> acc;   // dV += P^T dO, dK += dS^T Q

  auto dVc = acc.get_destination_cooperative_tensor<decltype(Xt), QB, float>();
  auto dKc = acc.get_destination_cooperative_tensor<decltype(Xt), QB, float>();
  for (ushort i = 0; i < dVc.get_capacity(); ++i)
    if (dVc.is_valid_element(i)) { dVc[i] = 0.0f; dKc[i] = 0.0f; }

  int qstart = max(0, k0 - P);                      // the first query that sees key k0
  qstart = (qstart / BQ) * BQ;
  for (int q0 = qstart; q0 < Tq; q0 += BQ) {
    threadgroup_barrier(mem_flags::mem_threadgroup);   // the last block's runs are done
    if (tid < BQ) {
      int qi = q0 + tid;
      L_s[tid] = qi < Tq ? L[qi] : 0.0f;
      D_s[tid] = qi < Tq ? Dl[qi] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto Qb = Qt.slice(0, q0);
    auto dOb = dOt.slice(0, q0);
    auto STc = st.get_destination_cooperative_tensor<decltype(Kb), QB, float>();
    auto dPTc = st.get_destination_cooperative_tensor<decltype(Kb), QB, float>();
    st.run(Kb, Qb, STc);
    st.run(Vb, dOb, dPTc);
    const bool edge = (k0 + BK - 1 > P + q0) || (k0 + BK > Sk) || (q0 + BQ > Tq);
    for (ushort i = 0; i < STc.get_capacity(); ++i) {
      if (!STc.is_valid_element(i)) continue;
      auto ix = STc.get_multidimensional_index(i);          // (query, key)
      float p = exp2(STc[i] * c - L_s[ix[0]]);
      if (edge) {
        int kp = k0 + ix[1], ql = q0 + ix[0];
        if (kp > P + ql || kp >= Sk || ql >= Tq) p = 0.0f;
      }
      STc[i] = p;
      Xs[ix[1] * BQ + ix[0]] = PT(p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc.run(Xt, dOb, dVc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort i = 0; i < STc.get_capacity(); ++i) {
      if (!STc.is_valid_element(i)) continue;
      auto ix = STc.get_multidimensional_index(i);
      Xs[ix[1] * BQ + ix[0]] = PT(STc[i] * (dPTc[i] - D_s[ix[0]]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc.run(Xt, Qb, dKc);
  }

  for (ushort i = 0; i < dVc.get_capacity(); ++i) {
    if (!dVc.is_valid_element(i)) continue;
    auto ix = dVc.get_multidimensional_index(i);
    int kr = k0 + ix[1];
    if (kr < Sk) {
      dV[size_t(kr) * HD + ix[0]] = T(dVc[i]);
      dK[size_t(kr) * HD + ix[0]] = T(dKc[i] * scale);
    }
  }
}

// ---------------------------------------------------------------- entry points
#define ATTN_KERNELS(NAME, T, PT, FWD_PS)                                                     \
kernel void attn_fwd_##NAME(device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]], \
    device const T* V [[buffer(2)]], device T* O [[buffer(3)]], device float* L [[buffer(4)]], \
    constant int& Tq [[buffer(5)]], constant int& Sk [[buffer(6)]], constant int& P [[buffer(7)]], \
    constant float& c [[buffer(8)]],                                                          \
    uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {  \
  threadgroup float Ss[BQ * BK];                                                              \
  FWD_PS                                                                                      \
  threadgroup float m_s[BQ], l_s[BQ], corr_s[BQ];                                             \
  fwd_impl<T, PT>(Q, K, V, O, L, Tq, Sk, P, c, tg.x, tg.y, tid, Ss, Ps, m_s, l_s, corr_s);    \
}                                                                                             \
kernel void attn_dq_##NAME(device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]],  \
    device const T* V [[buffer(2)]], device const T* dO [[buffer(3)]],                        \
    device const float* L [[buffer(4)]], device const float* Dl [[buffer(5)]],                \
    device T* dQ [[buffer(6)]],                                                               \
    constant int& Tq [[buffer(7)]], constant int& Sk [[buffer(8)]], constant int& P [[buffer(9)]], \
    constant float& c [[buffer(10)]], constant float& scale [[buffer(11)]],                   \
    uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {  \
  threadgroup PT dSs[BQ * BK];                                                                \
  threadgroup float L_s[BQ], D_s[BQ];                                                         \
  dq_impl<T, PT>(Q, K, V, dO, L, Dl, dQ, Tq, Sk, P, c, scale, tg.x, tg.y, tid, dSs, L_s, D_s); \
}                                                                                             \
kernel void attn_dkv_##NAME(device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]], \
    device const T* V [[buffer(2)]], device const T* dO [[buffer(3)]],                        \
    device const float* L [[buffer(4)]], device const float* Dl [[buffer(5)]],                \
    device T* dK [[buffer(6)]], device T* dV [[buffer(7)]],                                   \
    constant int& Tq [[buffer(8)]], constant int& Sk [[buffer(9)]], constant int& P [[buffer(10)]], \
    constant float& c [[buffer(11)]], constant float& scale [[buffer(12)]],                   \
    uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {  \
  threadgroup PT Xs[BQ * BK];                                                                 \
  threadgroup float L_s[BQ], D_s[BQ];                                                         \
  dkv_impl<T, PT>(Q, K, V, dO, L, Dl, dK, dV, Tq, Sk, P, c, scale, tg.x, tg.y, tid, Xs, L_s, D_s); \
}

// bf16 and fp16 keep P in its own buffer; fp32 writes it over the scores in
// place (each thread rewrites exactly the elements it read), which keeps the
// forward inside 32 KB of threadgroup memory
ATTN_KERNELS(bf16, bfloat, bfloat, threadgroup bfloat Ps[BQ * BK];)
ATTN_KERNELS(f16, half, half, threadgroup half Ps[BQ * BK];)
ATTN_KERNELS(f32, float, float, threadgroup float* Ps = Ss;)
'''

_NAMES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
_LIB = {}


def _lib():
    """The compiled kernels, or None where they cannot be had (not MPS, or a
    Metal without the Performance Primitives). Compiled once per process."""
    if "lib" not in _LIB:
        try:
            _LIB["lib"] = torch.mps.compile_shader(SOURCE)
        except Exception as e:                             # noqa: BLE001
            _LIB["lib"] = None
            _LIB["why"] = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
    return _LIB["lib"]


def supported(device, dtype, head_dim):
    """Whether this kernel can compute attention for these tensors at all.
    Whether it computes it RIGHT is model.py's probe to decide."""
    if os.environ.get("MINAGI_METAL_ATTENTION", "1").strip().lower() in ("0", "off", "no", "false"):
        return False
    return (getattr(device, "type", str(device)) == "mps" and dtype in _NAMES
            and head_dim == HD and torch.backends.mps.is_available()
            and _lib() is not None)


def why_unsupported():
    return _LIB.get("why")


def _launch(fn, groups, bh, *args):
    fn(*args, threads=(groups * THREADS, bh), group_size=(THREADS, 1))


class _MetalAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, P):
        B, H, T, D = q.shape
        S = k.shape[2]
        name = _NAMES[v.dtype]
        q, k, v = (t.contiguous() for t in (q, k, v))
        scale = D ** -0.5
        o = torch.empty_like(q)
        lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
        _launch(getattr(_lib(), f"attn_fwd_{name}"), (T + BQ - 1) // BQ, B * H,
                q, k, v, o, lse, T, S, P, scale * math.log2(math.e))
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.P = P
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, T, D = q.shape
        S = k.shape[2]
        name = _NAMES[v.dtype]
        do = do.to(v.dtype).contiguous()
        scale = D ** -0.5
        c = scale * math.log2(math.e)
        delta = (do.float() * o.float()).sum(-1)               # [B, H, T]
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        lib = _lib()
        _launch(getattr(lib, f"attn_dq_{name}"), (T + BQ - 1) // BQ, B * H,
                q, k, v, do, lse, delta, dq, T, S, ctx.P, c, scale)
        _launch(getattr(lib, f"attn_dkv_{name}"), (S + BK - 1) // BK, B * H,
                q, k, v, do, lse, delta, dk, dv, T, S, ctx.P, c, scale)
        return dq, dk, dv, None


def metal_attention(q, k, v, P):
    """Causal attention for T queries at absolute positions P..P+T-1 against
    P+T keys - blocked_attention's contract, on the Metal kernel. q, k and v
    share one dtype (bf16, fp16 or fp32) and head_dim 64."""
    return _MetalAttention.apply(q, k, v, P)
