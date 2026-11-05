# okc: problem=muse_moe_w4a16
"""Own W4A16 (NVFP4 weights) MoE in hand-written CUDA, sm_90a.

Weight-stationary warps: each warp streams RB 16-row weight blocks over the full K=4096 and
multiplies them against NT 8-token tiles with mma.sync m16n8k16 f16. Weights are repacked so a
lane's 16-byte load is exactly its A fragments for one 64-wide k chunk, and the k order inside a
chunk is permuted so the lane's B fragments are one contiguous 32-byte run of activations and its
scale group is a single e4m3 byte per row. FP4 -> f16 is pure bit placement (mags into the
exponent/mantissa, sign to bit 15), the 2^-14 offset is folded into the epilogue.
GEMM1 rows are gate/up interleaved per 8 neurons, so SiLU*up happens in registers.
"""
import os
import time
from typing import TypedDict

import torch


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


TOP_K = 8
EPS = 1e-8
DIM = 4096
# (RB1, RB2, NT): row blocks per warp for GEMM1/GEMM2, 8-token tiles per warp.
CONFIGS = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 2, 2), (2, 1, 2), (2, 2, 4), (1, 1, 4)]
BUCKETS = (1, 8, 16, 32, 128, 256)
DEFAULT = {1: (1, 1, 1), 8: (1, 1, 1), 16: (1, 1, 2), 32: (1, 1, 2), 128: (2, 2, 4), 256: (2, 2, 4)}
SWEEP_BUDGET_S = float(os.environ.get("OWN_V5_SWEEP_S", "90"))

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace own5 {
constexpr int K = 4096;
constexpr int NCH = K / 64;
constexpr int NIT = NCH / 4;
constexpr int TOPK = 8;
constexpr int NLOC_MAX = 256;

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ld_stream2(const uint2* p) {
  uint2 r;
  asm("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// e4m3 pair -> f16x2 scaled by 128 (keeps 448*128 inside f16 range).
__device__ __forceinline__ uint32_t scale_pair(uint32_t b16) {
  uint32_t d;
  asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(d) : "r"(b16));
  return hmul2(d, 0x58005800u);
}
__device__ __forceinline__ void deq(uint32_t q, uint32_t sg, uint32_t sg8, uint32_t (&a)[4]) {
  a[0] = hmul2(q & 0x8E008E00u, sg);
  a[1] = hmul2(((q << 3) & 0x0E000E00u) | ((q << 1) & 0x80008000u), sg8);
  a[2] = hmul2(((q << 6) & 0x0E000E00u) | ((q << 2) & 0x80008000u), sg);
  a[3] = hmul2(((q << 9) & 0x0E000E00u) | ((q << 3) & 0x80008000u), sg8);
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pick(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// tiles: [3][tmax] = expert (-1 empty), row0, rowend over the expert-sorted pair rows.
template <int RB, int NT, bool GLU>
__global__ void __launch_bounds__(128) gemm(
    const uint4* __restrict__ Wp, const uint2* __restrict__ Sp, const float* __restrict__ gs,
    const __half* __restrict__ X, const int* __restrict__ rowmap, const int* __restrict__ tiles,
    int tmax, int nrb, int lo, int out_w, __half* __restrict__ out_h, float* __restrict__ out_f) {
  const int t = blockIdx.x;
  const int e = tiles[t];
  if (e < 0) return;
  const int row0 = tiles[tmax + t], rend = tiles[2 * tmax + t];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rbase = (blockIdx.y * 4 + warp) * RB;
  if (rbase >= nrb) return;
  const int g = lane >> 2, c = lane & 3;
  const int eg = lo + e;

  const uint4* xp[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    int r = row0 + n * 8 + g;
    r = r < rend ? r : row0;
    const int xr = rowmap ? rowmap[r] : r;
    xp[n] = reinterpret_cast<const uint4*>(X + (size_t)xr * K) + c * 2;
  }
  const uint4* wp[RB];
  const uint2* sp[RB];
#pragma unroll
  for (int r = 0; r < RB; ++r) {
    const size_t blk = (size_t)eg * nrb + rbase + r;
    wp[r] = Wp + blk * NCH * 32 + lane;
    sp[r] = Sp + blk * NIT * 32 + lane;
  }

  float acc[RB][NT][4];
#pragma unroll
  for (int r = 0; r < RB; ++r)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[r][n][j] = 0.f;

  uint4 w[4][RB];
  uint2 s[RB];
#pragma unroll
  for (int r = 0; r < RB; ++r) {
#pragma unroll
    for (int u = 0; u < 4; ++u) w[u][r] = ld_stream(wp[r] + u * 32);
    s[r] = ld_stream2(sp[r]);
  }

#pragma unroll 1
  for (int it = 0; it < NIT; ++it) {
    const int nx = it + 1 < NIT ? it + 1 : it;
    uint4 wn[4][RB];
    uint2 sn[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) {
#pragma unroll
      for (int u = 0; u < 4; ++u) wn[u][r] = ld_stream(wp[r] + (nx * 4 + u) * 32);
      sn[r] = ld_stream2(sp[r] + nx * 32);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int ch = it * 4 + u;
      uint4 xa[NT], xb[NT];
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        xa[n] = __ldg(xp[n] + ch * 8);
        xb[n] = __ldg(xp[n] + ch * 8 + 1);
      }
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const uint32_t sw = (u < 2 ? s[r].x : s[r].y) >> (16 * (u & 1));
        const uint32_t sc = scale_pair(sw & 0xFFFFu);
        const uint32_t sg = __byte_perm(sc, 0, 0x1010), sg8 = __byte_perm(sc, 0, 0x3232);
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t a[4];
          deq(pick(w[u][r], st), sg, sg8, a);
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            const uint4& xv = st < 2 ? xa[n] : xb[n];
            mma(acc[r][n], a, pick(xv, (st & 1) * 2), pick(xv, (st & 1) * 2 + 1));
          }
        }
      }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) {
#pragma unroll
      for (int u = 0; u < 4; ++u) w[u][r] = wn[u][r];
      s[r] = sn[r];
    }
  }

  // f16 weights carry e2m1*sf*128*2^-14 = true/(gs*128).
  const float f = gs[eg] * 128.f;
#pragma unroll
  for (int r = 0; r < RB; ++r) {
    const int rb = rbase + r;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int row = row0 + n * 8 + c * 2 + j;
        if (row >= rend) continue;
        if (GLU) {
          const float gt = acc[r][n][j] * f, up = acc[r][n][2 + j] * f;
          const float h = gt / (1.f + __expf(-gt)) * up;
          out_h[(size_t)row * out_w + rb * 8 + g] = __float2half_rn(h);
        } else {
          out_f[(size_t)row * out_w + rb * 16 + g] = acc[r][n][j] * f;
          out_f[(size_t)row * out_w + rb * 16 + g + 8] = acc[r][n][2 + j] * f;
        }
      }
    }
  }
}

// Block 0 routes; blocks 1..T convert one token row of x to f16.
__global__ void __launch_bounds__(1024) route(
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ x, __half* __restrict__ x16,
    int* __restrict__ tok, int* __restrict__ inv, int* __restrict__ tiles, int P, int lo, int nloc,
    int ts, int tmax) {
  if (blockIdx.x > 0) {
    const int row = blockIdx.x - 1;
    for (int i = threadIdx.x; i < K / 8; i += blockDim.x) {
      const uint4 v = reinterpret_cast<const uint4*>(x + (size_t)row * K)[i];
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
      uint4 o;
      __half2* h = reinterpret_cast<__half2*>(&o);
#pragma unroll
      for (int j = 0; j < 4; ++j) h[j] = __float22half2_rn(__bfloat1622float2(b[j]));
      reinterpret_cast<uint4*>(x16 + (size_t)row * K)[i] = o;
    }
    return;
  }
  __shared__ int cnt[NLOC_MAX], start[NLOC_MAX];
  for (int i = threadIdx.x; i < nloc; i += blockDim.x) cnt[i] = 0;
  __syncthreads();
  constexpr int PER = 4;
  int rk[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int p = threadIdx.x + j * blockDim.x;
    rk[j] = -1;
    if (p < P) {
      const int el = ids[p] - lo;
      if (el >= 0 && el < nloc) rk[j] = atomicAdd(&cnt[el], 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0, nt = 0;
    for (int el = 0; el < nloc; ++el) {
      start[el] = acc;
      for (int r0 = acc; r0 < acc + cnt[el]; r0 += ts) {
        if (nt < tmax) {
          tiles[nt] = el;
          tiles[tmax + nt] = r0;
          tiles[2 * tmax + nt] = min(r0 + ts, acc + cnt[el]);
        }
        ++nt;
      }
      acc += cnt[el];
    }
    for (; nt < tmax; ++nt) tiles[nt] = -1;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int p = threadIdx.x + j * blockDim.x;
    if (p < P) {
      if (rk[j] >= 0) {
        const int pos = start[ids[p] - lo] + rk[j];
        inv[p] = pos;
        tok[pos] = p / TOPK;
      } else {
        inv[p] = -1;
      }
    }
  }
}

__global__ void __launch_bounds__(256) combine(
    const float* __restrict__ y, const int* __restrict__ inv, const float* __restrict__ tw,
    const __nv_bfloat16* __restrict__ pw, __nv_bfloat16* __restrict__ out, float eps) {
  constexpr int PT = K / 256;
  __shared__ float red[8];
  const int t = blockIdx.x;
  float scale[PT], acc[PT];
#pragma unroll
  for (int i = 0; i < PT; ++i) {
    scale[i] = 1.f + __bfloat162float(pw[threadIdx.x + i * 256]);
    acc[i] = 0.f;
  }
  for (int s = 0; s < TOPK; ++s) {
    const int pos = inv[t * TOPK + s];
    if (pos < 0) continue;
    float v[PT], ss = 0.f;
#pragma unroll
    for (int i = 0; i < PT; ++i) {
      v[i] = y[(size_t)pos * K + threadIdx.x + i * 256] * scale[i];
      ss += v[i] * v[i];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += red[i];
    __syncthreads();
    const float m = rsqrtf(tot / K + eps) * tw[t * TOPK + s];
#pragma unroll
    for (int i = 0; i < PT; ++i) acc[i] += v[i] * m;
  }
#pragma unroll
  for (int i = 0; i < PT; ++i) out[(size_t)t * K + threadIdx.x + i * 256] = __float2bfloat16(acc[i]);
}
}  // namespace own5
"""

HOST_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
using namespace own5;

template <int RB, int NT, bool GLU>
static void launch_gemm(const torch::Tensor& Wp, const torch::Tensor& Sp, const torch::Tensor& gs,
                        const torch::Tensor& X, const int* rowmap, const torch::Tensor& tiles, int tmax,
                        int nrb, int lo, int out_w, __half* oh, float* of, cudaStream_t st) {
  dim3 grid(tmax, (nrb + 4 * RB - 1) / (4 * RB));
  gemm<RB, NT, GLU><<<grid, 128, 0, st>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()), reinterpret_cast<const uint2*>(Sp.data_ptr()),
      gs.data_ptr<float>(), reinterpret_cast<const __half*>(X.data_ptr()), rowmap, tiles.data_ptr<int>(),
      tmax, nrb, lo, out_w, oh, of);
}

#define DISPATCH(RB_, NT_, GLU_, ...)                                                   \
  if (rb == RB_ && nt == NT_) { launch_gemm<RB_, NT_, GLU_>(__VA_ARGS__); return; }

template <bool GLU>
static void gemm_any(int rb, int nt, const torch::Tensor& Wp, const torch::Tensor& Sp,
                     const torch::Tensor& gs, const torch::Tensor& X, const int* rowmap,
                     const torch::Tensor& tiles, int tmax, int nrb, int lo, int out_w, __half* oh,
                     float* of, cudaStream_t st) {
  DISPATCH(1, 1, GLU, Wp, Sp, gs, X, rowmap, tiles, tmax, nrb, lo, out_w, oh, of, st)
  DISPATCH(1, 2, GLU, Wp, Sp, gs, X, rowmap, tiles, tmax, nrb, lo, out_w, oh, of, st)
  DISPATCH(1, 4, GLU, Wp, Sp, gs, X, rowmap, tiles, tmax, nrb, lo, out_w, oh, of, st)
  DISPATCH(2, 1, GLU, Wp, Sp, gs, X, rowmap, tiles, tmax, nrb, lo, out_w, oh, of, st)
  DISPATCH(2, 2, GLU, Wp, Sp, gs, X, rowmap, tiles, tmax, nrb, lo, out_w, oh, of, st)
  DISPATCH(2, 4, GLU, Wp, Sp, gs, X, rowmap, tiles, tmax, nrb, lo, out_w, oh, of, st)
  TORCH_CHECK(false, "unsupported rb/nt ", rb, " ", nt);
}

torch::Tensor run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w13p,
                  torch::Tensor s13p, torch::Tensor gs13, torch::Tensor w2p, torch::Tensor s2p,
                  torch::Tensor gs2, torch::Tensor pw, int64_t lo, int64_t nloc, int64_t rb1,
                  int64_t rb2, int64_t nt) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.size(1) == K, "x must be [T,4096] bf16 cuda");
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 && tw.scalar_type() == torch::kFloat32, "ids int32, tw fp32");
  TORCH_CHECK(nloc <= NLOC_MAX, "too many local experts");
  const int T = x.size(0), P = ids.numel();
  TORCH_CHECK(P <= 1024 * 4, "too many pairs");
  auto st = c10::cuda::getCurrentCUDAStream();
  auto oi = x.options().dtype(torch::kInt32);
  const int ts = 8 * nt;
  const int tmax = std::max(1, std::min<int>(P, (P + ts - 1) / ts + (int)nloc));
  auto x16 = torch::empty({T, K}, x.options().dtype(torch::kHalf));
  auto tok = torch::empty({std::max(P, 1)}, oi);
  auto inv = torch::empty({std::max(P, 1)}, oi);
  auto tiles = torch::empty({3 * tmax}, oi);
  route<<<1 + T, 1024, 0, st>>>(ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                reinterpret_cast<__half*>(x16.data_ptr()), tok.data_ptr<int>(),
                                inv.data_ptr<int>(), tiles.data_ptr<int>(), P, lo, nloc, ts, tmax);
  auto inter = torch::empty({std::max(P, 1), K}, x.options().dtype(torch::kHalf));
  gemm_any<true>(rb1, nt, w13p, s13p, gs13, x16, tok.data_ptr<int>(), tiles, tmax, K / 8, lo, K,
                 reinterpret_cast<__half*>(inter.data_ptr()), nullptr, st);
  auto y = torch::empty({std::max(P, 1), K}, x.options().dtype(torch::kFloat32));
  gemm_any<false>(rb2, nt, w2p, s2p, gs2, inter, nullptr, tiles, tmax, K / 16, lo, K, nullptr,
                  y.data_ptr<float>(), st);
  auto out = torch::empty({T, K}, x.options());
  combine<<<T, 256, 0, st>>>(y.data_ptr<float>(), inv.data_ptr<int>(), tw.data_ptr<float>(),
                             reinterpret_cast<const __nv_bfloat16*>(pw.data_ptr()),
                             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), 1e-8f);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        _EXT = load_inline(
            name="own_moe_v5_ext", cpp_sources="", cuda_sources=CUDA_SRC + HOST_SRC,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math",
                               "-std=c++20"],
            verbose=False)
        print(f"[v5] compiled in {time.time() - t0:.1f}s", flush=True)
    return _EXT


def _pack_weights(codes_u8: torch.Tensor) -> torch.Tensor:
    """[N, K/2] packed e2m1 (low nibble first) -> [N/16, K/64, 32 lanes, 4] int32 fragment words."""
    n, kh = codes_u8.shape
    k = kh * 2
    codes = torch.stack((codes_u8 & 15, codes_u8 >> 4), dim=-1).view(n, k).to(torch.int64)
    # physical k = 64*ch + 16*c + 4*s + 2*ih + il ; row = 8*h + g
    v = codes.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2)  # nb,h,g,ch,c,s,ih,il
    v = v.permute(0, 3, 2, 4, 5, 6, 1, 7)  # nb,ch,g,c,s,ih,h,il
    reg = (torch.arange(2, device=v.device).view(2, 1, 1) * 2 + torch.arange(2, device=v.device).view(1, 2, 1))
    il = torch.arange(2, device=v.device).view(1, 1, 2)
    mag_pos = 9 - 3 * reg + 16 * il
    sgn_pos = 15 - reg + 16 * il
    words = ((v & 7) << mag_pos) | ((v >> 3) << sgn_pos)
    words = words.sum(dim=(-3, -2, -1))  # disjoint bits
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.reshape(n // 16, k // 64, 32, 4).contiguous()


def _pack_scales(sf: torch.Tensor) -> torch.Tensor:
    """[N, K/16] e4m3 -> [N/16, K/256, 32 lanes, 4 chunks, 2 rows] bytes."""
    n, kg = sf.shape
    b = sf.view(torch.uint8).view(n // 16, 2, 8, kg // 16, 4, 4)  # nb,h,g,it,u,c
    return b.permute(0, 3, 2, 5, 4, 1).contiguous()  # nb,it,g,c,u,h


def _glu_rows(n2: int, dev) -> torch.Tensor:
    i = n2 // 2
    j = torch.arange(i // 8, device=dev).view(-1, 1, 1) * 8
    r = torch.arange(8, device=dev).view(1, 1, 8)
    half = torch.tensor([0, i], device=dev).view(1, 2, 1)
    return (j + half + r).reshape(-1)


def _repack(weights: Weights, n_exp: int):
    dev = weights["w13"].device
    perm = _glu_rows(weights["w13"].shape[1], dev)
    w13p, s13p, w2p, s2p = [], [], [], []
    for e in range(n_exp):
        w13p.append(_pack_weights(weights["w13"][e].index_select(0, perm)))
        s13p.append(_pack_scales(weights["w13_sf"][e].view(torch.uint8).index_select(0, perm)))
        w2p.append(_pack_weights(weights["w2"][e]))
        s2p.append(_pack_scales(weights["w2_sf"][e]))
    return torch.stack(w13p), torch.stack(s13p), torch.stack(w2p), torch.stack(s2p)


def _bucket(m: int) -> int:
    for b in BUCKETS:
        if m <= b:
            return b
    return BUCKETS[-1]


def _run(data, state, cfg):
    rb1, rb2, nt = cfg
    c = data["config"]
    return state["ext"].run(
        data["hidden_states"].contiguous(), data["topk_ids"].to(torch.int32).contiguous(),
        data["topk_weights"].float().contiguous(), state["w13p"], state["s13p"], state["gs13"],
        state["w2p"], state["s2p"], state["gs2"], state["pw"], c["local_start"], c["local_count"],
        rb1, rb2, nt)


def _synthetic(bs, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    n_exp, nloc = 256, 32
    x = torch.randn((bs, DIM), generator=g, device=dev).to(torch.bfloat16)
    pop = 0.03 * torch.randn(n_exp, generator=g, device=dev) / (
        -torch.rand(n_exp, generator=g, device=dev).clamp(min=1e-12).log()).sqrt()
    scores = torch.rand((bs, n_exp), generator=g, device=dev) + pop
    top = scores.topk(TOP_K, dim=-1)
    ids = top.indices
    load = torch.stack([((ids // nloc) == r).sum() for r in range(n_exp // nloc)])
    ids = (ids - nloc * load.argmax()) % n_exp
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": DIM, "d_expert": DIM, "n_experts": n_exp,
                       "local_start": 0, "local_count": nloc, "top_k": TOP_K, "sharding": "ep8"}}


def _time(data, state, cfg, flush, reps=8):
    _run(data, state, cfg)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(data, state, cfg)
    ts = []
    for _ in range(reps):
        flush.add_(1)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    del graph
    return sorted(ts)[len(ts) // 2]


def _sweep(state, dev):
    t0 = time.perf_counter()
    chosen = dict(DEFAULT)
    flush = torch.empty(64 << 20, dtype=torch.int32, device=dev)
    for bs in BUCKETS:
        data = _synthetic(bs, dev, 1234 + bs)
        best = None
        for cfg in CONFIGS:
            if time.perf_counter() - t0 > SWEEP_BUDGET_S:
                break
            t = _time(data, state, cfg, flush)
            if best is None or t < best[0]:
                best = (t, cfg)
        if best is not None:
            chosen[bs] = best[1]
            print(f"[v5] sweep bs={bs} -> {best[1]} {best[0] * 1e3:.1f} us", flush=True)
    return chosen


def prepare(weights: Weights):
    ext = _ext()
    n_exp = weights["w13"].shape[0]
    w13p, s13p, w2p, s2p = _repack(weights, n_exp)
    state = {
        "ext": ext, "w13p": w13p, "s13p": s13p, "w2p": w2p, "s2p": s2p,
        "gs13": weights["w13_gs"].float().contiguous(), "gs2": weights["w2_gs"].float().contiguous(),
        "pw": weights["post_norm_w"].to(torch.bfloat16).contiguous(),
    }
    torch.cuda.synchronize()
    state["cfg"] = _sweep(state, weights["w13"].device) if SWEEP_BUDGET_S > 0 else dict(DEFAULT)
    return state


def kernel_fn(data: Inputs, state):
    return _run(data, state, state["cfg"][_bucket(data["hidden_states"].shape[0])])
