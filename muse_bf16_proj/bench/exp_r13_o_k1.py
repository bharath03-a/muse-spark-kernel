"""r13 short-K / x-heavy GEMM: SS wgmma (W rows on M, tokens on N) with clusters of adjacent W tiles sharing one
TMA-multicast x stream (x read from L2 once per cluster), k-interleaved consumer warpgroups, no split-K.
Also tries the gluon gemv tile at m=1. Spliced into the candidate by exp_r13_o_build.py.
"""


def _r13_gen_wgmma() -> str:
    parts = ["template <int BT>\n__device__ __forceinline__ void r13_mma(float (&d)[BT / 8][4], uint64_t da, uint64_t db);\n"]
    for n in (64, 128, 256):
        nt = n // 8
        regs = ", ".join(f"%{i}" for i in range(n // 2))
        outs = ", ".join(f'"+f"(d[{j}][{r}])' for j in range(nt) for r in range(4))
        parts.append(
            "template <>\n"
            f"__device__ __forceinline__ void r13_mma<{n}>(float (&d)[{nt}][4], uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 "\n'
            f'      "{{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;\\n"\n'
            f"      : {outs}\n"
            '      : "l"(da), "l"(db));\n'
            "}\n")
    return "".join(parts)


R13_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t r13_su(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void r13_cl_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void r13_cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void r13_mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(r13_su(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void r13_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(r13_su(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void r13_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(r13_su(b)) : "memory");
}
__device__ __forceinline__ void r13_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n R13_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra R13_DONE;\n bra R13_WAIT;\n R13_DONE:\n}\n"
      ::"r"(r13_su(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void r13_tma(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(r13_su(dst)), "l"(map), "r"(c0), "r"(c1), "r"(r13_su(bar)) : "memory");
}
__device__ __forceinline__ void r13_tma_h(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
      ::"r"(r13_su(dst)), "l"(map), "r"(c0), "r"(c1), "r"(r13_su(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t r13_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// K-major operand, 128B swizzle, 8-row groups 1024 B apart
__device__ __forceinline__ uint64_t r13_desc(const void* p) {
  return (uint64_t)((r13_su(p) >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void r13_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void r13_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void r13_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
""" + _r13_gen_wgmma() + r"""
__device__ __forceinline__ uint32_t r13_mapa(uint32_t a, int r) {
  uint32_t o;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(o) : "r"(a), "r"(r));
  return o;
}
__device__ __forceinline__ void r13_arrive_cl(uint32_t bar_cl) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" ::"r"(bar_cl) : "memory");
}
__device__ __forceinline__ void r13_tma_mc(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar,
                                           uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%2, %3}], [%4], %5;\n"
      ::"r"(r13_su(dst)), "l"(map), "r"(c0), "r"(c1), "r"(r13_su(bar)), "h"(mask) : "memory");
}

template <int NWG, int BT, int NKW, int MC>
struct R13Geo {
  static constexpr int BN = 64 * NWG;
  static constexpr int NT = BT / 8;
  static constexpr int NCT = NWG * 128;
  static constexpr int NCONS = NCT * NKW;
  static constexpr int THREADS = NCONS + 32;
  static constexpr int XR = BT / MC;                 // x rows this CTA loads and multicasts to the cluster
  static constexpr uint32_t XBOX = BT * 128;
  static constexpr uint32_t WBOX = BN * 128;
  static constexpr uint32_t STB = XBOX + WBOX;       // x first (1024-aligned), then W
  static constexpr int OP = BN + 8;
  static constexpr uint32_t OST = BT * OP * 2;
};

// W rows of MC adjacent tiles form a cluster sharing one x stream: each CTA TMA-multicasts BT/MC token rows of every
// x stage to the whole cluster, so x is read from L2 once per cluster instead of once per CTA.
template <int NWG, int BT, int NKW, int MC, int D>
__global__ void __launch_bounds__(NWG * NKW * 128 + 32, 1)
r13_kernel(const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap xmap,
           bf16* __restrict__ out, int m, int N, int K) {
  using G = R13Geo<NWG, BT, NKW, MC>;
  static_assert(D > NKW && BT % (8 * MC) == 0 && (G::XR * 128) % 1024 == 0, "r13: geometry");
  static_assert(G::OST <= D * G::STB && (NKW - 1) * G::NT * G::NCT * 16 <= D * G::STB, "r13: ring too small");
  constexpr uint32_t BOFF = D * G::STB;
  extern __shared__ __align__(1024) unsigned char r13_raw[];
  unsigned char* sm = r13_raw + ((1024 - (r13_su(r13_raw) & 1023)) & 1023);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + BOFF);
  uint64_t* empty = full + D;

  int rank = 0;
  if constexpr (MC > 1) rank = (int)cg::this_cluster().block_rank();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntiles = N / G::BN;
  const int tile = blockIdx.x % ntiles, tt = blockIdx.x / ntiles;
  const int n0 = tile * G::BN, t0 = tt * BT;
  const int nk = K >> 6;

  if (tid == 0) {
    // every consumer warp of every cluster CTA releases each stage: peers multicast into it
    for (int d = 0; d < D; ++d) { r13_mbar_init(&full[d], 1); r13_mbar_init(&empty[d], NWG * 4 * MC); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if constexpr (MC > 1) {
    r13_cl_arrive_release();
    r13_cl_wait();
  }

  float acc[G::NT][4];
#pragma unroll
  for (int j = 0; j < G::NT; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  if (warp == NWG * NKW * 4) {
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % D;
        if (i >= D) {
          r13_wait(&empty[s], ((i / D) - 1) & 1);
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        unsigned char* st = sm + s * G::STB;
        r13_expect_tx(&full[s], G::STB);  // own W box + the whole x box (all ranks' slices); OOB rows zero-filled
        if constexpr (MC > 1) {
          r13_tma_mc(st + rank * G::XR * 128, &xmap, i * 64, t0 + rank * G::XR, &full[s], (uint16_t)((1 << MC) - 1));
        } else {
          r13_tma(st, &xmap, i * 64, t0, &full[s]);
        }
        r13_tma(st + G::XBOX, &wmap, i * 64, n0, &full[s]);
      }
    }
    __syncwarp();
  } else {
    const int cw = warp >> 2, wg = cw % NWG, wk = cw / NWG;
    uint32_t empty_cl[MC];
#pragma unroll
    for (int q = 0; q < MC; ++q) empty_cl[q] = MC > 1 ? r13_mapa(r13_su(empty), q) : r13_su(empty);
    int it = 0;
    for (int i = wk; i < nk; i += NKW, ++it) {
      const int s = i % D;
      r13_wait(&full[s], (i / D) & 1);
      const unsigned char* st = sm + s * G::STB;
      const uint64_t da = r13_desc(st + G::XBOX + wg * 8192);
      const uint64_t db = r13_desc(st);
      r13_wg_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) r13_mma<BT>(acc, da + (uint64_t)(2 * ks), db + (uint64_t)(2 * ks));
      r13_wg_commit();
      r13_wg_wait<1>();
      if (it > 0 && lane == 0) {
        const uint32_t off = (uint32_t)(((i - NKW) % D) * 8);
#pragma unroll
        for (int q = 0; q < MC; ++q) r13_arrive_cl(empty_cl[q] + off);
      }
    }
    r13_wg_wait<0>();
    if constexpr (NKW > 1) {
      float4* scr = reinterpret_cast<float4*>(sm);
      const int lt = tid - wk * G::NCT;
      asm volatile("bar.sync 2, %0;\n" ::"n"(G::NCONS) : "memory");
      if (wk > 0) {
#pragma unroll
        for (int j = 0; j < G::NT; ++j)
          scr[((wk - 1) * G::NT + j) * G::NCT + lt] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
      }
      asm volatile("bar.sync 2, %0;\n" ::"n"(G::NCONS) : "memory");
      if (wk == 0) {
#pragma unroll
        for (int q = 1; q < NKW; ++q)
#pragma unroll
          for (int j = 0; j < G::NT; ++j) {
            const float4 p = scr[((q - 1) * G::NT + j) * G::NCT + lt];
            acc[j][0] += p.x;
            acc[j][1] += p.y;
            acc[j][2] += p.z;
            acc[j][3] += p.w;
          }
      }
    }
  }
  // all of this CTA's stages are consumed, so no peer multicast still targets its ring
  __syncthreads();
  bf16* ost = reinterpret_cast<bf16*>(sm);
  if (warp < NWG * 4) {
    const int wq = warp & 3, g = lane >> 2, t4 = lane & 3;
    const int nl = (warp >> 2) * 64 + wq * 16 + g;
#pragma unroll
    for (int j = 0; j < G::NT; ++j) {
      const int tl = 8 * j + 2 * t4;
      ost[tl * G::OP + nl] = __float2bfloat16(acc[j][0]);
      ost[(tl + 1) * G::OP + nl] = __float2bfloat16(acc[j][1]);
      ost[tl * G::OP + nl + 8] = __float2bfloat16(acc[j][2]);
      ost[(tl + 1) * G::OP + nl + 8] = __float2bfloat16(acc[j][3]);
    }
    asm volatile("bar.sync 1, %0;\n" ::"n"(G::NCT) : "memory");
    constexpr int V = G::BN / 8;
    for (int e = tid; e < BT * V; e += G::NCT) {
      const int tl = e / V, cv = e % V;
      const int t = t0 + tl;
      if (t < m) {
        *reinterpret_cast<uint4*>(out + (size_t)t * N + n0 + cv * 8) =
            *reinterpret_cast<const uint4*>(ost + tl * G::OP + cv * 8);
      }
    }
  }
  // peers may still arrive on this CTA's empty barriers until they finish their mainloops
  if constexpr (MC > 1) {
    r13_cl_arrive_release();
    r13_cl_wait();
  }
}

inline CUtensorMap r13_map(const void* ptr, int rows, int cols, int row_stride, int box_rows) {
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_stride * sizeof(bf16)};
  cuuint32_t bdim[2] = {64u, (cuuint32_t)box_rows};
  cuuint32_t estride[2] = {1u, 1u};
  CUresult res = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
                                        bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "r13: cuTensorMapEncodeTiled failed ", (int)res);
  return map;
}

template <int NWG, int BT, int NKW, int MC, int D>
cudaLaunchConfig_t r13_cfg(int blocks, cudaLaunchAttribute* attr) {
  using G = R13Geo<NWG, BT, NKW, MC>;
  constexpr size_t smem = 1024 + (size_t)D * G::STB + (size_t)2 * D * 8;
  static_assert(smem <= 232448, "r13: smem");
  auto kern = r13_kernel<NWG, BT, NKW, MC, D>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks, 1, 1);
  cfg.blockDim = dim3(G::THREADS, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  attr->id = cudaLaunchAttributeClusterDimension;
  attr->val.clusterDim.x = MC;
  attr->val.clusterDim.y = 1;
  attr->val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cfg;
}

template <int NWG, int BT, int NKW, int MC, int D>
void r13_launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
  using G = R13Geo<NWG, BT, NKW, MC>;
  const int m = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(m >= 1 && N % (G::BN * MC) == 0 && K % 64 == 0 && w.size(1) == K, "r13: shape");
  TORCH_CHECK(out.size(0) == m && out.size(1) == N && out.is_contiguous() && w.is_contiguous(), "r13: out/w layout");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "r13: x layout");
  const int TT = (m + BT - 1) / BT;
  const CUtensorMap wmap = r13_map(w.data_ptr(), N, K, K, G::BN);
  const CUtensorMap xmap = r13_map(x.data_ptr(), m, K, (int)x.stride(0), G::XR);
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = r13_cfg<NWG, BT, NKW, MC, D>((N / G::BN) * TT, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, r13_kernel<NWG, BT, NKW, MC, D>, wmap, xmap,
                                    static_cast<bf16*>(out.data_ptr()), m, N, K));
}

template <int NWG, int BT, int NKW, int MC, int D>
int64_t r13_clusters() {
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = r13_cfg<NWG, BT, NKW, MC, D>(132 * MC, &attr);
  int n = -1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(r13_kernel<NWG, BT, NKW, MC, D>), &cfg));
  return n;
}

#define R13_CASE(ID, NWG_, BT_, NKW_, MC_, D_)                    \
  if (id == ID) {                                                 \
    if (query) return r13_clusters<NWG_, BT_, NKW_, MC_, D_>();   \
    r13_launch<NWG_, BT_, NKW_, MC_, D_>(x, w, out);              \
    return 0;                                                     \
  }

int64_t launch_r13(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t id, int64_t query) {
R13_CASES
  TORCH_CHECK(false, "r13: config id not compiled ", id);
  return 0;
}
"""
# id -> (NWG, BT, NKW, MC, D): BN = 64 NWG W rows per CTA, BT tokens, NKW k-interleaved warpgroups, MC-CTA cluster of
# adjacent W tiles sharing a TMA-multicast x stream, D-stage ring
R13_CFGS = {
    1: (1, 128, 2, 2, 8),
    2: (1, 128, 2, 4, 8),
    3: (1, 128, 1, 2, 8),
    4: (1, 256, 2, 2, 5),
    5: (1, 256, 2, 4, 5),
    6: (2, 128, 1, 2, 6),
    7: (2, 256, 1, 2, 4),
}
R13_SRC = R13_SRC.replace("R13_CASES", "\n".join(f"  R13_CASE({i}, {', '.join(map(str, v))})" for i, v in R13_CFGS.items()))
R13_CPP = "int64_t launch_r13(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t id, int64_t query);"
# (layer, m) -> candidate ids ("g" = gluon gemv tile reused at m=1), timed in prepare against the current path
R13_PLAN = {
    ("o", 1): ("g",),
    ("o", 128): (1, 2, 3, 6),
    ("o", 256): (4, 5, 7),
    ("unpooler", 128): (1, 2, 6),
    ("unpooler", 256): (4, 5, 7),
}
R13_DET_REPS = 10
R13_BENCH_REPS = 30
R13_MARGIN_US = 0.2


def _r13_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((R13_SRC + " ".join(CUDA_FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_r13_{tag}", cpp_sources=R13_CPP, cuda_sources=R13_SRC, functions=["launch_r13"],
                       extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=CUDA_LDFLAGS)


def _r13_gtile(layer: str):
    tiles = GLUON_TILES.get(layer, ())
    return tiles[0] if tiles else None


def _r13_run(state: dict, x: torch.Tensor, layer: str, cid) -> torch.Tensor:
    w = state[layer]
    out = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
    if cid == "g":
        _gluon_launch(x, out, state, layer, _r13_gtile(layer))
    else:
        state["r13_ext"].launch_r13(x, w, out, cid, 0)
    return out


def _r13_fits(cid, layer: str, n: int, m: int, clusters: dict, state: dict) -> str | None:
    if cid == "g":
        t = _r13_gtile(layer)
        return None if t is not None and state.get("gl_desc") and m <= t.max_m else "no gluon tile"
    nwg, bt, _, mc, _ = R13_CFGS[cid]
    if n % (64 * nwg * mc):
        return "shape"
    need = (n // (64 * nwg * mc)) * ((m + bt - 1) // bt)
    if need > clusters.get(cid, 0):
        return f"needs {need} clusters > {clusters.get(cid, 0)}"
    return None


def _r13_selftest(state: dict, layer: str, m: int, cid, gen: torch.Generator) -> None:
    w = state[layer]
    k = w.shape[1]
    x = torch.randn(m, k, device=w.device, generator=gen).to(torch.bfloat16)
    x[:, ::97] *= 30.0
    ref = x.double() @ w.double().t()
    got = _r13_run(state, x, layer, cid)
    gd = got.double()
    rel = float((gd - ref).norm() / ref.norm())
    mx = float((gd - ref).abs().max() / ref.abs().max())
    if not bool(torch.isfinite(gd).all()) or rel > 4e-3 or mx > 2e-2:
        raise ValueError(f"smoke rel={rel:.2e} max={mx:.2e}")
    for _ in range(R13_DET_REPS):
        if not torch.equal(_r13_run(state, x, layer, cid), got):
            raise ValueError("non-deterministic")


def _r13_time(graphs: dict, flush: torch.Tensor) -> dict:
    import statistics
    res = {nm: [] for nm in graphs}
    for r in range(R13_BENCH_REPS):
        for nm in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.sum()
            a.record()
            graphs[nm].replay()
            b.record()
            b.synchronize()
            res[nm].append(a.elapsed_time(b) * 1e3)
    return {nm: statistics.median(v) for nm, v in res.items()}


def _r13_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _r13_prepare(state: dict) -> None:
    import time
    t_start = time.time()
    state["r13"] = {}
    if state.get("ext") is None:
        return
    try:
        state["r13_ext"] = _r13_build()
        print(f"r13 build {time.time() - t_start:.1f}s")
    except Exception as exc:
        print(f"r13 kernels disabled: {type(exc).__name__}: {str(exc)[-3000:]}")
        state["r13_ext"] = None
    dev = state["pooler"].device
    clusters = {}
    if state["r13_ext"] is not None:
        e = torch.empty(0, device=dev)
        for cid in R13_CFGS:
            try:
                clusters[cid] = int(state["r13_ext"].launch_r13(e, e, e, cid, 1))
            except Exception as exc:
                print(f"r13 id={cid} occupancy query failed: {type(exc).__name__}: {exc}")
                clusters[cid] = 0
        print("r13 max clusters: " + " ".join(f"id{c}{R13_CFGS[c]}={v}" for c, v in clusters.items()))
    gen = torch.Generator(device=dev).manual_seed(13)
    flush = torch.empty(64 * 2**20, device=dev, dtype=torch.float32)
    flush.normal_()
    picks = {}
    for (layer, m), ids in R13_PLAN.items():
        n, k = state[layer].shape
        good = []
        for cid in ids:
            why = _r13_fits(cid, layer, n, m, clusters, state)
            if why:
                print(f"r13 skip {layer} m={m} id={cid}: {why}")
                continue
            try:
                _r13_selftest(state, layer, m, cid, gen)
                good.append(cid)
            except Exception as exc:
                print(f"r13 selftest FAILED {layer} m={m} id={cid}: {type(exc).__name__}: {str(exc)[-1500:]}")
        if not good:
            continue
        try:
            x = torch.randn(m, k, device=dev, generator=gen).to(torch.bfloat16)
            data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
            graphs = {"cur": _r13_graph(lambda: kernel_fn(data, state))}
            for cid in good:
                graphs[f"id{cid}"] = _r13_graph(lambda c=cid: _r13_run(state, x, layer, c))
            med = _r13_time(graphs, flush)
            best = min(good, key=lambda c: med[f"id{c}"])
            win = med[f"id{best}"] < med["cur"] - R13_MARGIN_US
            print(f"r13 bench {layer} m={m} (raw us, ~4 graph overhead): "
                  + " ".join(f"{nm}={v:.2f}" for nm, v in med.items()) + (f" -> id{best}" if win else " -> cur"))
            if win:
                picks[(layer, m)] = best
        except Exception as exc:
            print(f"r13 bench error {layer} m={m}: {type(exc).__name__}: {exc}")
    state["r13"] = picks
    torch.cuda.synchronize()
    print(f"r13 prepare {time.time() - t_start:.1f}s picks={picks}")


def _r13_try(x: torch.Tensor, layer: str, n: int, state: dict):
    cid = state.get("r13", {}).get((layer, x.shape[0]))
    if cid is None or n != state[layer].shape[0] or x.shape[1] != state[layer].shape[1]:
        return None
    return _r13_run(state, x, layer, cid)
