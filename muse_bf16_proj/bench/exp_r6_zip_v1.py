"""v1: register-direct ZipGEMM 12-bit (no smem ring). Packer/format shared with exp_r6_zip_v2."""
from exp_r6_zip_v2 import REBASE, pack, pick_eb, reference_out, unpack  # noqa: F401

ZIP_SRC = r"""
__device__ __forceinline__ uint4 zip_ld_w(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t zip_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t zip_lop(uint32_t a) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(0x87878787u), "r"(0x38383838u));
  return r;
}
__device__ __forceinline__ void zip_mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int NT, int RB>
struct ZipStage {
  uint4 w[RB][3];
  uint4 x[NT][2];
};

template <int NT, int RB, int CS, int WARPS, int STAGES>
__global__ void __launch_bounds__((WARPS + 1) * 32, 1)
zip_kernel(const uint4* __restrict__ wp, const bf16* __restrict__ x, bf16* __restrict__ out,
           const int* __restrict__ rowptr, const int* __restrict__ esck, const float* __restrict__ escd,
           int m, int N, int K, int xs, float scale, long wrap_u4) {
  constexpr int BT = NT * 8, R = RB * 16, RP = R + 4, NTHR = (WARPS + 1) * 32;
  constexpr int NV = (BT * R + NTHR - 1) / NTHR;
  extern __shared__ __align__(16) unsigned char zsm[];
  float* red = reinterpret_cast<float*>(zsm);
  float* fixs = red + WARPS * BT * RP;
  float* tot = fixs + BT * R;

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = CS > 1 ? (int)cluster.block_rank() : 0;
  if constexpr (CS > 1) cluster_arrive_relaxed();

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rg = blockIdx.x / CS;
  const int U = K / 64, UH = U / CS;

  if (warp == WARPS) {
    if (rank == 0) {
      for (int row = lane; row < R; row += 32) {
        float f[BT];
#pragma unroll
        for (int tk = 0; tk < BT; ++tk) f[tk] = 0.f;
        const int gr = rg * R + row;
        const int e1 = rowptr[gr + 1];
        for (int e = rowptr[gr]; e < e1; ++e) {
          const int kk = esck[e];
          const float d = escd[e];
#pragma unroll
          for (int tk = 0; tk < BT; ++tk)
            if (tk < m) f[tk] += d * __bfloat162float(x[(long)tk * xs + kk]);
        }
#pragma unroll
        for (int tk = 0; tk < BT; ++tk) fixs[tk * R + row] = f[tk];
      }
    }
  } else {
    float acc[RB][NT][4];
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[rb][nt][i] = 0.f;

    ZipStage<NT, RB> st[STAGES];
    const int niter = UH / WARPS;

    auto ldstage = [&](ZipStage<NT, RB>& s, int i) {
      const int u = rank * UH + warp + i * WARPS;
      long ub = ((long)rg * U + u) * (RB * 96);
      if (wrap_u4) ub %= wrap_u4;
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        const uint4* p = wp + ub + rb * 96 + lane;
        s.w[rb][0] = zip_ld_w(p);
        s.w[rb][1] = zip_ld_w(p + 32);
        s.w[rb][2] = zip_ld_w(p + 64);
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int tok = nt * 8 + g;
        if (tok < m) {
          const uint4* xp = reinterpret_cast<const uint4*>(x + (long)tok * xs + 64 * u + 8 * t);
          s.x[nt][0] = __ldg(xp);
          s.x[nt][1] = __ldg(xp + 4);
        } else {
          s.x[nt][0] = make_uint4(0, 0, 0, 0);
          s.x[nt][1] = make_uint4(0, 0, 0, 0);
        }
      }
    };

    auto compute = [&](const ZipStage<NT, RB>& s) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
          const uint32_t a0w = reinterpret_cast<const uint32_t*>(&s.w[rb][0])[j];
          const uint32_t a1w = reinterpret_cast<const uint32_t*>(&s.w[rb][1])[j];
          const uint32_t bw = reinterpret_cast<const uint32_t*>(&s.w[rb][2])[j];
          const uint32_t h0 = zip_lop(bw);
          const uint32_t h1 = zip_lop(__funnelshift_l(bw, bw, 4));
          const uint32_t r0 = zip_prmt(a0w, h0, 0x5140);
          const uint32_t r1 = zip_prmt(a0w, h0, 0x7362);
          const uint32_t r2 = zip_prmt(a1w, h1, 0x5140);
          const uint32_t r3 = zip_prmt(a1w, h1, 0x7362);
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            const uint32_t* xv = reinterpret_cast<const uint32_t*>(&s.x[nt][j >> 1]);
            zip_mma(acc[rb][nt], r0, r1, r2, r3, xv[(j & 1) * 2], xv[(j & 1) * 2 + 1]);
          }
        }
      }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
      if (s < niter) ldstage(st[s], s);
    for (int i0 = 0; i0 < niter; i0 += STAGES) {
#pragma unroll
      for (int s = 0; s < STAGES; ++s) {
        const int i = i0 + s;
        if (i < niter) {
          if (i + STAGES - 1 < niter) ldstage(st[(s + STAGES - 1) % STAGES], i + STAGES - 1);
          compute(st[s]);
        }
      }
    }

#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = rb * 16 + g, tk = nt * 8 + 2 * t;
        float* d = red + warp * BT * RP;
        d[tk * RP + row] = acc[rb][nt][0];
        d[(tk + 1) * RP + row] = acc[rb][nt][1];
        d[tk * RP + row + 8] = acc[rb][nt][2];
        d[(tk + 1) * RP + row + 8] = acc[rb][nt][3];
      }
  }
  __syncthreads();

  float sv[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int idx = tid + v * NTHR;
    float s = 0.f;
    if (idx < BT * R) {
      const int tk = idx / R, row = idx % R;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) s += red[w * BT * RP + tk * RP + row];
    }
    sv[v] = s;
  }

  if constexpr (CS > 1) {
    cluster_wait();
    if (rank != 0) {
      float* dst = cluster.map_shared_rank(tot, 0) + (rank - 1) * BT * R;
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        const int idx = tid + v * NTHR;
        if (idx < BT * R) dst[idx] = sv[v];
      }
    }
    cluster_arrive_release();
    cluster_wait();
    if (rank != 0) return;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const int idx = tid + v * NTHR;
      if (idx < BT * R) {
#pragma unroll
        for (int c = 0; c < CS - 1; ++c) sv[v] += tot[c * BT * R + idx];
      }
    }
  }

#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int idx = tid + v * NTHR;
    if (idx < BT * R) {
      const int tk = idx / R, row = idx % R;
      if (tk < m) out[(long)tk * N + rg * R + row] = __float2bfloat16(sv[v] * scale + fixs[idx]);
    }
  }
}

template <int NT, int RB, int CS, int WARPS, int STAGES>
void zip_launch(const torch::Tensor& x, const torch::Tensor& wp, torch::Tensor& out, const torch::Tensor& rowptr,
                const torch::Tensor& esck, const torch::Tensor& escd, double scale, int64_t wrap_u4) {
  constexpr int BT = NT * 8, R = RB * 16, RP = R + 4;
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(m <= BT && N % R == 0 && (K / 64) % (CS * WARPS) == 0, "zip: shape not supported");
  const int smem = (WARPS * BT * RP + BT * R + (CS > 1 ? CS - 1 : 1) * BT * R) * (int)sizeof(float);
  auto kern = zip_kernel<NT, RB, CS, WARPS, STAGES>;
  set_attrs(kern, smem);
  const dim3 grid((N / R) * CS, 1, 1);
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = cluster_cfg(grid, (WARPS + 1) * 32, smem, CS, &attr);
  if (CS == 1) cfg.numAttrs = 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, static_cast<const uint4*>(wp.data_ptr()),
                                    static_cast<const bf16*>(x.data_ptr()), static_cast<bf16*>(out.data_ptr()),
                                    static_cast<const int*>(rowptr.data_ptr()), static_cast<const int*>(esck.data_ptr()),
                                    static_cast<const float*>(escd.data_ptr()), m, N, K, (int)x.stride(0),
                                    (float)scale, (long)wrap_u4));
}

#define ZIP_CASE(NT_, RB_, CS_, W_, S_) \
  if (nt == NT_ && rb == RB_ && cs == CS_ && warps == W_ && stages == S_) { \
    zip_launch<NT_, RB_, CS_, W_, S_>(x, wp, out, rowptr, esck, escd, scale, wrap_u4); return; }

void launch_zip(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor rowptr, torch::Tensor esck,
                torch::Tensor escd, double scale, int64_t wrap_u4, int64_t nt, int64_t rb, int64_t cs,
                int64_t warps, int64_t stages) {
  ZIP_CASE(1, 4, 2, 8, 2)
  ZIP_CASE(1, 4, 2, 8, 3)
  ZIP_CASE(1, 4, 4, 8, 2)
  ZIP_CASE(1, 4, 4, 8, 3)
  ZIP_CASE(1, 2, 2, 8, 2)
  ZIP_CASE(1, 2, 4, 8, 2)
  ZIP_CASE(1, 2, 1, 8, 2)
  TORCH_CHECK(false, "zip: config not compiled nt=", nt, " rb=", rb, " cs=", cs, " w=", warps, " s=", stages);
}
"""

ZIP_CPP = ("void launch_zip(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor rowptr,"
           " torch::Tensor esck, torch::Tensor escd, double scale, int64_t wrap_u4, int64_t nt, int64_t rb,"
           " int64_t cs, int64_t warps, int64_t stages);")

PLAN = [
    ("pooler", 1, (1, 4, 2, 8, 2), None),
    ("pooler", 1, (1, 4, 2, 8, 3), None),
    ("pooler", 1, (1, 2, 4, 8, 2), None),
    ("qkv_gate", 1, (1, 4, 4, 8, 2), None),
    ("qkv_gate", 1, (1, 2, 2, 8, 2), None),
    ("qkv_gate", 1, (1, 2, 4, 8, 2), None),
    ("qkv_gate", 1, (1, 4, 4, 8, 3), None),
]
