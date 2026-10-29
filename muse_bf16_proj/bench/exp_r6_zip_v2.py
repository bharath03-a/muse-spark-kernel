"""ZipGEMM-style 12-bit lossless bf16 GEMV/skinny GEMM (CUDA C++, sm_90a): source + packer + CPU emulation.

Weight W [N, K] bf16 -> per (row-group rg of 16*RB rows, 64-k unit u, row-block rb) one 1536 B record:
  [A0: 32 lanes x 16 B][A1: 32 x 16 B][B: 32 x 16 B]; lane = g*4+t (mma m16n8k16 A-fragment owner).
  Lane's 16 regs of a unit = 4 mma k-steps j x {a0..a3}; a = 2p+h (h: row g / g+8, p: k-pair 2j / 2j+1).
  A_p word j = low bytes [h0e, h0o, h1e, h1o]; B word j byte i: s(W_i)@7, c(W_i)@0..2, s(X_i)@3, c(X_{i+1})@4..6
  with W_i = A0 weights, X_i = A1 weights. High byte = s<<7 | 0x38 | c (exponent rebased by 2^(112-2eb));
  the accumulator is scaled by 2^(2eb-112) at the end. Escapes (E>>1 outside the 8-wide window) decode to
  2^-15 and are fixed up from a per-row CSR list (delta * x[k]) by a dedicated warp.
  k offset in unit of (chunk c, thread t, pair qq, elem eo) = 32c + 8t + 2qq + eo; pair q = 4c+qq = 2j+p.
"""

from __future__ import annotations

import torch

REBASE = 112

ZIP_SRC = r"""
// ===================== ZipGEMM 12-bit (bulk-copy ring, mma.sync) =====================
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

__device__ __forceinline__ void zip_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void zip_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}

constexpr int ZIP_XCH = 4;

template <int NT, int RB, int CS, int WARPS, int S>
__global__ void __launch_bounds__((WARPS + 2) * 32, 1)
zip_kernel(const uint4* __restrict__ wp, const bf16* __restrict__ x, bf16* __restrict__ out,
           const int* __restrict__ rowptr, const int* __restrict__ esck, const float* __restrict__ escd,
           int m, int N, int K, int xstride, float scale, long wrap_u4) {
  constexpr int BT = NT * 8, R = RB * 16, RP = R + 4, NTHR = (WARPS + 2) * 32, SLOT = RB * 1536;
  constexpr int NV = (BT * R + NTHR - 1) / NTHR;
  extern __shared__ __align__(128) unsigned char zsm[];
  const int U = K / 64, UH = U / CS;
  const int XSTR = UH * 128 + 64;
  uint64_t* full = reinterpret_cast<uint64_t*>(zsm);
  uint64_t* empty = full + S;
  uint64_t* xbar = empty + S;
  float* fixs = reinterpret_cast<float*>(zsm + 512);
  float* tot = fixs + BT * R;
  unsigned char* xs = reinterpret_cast<unsigned char*>(tot + (CS > 1 ? CS - 1 : 1) * BT * R);
  unsigned char* ring = xs + (size_t)BT * XSTR;
  float* red = reinterpret_cast<float*>(xs);

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = CS > 1 ? (int)cluster.block_rank() : 0;
  if constexpr (CS > 1) cluster_arrive_relaxed();

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rg = blockIdx.x / CS;

  if (tid == 0) {
    for (int s = 0; s < S; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 1); }
    for (int c = 0; c < ZIP_XCH; ++c) mbar_init(&xbar[c], 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  float acc[RB][NT][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[rb][nt][i] = 0.f;

  if (warp == WARPS) {
    if (lane == 0) {
      const uint32_t cb = UH * 32;
      for (int n = 0; n < UH; ++n) {
        const int s = n % S;
        if (n >= S) mbar_wait(&empty[s], ((n / S) - 1) & 1);
        mbar_expect_tx(&full[s], SLOT);
        long ub = ((long)rg * U + rank * UH + n) * (RB * 96);
        if (wrap_u4) ub %= wrap_u4;
        zip_bulk(ring + (size_t)s * SLOT, wp + ub, SLOT, &full[s]);
        if (n == 1) {
          for (int c = 0; c < ZIP_XCH; ++c) {
            mbar_expect_tx(&xbar[c], cb * m);
            for (int tk = 0; tk < m; ++tk)
              zip_bulk(xs + (size_t)tk * XSTR + c * cb, x + (long)tk * xstride + rank * UH * 64 + c * UH * 16, cb,
                       &xbar[c]);
          }
        }
      }
    }
  } else if (warp == WARPS + 1) {
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
            if (tk < m) f[tk] += d * __bfloat162float(x[(long)tk * xstride + kk]);
        }
#pragma unroll
        for (int tk = 0; tk < BT; ++tk) fixs[tk * R + row] = f[tk];
      }
    }
  } else {
    for (int n = warp; n < UH; n += WARPS) {
      const int s = n % S;
      mbar_wait(&full[s], (n / S) & 1);
      mbar_wait(&xbar[(n * ZIP_XCH) / UH], 0);
      const uint4* sp = reinterpret_cast<const uint4*>(ring + (size_t)s * SLOT) + lane;
      uint4 xv[NT][2];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const uint4* xp = reinterpret_cast<const uint4*>(xs + (size_t)(nt * 8 + g) * XSTR + n * 128 + 16 * t);
        xv[nt][0] = xp[0];
        xv[nt][1] = xp[4];
      }
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        const uint4 w0 = sp[rb * 96], w1 = sp[rb * 96 + 32], wb = sp[rb * 96 + 64];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t a0w = reinterpret_cast<const uint32_t*>(&w0)[j];
          const uint32_t a1w = reinterpret_cast<const uint32_t*>(&w1)[j];
          const uint32_t bw = reinterpret_cast<const uint32_t*>(&wb)[j];
          const uint32_t h0 = zip_lop(bw);
          const uint32_t h1 = zip_lop(__funnelshift_l(bw, bw, 4));
          const uint32_t r0 = zip_prmt(a0w, h0, 0x5140);
          const uint32_t r1 = zip_prmt(a0w, h0, 0x7362);
          const uint32_t r2 = zip_prmt(a1w, h1, 0x5140);
          const uint32_t r3 = zip_prmt(a1w, h1, 0x7362);
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[nt][j >> 1]);
            zip_mma(acc[rb][nt], r0, r1, r2, r3, xw[(j & 1) * 2], xw[(j & 1) * 2 + 1]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) zip_arrive(&empty[s]);
    }
  }
  __syncthreads();

  if (warp < WARPS) {
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

template <int NT, int RB, int CS, int WARPS, int S>
void zip_launch(const torch::Tensor& x, const torch::Tensor& wp, torch::Tensor& out, const torch::Tensor& rowptr,
                const torch::Tensor& esck, const torch::Tensor& escd, double scale, int64_t wrap_u4) {
  constexpr int BT = NT * 8, R = RB * 16, RP = R + 4;
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  const int UH = K / 64 / CS;
  TORCH_CHECK(m <= BT && N % R == 0 && (K / 64) % (CS * 4) == 0 && UH >= S && UH >= 2, "zip: shape not supported");
  const size_t xsz = (size_t)BT * (UH * 128 + 64);
  const size_t red = (size_t)WARPS * BT * RP * sizeof(float);
  const size_t body = xsz + (size_t)S * RB * 1536;
  const size_t smem = 512 + (size_t)(BT * R + (CS > 1 ? CS - 1 : 1) * BT * R) * sizeof(float) + (body > red ? body : red);
  TORCH_CHECK(smem <= 232448, "zip: smem ", smem);
  auto kern = zip_kernel<NT, RB, CS, WARPS, S>;
  set_attrs(kern, (int)smem);
  const dim3 grid((N / R) * CS, 1, 1);
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = cluster_cfg(grid, (WARPS + 2) * 32, (int)smem, CS, &attr);
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
  ZIP_CASE(1, 4, 2, 8, 16)
  
  ZIP_CASE(2, 4, 2, 8, 12)
  
  ZIP_CASE(2, 4, 2, 8, 8)
  ZIP_CASE(2, 4, 1, 8, 12)
  ZIP_CASE(1, 4, 4, 8, 16)
  ZIP_CASE(2, 4, 4, 8, 16)
  TORCH_CHECK(false, "zip: config not compiled nt=", nt, " rb=", rb, " cs=", cs, " w=", warps, " s=", stages);
}
"""

ZIP_CPP = ("void launch_zip(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor rowptr,"
           " torch::Tensor esck, torch::Tensor escd, double scale, int64_t wrap_u4, int64_t nt, int64_t rb,"
           " int64_t cs, int64_t warps, int64_t stages);")


def pick_eb(bits: torch.Tensor) -> int:
    eh = ((bits >> 8) & 0x7F).flatten()
    hist = torch.bincount(eh, minlength=128).to(torch.int64)
    return int(hist.unfold(0, 8, 1).sum(1).argmax())


def pack(w: torch.Tensor, rb: int = 4, max_esc_frac: float = 2e-3) -> dict:
    """-> dict(buf uint8 [RG,U,RB,3,32,16], scale, eb, rowptr, esck, escd, n_esc). Raises if unsuitable."""
    n, k = w.shape
    r = 16 * rb
    if w.dtype != torch.bfloat16 or n % r or k % 64:
        raise ValueError(f"cannot pack {tuple(w.shape)} {w.dtype} rb={rb}")
    rg, u = n // r, k // 64
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    eb = pick_eb(bits)
    s, e, low = bits >> 15, (bits >> 7) & 0xFF, bits & 0xFF
    c = (e >> 1) - eb
    ok = (c >= 0) & (c < 8)
    n_esc = int((~ok).sum())
    if n_esc > max_esc_frac * w.numel():
        raise ValueError(f"too many escapes: {n_esc}")
    s, c, low = (torch.where(ok, v, torch.zeros_like(v)) for v in (s, c, low))
    nib = (s << 3) | c

    def arr(v):  # -> [RG, U, RB, p, g, t, c, jl, h, eo]
        v = v.reshape(rg, rb, 2, 8, u, 2, 4, 2, 2, 2)
        return v.permute(0, 4, 1, 8, 3, 6, 5, 7, 2, 9)

    a = arr(low).reshape(rg, u, rb, 2, 32, 16).to(torch.uint8)
    nb = arr(nib).reshape(rg, u, rb, 2, 32, 4, 4)
    nw, nx = nb[:, :, :, 0], nb[:, :, :, 1]
    byte = ((nw & 8) << 4) | (nw & 7) | (nx & 8) | (torch.roll(nx & 7, -1, dims=-1) << 4)
    b = byte.reshape(rg, u, rb, 1, 32, 16).to(torch.uint8)
    buf = torch.cat([a, b], dim=3).contiguous()

    scale = 2.0 ** (2 * eb - REBASE)
    bad = (~ok).nonzero()
    rows, cols = bad[:, 0], bad[:, 1]
    order = torch.argsort(rows * k + cols)
    rows, cols = rows[order], cols[order]
    delta = (w[rows, cols].double() - 2.0 ** -15 * scale).float()
    rowptr = torch.zeros(n + 1, dtype=torch.int32, device=w.device)
    if n_esc:
        rowptr[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0).to(torch.int32)
    esck = torch.zeros(max(n_esc, 1), dtype=torch.int32, device=w.device)
    escd = torch.zeros(max(n_esc, 1), dtype=torch.float32, device=w.device)
    esck[:n_esc] = cols.to(torch.int32)
    escd[:n_esc] = delta
    return dict(buf=buf, scale=scale, eb=eb, rowptr=rowptr, esck=esck, escd=escd, n_esc=n_esc, rb=rb)


def unpack(p: dict, n: int, k: int) -> torch.Tensor:
    """Emulate the kernel's lop3/funnel/prmt decode on the packed words -> rebased fp32 weights [n, k]."""
    buf, rb = p["buf"], p["rb"]
    rg, u = buf.shape[0], buf.shape[1]
    by = buf.reshape(rg, u, rb, 3, 32, 4, 4).to(torch.int64)
    wd = by[..., 0] | (by[..., 1] << 8) | (by[..., 2] << 16) | (by[..., 3] << 24)  # [rg,u,rb,3,32,4]
    a0w, a1w, bw = wd[:, :, :, 0], wd[:, :, :, 1], wd[:, :, :, 2]
    m32 = 0xFFFFFFFF

    def lop(v):
        return ((v & 0x87878787) | 0x38383838) & m32

    h0 = lop(bw)
    h1 = lop(((bw << 4) | (bw >> 28)) & m32)

    def prmt(a, b, sel):
        src = torch.stack([(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)])
        out = torch.zeros_like(a)
        for i in range(4):
            out |= src[(sel >> (4 * i)) & 7] << (8 * i)
        return out

    r = torch.stack([prmt(a0w, h0, 0x5140), prmt(a0w, h0, 0x7362), prmt(a1w, h1, 0x5140),
                     prmt(a1w, h1, 0x7362)], dim=-1)  # [rg,u,rb,lane,j,a]
    pair = torch.stack([r & 0xFFFF, (r >> 16) & 0xFFFF], dim=-1)  # [..., a, eo]
    val = pair.to(torch.int32).to(torch.int16).contiguous().view(torch.bfloat16).float()
    val = val.reshape(rg, u, rb, 8, 4, 2, 2, 2, 2, 2)  # g,t,c,jl,p,h,eo   (a = 2p+h)
    val = val.permute(0, 2, 8, 3, 1, 5, 4, 6, 7, 9)  # -> rg,rb,h,g,u,c,t,jl,p,eo
    return val.reshape(n, k)


def reference_out(p: dict, w_dec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """fp32 [m, n] the kernel should produce (before bf16 rounding): decoded weights + escape fix."""
    out = (x.float() @ w_dec.t()) * p["scale"]
    rp, ek, ed = p["rowptr"].long(), p["esck"].long(), p["escd"]
    cnt = rp[1:] - rp[:-1]
    rows = torch.repeat_interleave(torch.arange(len(cnt), device=x.device), cnt)
    if rows.numel():
        out.index_put_((torch.arange(x.shape[0], device=x.device)[:, None].expand(-1, rows.numel()),
                        rows[None].expand(x.shape[0], -1)),
                       ed[: rows.numel()][None] * x.float()[:, ek[: rows.numel()]], accumulate=True)
    return out
