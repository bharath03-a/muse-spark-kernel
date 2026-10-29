"""12-bit lossless BF16 weights decoded straight into MMA A-operand registers (Gluon, sm_90).

Swap-AB GEMV: out^T[BN, BT] = W[BN, K] @ x^T. W is the A operand *from registers*; x goes through TMA
into smem (wgmma B) or smem -> registers (mma_v2 B). Same 12-bit code as compress_dev (plane A = bf16
low byte, plane B = sign + 3-bit exponent code, rebase 2^(112 - 2 eb), escapes decode to 2^-15 and are
corrected by a per-row-block list through one small mma_v2), but stored in *register order*:

  K is cut into groups of 128, each into halves h0 = k[0:64), h1 = k[64:128). Logical tile per (row
  block, group) is [BN, 64] (column j = k within the half). The A-operand register layout of that tile
  (k_width 2) is the linear layout REG/LANE/WARP below. Offline, each tile is written to global memory
  in the order (reg[:v], lane, reg[v:], warp), so a TMA box copies it verbatim into smem, the smem is
  reinterpreted with the matching SharedLinearLayout, and `load(DLL)` is one conflict-free 16-byte
  ld.shared per thread per vector. DLL -> DotOperandLayout is asserted trivial at compile time.
  Plane A: i16 (h0 low byte | h1 low byte << 8), v = 3. Plane B: u8, v = 4; 4 consecutive register
  elements share one 32-bit word with compress_dev's rotate trick (7 int ops per 8 weights).
  Dense twin (same layout/MMA, bf16 bits in the same register order) measures the layout/MMA cost.

    python bench/zip_gluon_dev.py --selftest   # CPU torch only: encoder + decode emulation, bit-exact
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.ampere import mma_v2
    from triton.experimental.gluon.language.nvidia.hopper import (mbarrier, tma, warpgroup_mma,
                                                                  warpgroup_mma_wait)
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GDesc
    HAVE_GLUON = True
except ImportError:  # CPU-only selftest
    HAVE_GLUON = False

GROUP = 128
HALF = 64
REBASE = 112
ESC_DEC = 2.0 ** -15
LOP_MASK, LOP_BASE = 0x87878787, 0x38383838
# A-operand (k_width 2) register layout of a [16 * warps, 64] tile, coordinates (n, j).
REG = ((0, 1), (8, 0), (0, 8), (0, 16), (0, 32))
LANE = ((0, 2), (0, 4), (1, 0), (2, 0), (4, 0))
VBITS_A, VBITS_B = 3, 4  # 16-byte vectors: 8 x i16, 16 x u8


def warp_bases(warps: int) -> tuple:
    return tuple((16 << i, 0) for i in range(warps.bit_length() - 1))


def storage_bases(vbits: int, warps: int) -> tuple:
    return REG[:vbits] + LANE + REG[vbits:] + warp_bases(warps)


def canon_bases(warps: int) -> tuple:
    return REG + LANE + warp_bases(warps)


def coords(bases: tuple, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    o = torch.arange(1 << len(bases), device=device)
    n = torch.zeros_like(o)
    j = torch.zeros_like(o)
    for i, (bn, bj) in enumerate(bases):
        bit = (o >> i) & 1
        n += bit * bn
        j += bit * bj
    return n, j


def _i16(v: torch.Tensor) -> torch.Tensor:
    return (((v.to(torch.int32) + 0x8000) & 0xFFFF) - 0x8000).to(torch.int16)


def pick_eb(w: torch.Tensor) -> int:
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hist = torch.bincount(((bits >> 8) & 0x7F).flatten(), minlength=128).to(torch.int64)
    return int(hist.unfold(0, 8, 1).sum(1).argmax())


@dataclass(frozen=True)
class ZPack:
    warps: int
    a: torch.Tensor      # compressed: i16 [NB*KB*BN/4, 256]; dense: bf16 [NB*KB*2*BN/4, 256]
    b: torch.Tensor | None  # u8 [NB*KB*BN/4, 256] (compressed only)
    enk: torch.Tensor    # i32 [NB, E_MAX]: n_local << 16 | k  (padding: BN << 16)
    ew: torch.Tensor     # bf16 [NB, E_MAX]: W * 2^(112 - 2 eb)
    e_max: int
    scale: float
    eb: int
    n: int
    k: int
    n_esc: int

    @property
    def comp(self) -> bool:
        return self.b is not None

    @property
    def block_n(self) -> int:
        return 16 * self.warps

    def nbytes(self) -> int:
        extra = self.b.numel() + self.enk.numel() * 4 + self.ew.numel() * 2 if self.comp else 0
        return self.a.numel() * self.a.element_size() + extra


def fold(w: torch.Tensor, rows: int, q: int) -> torch.Tensor:
    """[N, K] -> [N * q, K / q]: per block of `rows` real rows, q K-slices stacked as virtual rows."""
    n, k = w.shape
    return w.view(n // rows, rows, q, k // q).permute(0, 2, 1, 3).reshape(n * q, k // q).contiguous()


def _gather(t: torch.Tensor, bn: int, n_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
    """[N, K] -> [NB, KB, 2, T] in storage order (T = BN * 64)."""
    n, k = t.shape
    v = t.view(n // bn, bn, k // GROUP, 2, HALF)
    return v[:, n_idx, :, :, j_idx].permute(1, 2, 3, 0)


def _esc_lists(esc_n, esc_k, esc_w, nb: int, bn: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    blk = esc_n // bn
    counts = torch.bincount(blk, minlength=nb)
    most = int(counts.max()) if esc_n.numel() else 0
    e = 16
    while e < most:
        e *= 2
    nk = torch.full((nb, e), bn << 16, dtype=torch.int32, device=device)
    ew = torch.zeros((nb, e), dtype=torch.bfloat16, device=device)
    if esc_n.numel():
        order = torch.argsort(blk * (1 << 40) + esc_n * (1 << 20) + esc_k)
        blk_s, n_s, k_s, w_s = blk[order], esc_n[order], esc_k[order], esc_w[order]
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(esc_n.numel(), device=device) - start[blk_s]
        nk[blk_s, rank] = (((n_s % bn) << 16) | k_s).to(torch.int32)
        ew[blk_s, rank] = w_s
    return nk.contiguous(), ew.contiguous(), e


def encode(w: torch.Tensor, warps: int, dense: bool = False) -> ZPack:
    n, k = w.shape
    bn = 16 * warps
    if w.dtype != torch.bfloat16 or n % bn or k % GROUP:
        raise ValueError(f"cannot encode {tuple(w.shape)} {w.dtype} with BN {bn}")
    nb, kb, t = n // bn, k // GROUP, bn * HALF
    dev = w.device
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    na, ja = coords(storage_bases(VBITS_A, warps), dev)
    z = torch.zeros(nb, 0, dtype=torch.int32, device=dev)
    if dense:
        d = _i16(_gather(bits, bn, na, ja)).reshape(-1, 256).view(torch.bfloat16).contiguous()
        return ZPack(warps, d, None, z, z.to(torch.bfloat16), 0, 1.0, 0, n, k, 0)
    eb = pick_eb(w)
    s, eh, low = bits >> 15, (bits >> 8) & 0x7F, bits & 0xFF
    c = eh - eb
    ok = (c >= 0) & (c < 8)
    s, c, low = (torch.where(ok, x, torch.zeros_like(x)) for x in (s, c, low))
    la = _gather(low, bn, na, ja)
    a = _i16(la[:, :, 0] | (la[:, :, 1] << 8)).reshape(-1, 256).contiguous()
    nbi, jbi = coords(storage_bases(VBITS_B, warps), dev)
    sb, cb = _gather(s, bn, nbi, jbi), _gather(c, bn, nbi, jbi)
    c1n = cb[:, :, 1].reshape(nb, kb, t // 4, 4).roll(-1, dims=3).reshape(nb, kb, t)
    b = ((sb[:, :, 0] << 7) | cb[:, :, 0] | (sb[:, :, 1] << 3) | (c1n << 4)).to(torch.uint8)
    esc = (~ok).nonzero()
    shift = 2.0 ** (REBASE - 2 * eb)
    orig = w[esc[:, 0], esc[:, 1]]
    esc_w = (orig.double() * shift).to(torch.bfloat16)
    if not torch.equal(esc_w.double() / shift, orig.double()):
        raise ValueError("escape weight not exact after rebase")
    enk, ew, e = _esc_lists(esc[:, 0].contiguous(), esc[:, 1].contiguous(), esc_w, nb, bn, dev)
    return ZPack(warps, a, b.reshape(-1, 256).contiguous(), enk, ew, e, 2.0 ** (2 * eb - REBASE), eb, n, k,
                 int(esc.shape[0]))


# --------------------------------------------------------------------------- torch reference decode

def _prmt(a: torch.Tensor, b: torch.Tensor, sel: int) -> torch.Tensor:
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = torch.zeros_like(a)
    for i in range(4):
        out |= src[(sel >> (4 * i)) & 7] << (8 * i)
    return out


def _to_canon(storage: torch.Tensor, vbits: int, warps: int) -> torch.Tensor:
    """[..., T] storage order -> [..., T] canonical register order."""
    bn = 16 * warps
    ns, js = coords(storage_bases(vbits, warps), storage.device)
    nc, jc = coords(canon_bases(warps), storage.device)
    pos = torch.empty(bn * HALF, dtype=torch.int64, device=storage.device)
    pos[ns * HALF + js] = torch.arange(bn * HALF, device=storage.device)
    return storage[..., pos[nc * HALF + jc]]


def emulate(p: ZPack) -> torch.Tensor:
    """Decoded W' bits [N, K] (int32) through the exact asm op sequence, per register word."""
    bn, warps = p.block_n, p.warps
    nb, kb, t = p.n // bn, p.k // GROUP, bn * HALF
    a = _to_canon(p.a.view(nb, kb, t).to(torch.int64) & 0xFFFF, VBITS_A, warps).view(nb, kb, t // 4, 4)
    b = _to_canon(p.b.view(nb, kb, t).to(torch.int64), VBITS_B, warps).view(nb, kb, t // 4, 4)
    r0 = a[..., 0] | (a[..., 1] << 16)
    r1 = a[..., 2] | (a[..., 3] << 16)
    b32 = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)
    h1 = (b32 & LOP_MASK) | LOP_BASE
    rot = ((b32 << 4) | (b32 >> 28)) & 0xFFFFFFFF
    h2 = (rot & LOP_MASK) | LOP_BASE
    o0, o1 = _prmt(r0, h1, 0x5240), _prmt(r1, h1, 0x7260)
    q0, q1 = _prmt(r0, h2, 0x5341), _prmt(r1, h2, 0x7361)
    w0 = torch.stack([o0 & 0xFFFF, o0 >> 16, o1 & 0xFFFF, o1 >> 16], -1).view(nb, kb, t)
    w1 = torch.stack([q0 & 0xFFFF, q0 >> 16, q1 & 0xFFFF, q1 >> 16], -1).view(nb, kb, t)
    nc, jc = coords(canon_bases(warps), p.a.device)
    out = torch.empty(nb, bn, kb, 2, HALF, dtype=torch.int64, device=p.a.device)
    out[:, nc, :, 0, jc] = w0.permute(2, 0, 1)
    out[:, nc, :, 1, jc] = w1.permute(2, 0, 1)
    return out.view(p.n, p.k).to(torch.int32)


def expected_rebased(w: torch.Tensor, p: ZPack) -> tuple[torch.Tensor, torch.Tensor]:
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    reb = (bits & 0x807F) | ((e - 2 * p.eb + REBASE) << 7)
    c = ((bits >> 8) & 0x7F) - p.eb
    esc = (c < 0) | (c >= 8)
    return torch.where(esc, torch.full_like(reb, 0x3800), reb), esc


def round_trip(w: torch.Tensor, p: ZPack) -> None:
    """Bit-exact: emulated decode + escape list reproduces W (after undoing the exact rebase)."""
    got = emulate(p)
    want, esc = expected_rebased(w, p)
    if not torch.equal(got, want):
        bad = (got != want).nonzero()[:5].tolist()
        raise AssertionError(f"register-order decode mismatch at {bad}")
    nk = p.enk.to(torch.int64)
    real = (nk >> 16) < p.block_n
    rows = (torch.arange(p.enk.shape[0], device=w.device)[:, None] * p.block_n + (nk >> 16))[real]
    cols = (nk & 0xFFFF)[real]
    if int(real.sum()) != int(esc.sum()) or not bool(esc[rows, cols].all()):
        raise AssertionError("escape list does not cover the escaped weights")
    back = (p.ew[real].double() / 2.0 ** (REBASE - 2 * p.eb)).to(torch.bfloat16)
    if not torch.equal(back, w[rows, cols]):
        raise AssertionError("escape values not exact")


def selftest() -> None:
    gen = torch.Generator().manual_seed(5151)
    for n, k, warps in ((256, 1024, 4), (128, 2048, 2), (64, 512, 1)):
        w = (torch.randn((n, k), generator=gen) / k ** 0.5).to(torch.bfloat16)
        w[0, :6] = torch.tensor([0.0, -0.0, 1e-30, 3.0, -1e-9, 1e-3]).to(torch.bfloat16)
        p = encode(w, warps)
        round_trip(w, p)
        d = encode(w, warps, dense=True)
        dc = _to_canon(d.a.view(torch.int16).view(n // p.block_n, k // GROUP, 2, -1).to(torch.int64) & 0xFFFF,
                       VBITS_A, warps)
        nc, jc = coords(canon_bases(warps))
        rec = torch.empty(n // p.block_n, p.block_n, k // GROUP, 2, HALF, dtype=torch.int64)
        rec[:, nc, :, :, jc] = dc.permute(3, 0, 1, 2)
        assert torch.equal(rec.view(n, k), (w.view(torch.int16).to(torch.int64) & 0xFFFF)), "dense order"
        print(f"[{n}x{k} warps {warps}] eb={p.eb} escapes {p.n_esc} E_MAX {p.e_max} "
              f"bytes/w {p.nbytes() / w.numel():.4f} (dense {d.nbytes() / w.numel():.1f}) ok")
    print("selftest ok")


# ------------------------------------------------------------------------------------- kernels

if HAVE_GLUON:

    def layouts(warps: int):
        bn = 16 * warps
        dll = gl.DistributedLinearLayout([list(b) for b in REG], [list(b) for b in LANE],
                                         [list(b) for b in warp_bases(warps)], [], [bn, HALF])
        sha = gl.SharedLinearLayout([list(b) for b in storage_bases(VBITS_A, warps)])
        shb = gl.SharedLinearLayout([list(b) for b in storage_bases(VBITS_B, warps)])
        return dll, sha, shb

    PLANE16 = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=16, rank=2)
    PLANE8 = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=2)

    @gluon.jit
    def _decode(a, b):
        return gl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 rm, rc, rh1, rh2, rt;
            mov.b32 rm, 0x87878787;
            mov.b32 rc, 0x38383838;
            lop3.b32 rh1, $6, rm, rc, 0xEA;
            shf.l.wrap.b32 rt, $6, $6, 4;
            lop3.b32 rh2, rt, rm, rc, 0xEA;
            prmt.b32 $0, $4, rh1, 0x5240;
            prmt.b32 $1, $5, rh1, 0x7260;
            prmt.b32 $2, $4, rh2, 0x5341;
            prmt.b32 $3, $5, rh2, 0x7361;
            }
            """,
            constraints="=r,=r,=r,=r,r,r,r",
            args=[a, b],
            dtype=(gl.bfloat16, gl.bfloat16),
            is_pure=True,
            pack=4,
        )

    @gluon.jit
    def _raw(a_bufs, b_bufs, q, COMP: gl.constexpr, BN: gl.constexpr, DLL: gl.constexpr,
             SHA: gl.constexpr, SHB: gl.constexpr):
        """Group q of the ring: (plane A i16, plane B u8) or (dense h0, dense h1) in register order."""
        if COMP:
            a = a_bufs.index(q)._reinterpret(gl.int16, [BN, 64], SHA).load(DLL)
            b = b_bufs.index(q)._reinterpret(gl.uint8, [BN, 64], SHB).load(DLL)
        else:
            a = a_bufs.index(2 * q)._reinterpret(gl.bfloat16, [BN, 64], SHA).load(DLL)
            b = a_bufs.index(2 * q + 1)._reinterpret(gl.bfloat16, [BN, 64], SHA).load(DLL)
        return a, b

    @gluon.jit
    def _finish(a, b, COMP: gl.constexpr, ADOT: gl.constexpr):
        if COMP:
            w0, w1 = _decode(a, b)
        else:
            w0 = a
            w1 = b
        return gl.convert_layout(w0, ADOT, assert_trivial=True), gl.convert_layout(w1, ADOT, assert_trivial=True)

    @gluon.jit
    def _issue(a_desc, b_desc, x_desc, a_bufs, b_bufs, x_bufs, bar, s, i, row_a, row_b, pred,
               NA: gl.constexpr, KG: gl.constexpr, COMP: gl.constexpr, Q: gl.constexpr = 1, KV: gl.constexpr = 0):
        RA: gl.constexpr = a_desc.block_type.shape[0]
        BT: gl.constexpr = x_desc.block_type.shape[0]
        NBYTES: gl.constexpr = NA * a_desc.block_type.nbytes + 2 * KG * Q * x_desc.block_type.nbytes \
            + (KG * b_desc.block_type.nbytes if COMP else 0)
        mbarrier.expect(bar, NBYTES, pred=pred)
        for q in gl.static_range(NA):
            tma.async_load(a_desc, [row_a + (i * NA + q) * RA, 0], bar, a_bufs.index(s * NA + q), pred=pred)
        if COMP:
            RB: gl.constexpr = b_desc.block_type.shape[0]
            for q in gl.static_range(KG):
                tma.async_load(b_desc, [row_b + (i * KG + q) * RB, 0], bar, b_bufs.index(s * KG + q), pred=pred)
        for q in gl.static_range(2 * KG):
            if Q == 1:
                tma.async_load(x_desc, [0, (i * KG) * 128 + q * 64], bar, x_bufs.index(s * 2 * KG + q), pred=pred)
            else:
                for qq in gl.static_range(Q):
                    tma.async_load(x_desc, [0, qq * KV + (i * KG) * 128 + q * 64], bar,
                                   x_bufs.index(s * 2 * KG + q).slice(qq * BT, BT, dim=0), pred=pred)

    @gluon.jit
    def _escape_fix(acc, enk_ptr, ew_ptr, x_ptr, stride_xm, M, nb, BN: gl.constexpr, BT: gl.constexpr,
                    E_MAX: gl.constexpr, num_warps: gl.constexpr, Q: gl.constexpr = 1, KV: gl.constexpr = 0):
        # Q > 1: acc columns are Q blocks of BT tokens; block q2 multiplies x[:, q2 * KV + k].
        m2: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[num_warps, 1],
                                                     instr_shape=[16, 8])
        a2: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=m2, k_width=2)
        b2: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=m2, k_width=2)
        EC: gl.constexpr = 32 if E_MAX > 32 else E_MAX
        rows = gl.arange(0, BN, layout=gl.SliceLayout(1, a2))
        c_b = gl.arange(0, Q * BT, layout=gl.SliceLayout(0, b2))
        t_b = c_b % BT
        col = (t_b * stride_xm + (c_b // BT) * KV)[None, :]
        c = gl.zeros([BN, Q * BT], gl.float32, layout=m2)
        for e0 in gl.static_range(0, E_MAX, EC):
            e_a = e0 + gl.arange(0, EC, layout=gl.SliceLayout(0, a2))
            nk_a = gl.load(enk_ptr + nb * E_MAX + e_a)
            ew_a = gl.load(ew_ptr + nb * E_MAX + e_a)
            hit = rows[:, None] == (nk_a >> 16)[None, :]
            zero = gl.zeros([BN, EC], gl.bfloat16, layout=a2)
            a_true = gl.where(hit, ew_a[None, :], zero)
            a_dec = gl.where(hit, gl.full([BN, EC], -3.0517578125e-05, gl.bfloat16, layout=a2), zero)
            e_b = e0 + gl.arange(0, EC, layout=gl.SliceLayout(1, b2))
            nk_b = gl.load(enk_ptr + nb * E_MAX + e_b)
            xe = gl.load(x_ptr + col + (nk_b & 0xFFFF)[:, None], mask=(t_b < M)[None, :], other=0.0)
            c = mma_v2(a_true, xe, c)
            c = mma_v2(a_dec, xe, c)
        return acc + gl.convert_layout(c, acc.type.layout)

    @gluon.jit
    def zgemv(a_desc, b_desc, x_desc, enk_ptr, ew_ptr, x_ptr, out_ptr, M, stride_xm, scale,
              N: gl.constexpr, K: gl.constexpr, KG: gl.constexpr, STAGES: gl.constexpr, COMP: gl.constexpr,
              MMA: gl.constexpr, E_MAX: gl.constexpr, DLL: gl.constexpr, SHA: gl.constexpr,
              SHB: gl.constexpr, Q: gl.constexpr, num_warps: gl.constexpr):
        # Q > 1 (K fold): the BN virtual rows are Q K-slices (length K) of R = BN / Q real rows, and the
        # wgmma N axis carries Q token blocks (x of slice q); only diagonal blocks are kept.
        BN: gl.constexpr = 16 * num_warps
        R: gl.constexpr = BN // Q
        BT: gl.constexpr = x_desc.block_type.shape[0]
        RA: gl.constexpr = a_desc.block_type.shape[0]
        RB: gl.constexpr = b_desc.block_type.shape[0]
        KB: gl.constexpr = K // 128
        KI: gl.constexpr = KB // KG
        NA: gl.constexpr = KG if COMP else 2 * KG
        if MMA == 3:
            mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                          instr_shape=[16, Q * BT, 16])
        else:
            gl.static_assert(Q == 1, "K fold needs wgmma")
            mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[num_warps, 1],
                                                          instr_shape=[16, 8])
        adot: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
        bdot: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=2)
        nb = gl.program_id(0)
        row_a = nb * (KB * (NA // KG) * RA)
        row_b = nb * (KB * RB)
        a_bufs = gl.allocate_shared_memory(a_desc.dtype, [STAGES * NA] + a_desc.block_type.shape, a_desc.layout)
        if COMP:
            b_bufs = gl.allocate_shared_memory(b_desc.dtype, [STAGES * KG] + b_desc.block_type.shape,
                                               b_desc.layout)
        else:
            b_bufs = a_bufs
        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES * 2 * KG, Q * BT, 64], x_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for p0 in gl.static_range(STAGES):
            mbarrier.init(ready.index(p0), count=1)
        acc = gl.zeros([BN, Q * BT], gl.float32, layout=mma)
        if MMA == 3:
            for p0 in gl.static_range(STAGES - 1):
                _issue(a_desc, b_desc, x_desc, a_bufs, b_bufs, x_bufs, ready.index(p0), p0, p0, row_a, row_b,
                       p0 < KI, NA, KG, COMP, Q, K)
            for i in range(KI):
                s = i % STAGES
                mbarrier.wait(ready.index(s), (i // STAGES) & 1)
                for g in gl.static_range(KG):
                    a, b = _raw(a_bufs, b_bufs, s * KG + g, COMP, BN, DLL, SHA, SHB)
                    w0, w1 = _finish(a, b, COMP, adot)
                    acc = warpgroup_mma(w0, x_bufs.index(s * 2 * KG + 2 * g).permute((1, 0)), acc, is_async=True)
                    acc = warpgroup_mma(w1, x_bufs.index(s * 2 * KG + 2 * g + 1).permute((1, 0)), acc,
                                        is_async=True)
                acc = warpgroup_mma_wait(2 * KG, deps=(acc,))
                # every warp issued group i-1's wgmmas after its ld.shared of slot i-1, and they retired.
                j = i + STAGES - 1
                _issue(a_desc, b_desc, x_desc, a_bufs, b_bufs, x_bufs, ready.index(j % STAGES), j % STAGES, j,
                       row_a, row_b, j < KI, NA, KG, COMP, Q, K)
            acc = warpgroup_mma_wait(0, deps=(acc,))
        else:
            gl.static_assert(KG == 1, "mma_v2 path holds one group in registers")
            for p0 in gl.static_range(STAGES):
                _issue(a_desc, b_desc, x_desc, a_bufs, b_bufs, x_bufs, ready.index(p0), p0, p0, row_a, row_b,
                       p0 < KI, NA, KG, COMP)
            for i in range(KI):
                s = i % STAGES
                mbarrier.wait(ready.index(s), (i // STAGES) & 1)
                a, b = _raw(a_bufs, b_bufs, s, COMP, BN, DLL, SHA, SHB)
                x0 = x_bufs.index(2 * s).permute((1, 0)).load(bdot)
                x1 = x_bufs.index(2 * s + 1).permute((1, 0)).load(bdot)
                gl.barrier()
                j = i + STAGES
                _issue(a_desc, b_desc, x_desc, a_bufs, b_bufs, x_bufs, ready.index(s), s, j, row_a, row_b,
                       j < KI, NA, KG, COMP)
                w0, w1 = _finish(a, b, COMP, adot)
                acc = mma_v2(w0, x0, acc)
                acc = mma_v2(w1, x1, acc)
        if E_MAX > 0:
            acc = _escape_fix(acc, enk_ptr, ew_ptr, x_ptr, stride_xm, M, nb, BN, BT, E_MAX, num_warps, Q, K)
        acc = acc * scale
        if Q == 1:
            offs_n = nb * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
            offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
            gl.store(out_ptr + offs_t[None, :] * N + offs_n[:, None], acc.to(gl.bfloat16),
                     mask=(offs_t < M)[None, :])
        else:
            TW: gl.constexpr = 32 if BT >= 32 else BT
            epi: gl.constexpr = gl.BlockedLayout([1, BT // TW], [32 // TW, TW], [num_warps, 1], [1, 0])
            red = gl.allocate_shared_memory(gl.float32, [BN, Q * BT],
                                            gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))
            red.store(acc)
            gl.barrier()
            tot = gl.zeros([R, BT], gl.float32, layout=epi)
            for qq in gl.static_range(Q):
                tot += red.slice(qq * R, R, dim=0).slice(qq * BT, BT, dim=1).load(epi)
            offs_n = nb * R + gl.arange(0, R, layout=gl.SliceLayout(1, epi))
            offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, epi))
            gl.store(out_ptr + offs_t[None, :] * N + offs_n[:, None], tot.to(gl.bfloat16),
                     mask=(offs_t < M)[None, :])

    @gluon.jit
    def zdecode(a_desc, b_desc, out_ptr, K: gl.constexpr, COMP: gl.constexpr, DLL: gl.constexpr,
                SHA: gl.constexpr, SHB: gl.constexpr, num_warps: gl.constexpr):
        """Decoded register tile (rebased W' bits; dense: W) of group (nb, kb) -> out[N, K]."""
        BN: gl.constexpr = 16 * num_warps
        RA: gl.constexpr = a_desc.block_type.shape[0]
        RB: gl.constexpr = b_desc.block_type.shape[0]
        NA: gl.constexpr = 1 if COMP else 2
        KB: gl.constexpr = K // 128
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[num_warps, 1],
                                                      instr_shape=[16, 8])
        adot: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
        nb = gl.program_id(0)
        kb = gl.program_id(1)
        a_bufs = gl.allocate_shared_memory(a_desc.dtype, [NA] + a_desc.block_type.shape, a_desc.layout)
        if COMP:
            b_bufs = gl.allocate_shared_memory(b_desc.dtype, [1] + b_desc.block_type.shape, b_desc.layout)
        else:
            b_bufs = a_bufs
        bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=1)
        nbytes: gl.constexpr = NA * a_desc.block_type.nbytes + (b_desc.block_type.nbytes if COMP else 0)
        mbarrier.expect(bar.index(0), nbytes)
        for q in gl.static_range(NA):
            tma.async_load(a_desc, [((nb * KB + kb) * NA + q) * RA, 0], bar.index(0), a_bufs.index(q))
        if COMP:
            tma.async_load(b_desc, [(nb * KB + kb) * RB, 0], bar.index(0), b_bufs.index(0))
        mbarrier.wait(bar.index(0), 0)
        a, b = _raw(a_bufs, b_bufs, 0, COMP, BN, DLL, SHA, SHB)
        w0, w1 = _finish(a, b, COMP, adot)
        offs_n = nb * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, adot))
        offs_j = kb * 128 + gl.arange(0, 64, layout=gl.SliceLayout(0, adot))
        ptrs = out_ptr + offs_n[:, None] * K + offs_j[None, :]
        gl.store(ptrs, w0)
        gl.store(ptrs + 64, w1)

    def plane_descs(p: ZPack):
        rows = p.block_n // 4
        lay = PLANE16
        a_desc = GDesc.from_tensor(p.a, [rows, 256], lay)
        b_desc = GDesc.from_tensor(p.b, [rows, 256], PLANE8) if p.comp else a_desc
        return a_desc, b_desc

    def x_desc_for(x: torch.Tensor, bt: int):
        return GDesc.from_tensor(x, [bt, 64], gl.NVMMASharedLayout.get_default_for([bt, 64], gl.bfloat16))

    class Runner:
        """Pre-built descriptors/layouts for one packed weight; call(x, out) launches the GEMV.

        q > 1: p packs the K-folded matrix fold(W, R, q) (virtual N*q x K/q)."""

        def __init__(self, p: ZPack, mma: int, kg: int, stages: int, bt: int, q: int = 1):
            self.p, self.mma, self.kg, self.stages, self.bt, self.q = p, mma, kg, stages, bt, q
            self.a_desc, self.b_desc = plane_descs(p)
            self.dll, self.sha, self.shb = layouts(p.warps)
            self.scale = float(p.scale)
            self.e_max = p.e_max if p.comp else 0

        def __call__(self, x: torch.Tensor, out: torch.Tensor) -> None:
            p = self.p
            zgemv[(p.n // p.block_n,)](
                self.a_desc, self.b_desc, x_desc_for(x, self.bt), p.enk, p.ew, x, out, x.shape[0], x.stride(0),
                self.scale, N=p.n // self.q, K=p.k, KG=self.kg, STAGES=self.stages, COMP=p.comp, MMA=self.mma,
                E_MAX=self.e_max, DLL=self.dll, SHA=self.sha, SHB=self.shb, Q=self.q, num_warps=p.warps)

    def decode_full(p: ZPack) -> torch.Tensor:
        a_desc, b_desc = plane_descs(p)
        dll, sha, shb = layouts(p.warps)
        out = torch.empty(p.n, p.k, device=p.a.device, dtype=torch.bfloat16)
        zdecode[(p.n // p.block_n, p.k // GROUP)](a_desc, b_desc, out, K=p.k, COMP=p.comp, DLL=dll, SHA=sha,
                                                  SHB=shb, num_warps=p.warps)
        return out


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
