"""Lossless 12-bit BF16 weight format + fused-decode Triton GEMV/GEMM (phase 2.5 microbench).

Format for W [N, K] bf16, K cut into groups of G weights, each group into two halves (j indexes
the pair (half0[j], half1[j])):
  plane A  u8 [N, K]: bf16 low byte (E0 | m6..m0); the two halves interleave, so plane A read as
           u16 [N, K/2] holds (half0[j], half1[j]) in element j.
  plane B  u8 [N, K/2]: sign + 3-bit code c = (E >> 1) - eb of both weights of pair j.
Decoded high byte = s << 7 | (0x38 + c): every exponent is rebased by 112 - 2 eb, so the kernels
scale the fp32 accumulator by 2^(2 eb - 112) (exact). A weight with E >> 1 outside [eb, eb + 8)
is an escape: it decodes to 2^-15 and a per-(BN row block) list (n_local, k, W * 2^(112 - 2 eb))
adds x[k] * (W' - 2^-15) back. Layout "asm": B byte i of each 4-byte word holds s0_i at bit 7,
c0_i at bits 0..2, s1_i at bit 3 and c1_{i+1 mod 4} at bits 4..6, so one rotate gives the second
half's high bytes (7 int ops per 8 weights). Layout "plain": low nibble (s0, c0), high (s1, c1).

    python bench/compress_dev.py --selftest     # CPU: encoder round trip + PTX-op emulation
    python bench/compress_dev.py --compile      # CPU: compile for sm_90a, print op counts
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import triton
import triton.language as tl

GROUP = 128
ESC_DEC = 2.0 ** -15
REBASE = 112
LOP_MASK, LOP_BASE = 0x87878787, 0x38383838


class CTile(NamedTuple):
    kind: str  # "fma" (m = 1) or "dot"
    block_n: int
    block_k: int
    block_t: int
    split: int
    warps: int
    stages: int
    plain: bool = False


@dataclass
class Packed:
    a: torch.Tensor      # int16 [N, K/2]
    b: torch.Tensor      # uint8 [N, K/2]
    scale: float
    eb: int
    esc_n: torch.Tensor  # int64 [n_esc]
    esc_k: torch.Tensor
    esc_w: torch.Tensor  # bf16 [n_esc], W * 2^(112 - 2 eb)
    plain: bool
    group: int
    lists: dict = field(default_factory=dict)

    @property
    def n_esc(self) -> int:
        return int(self.esc_n.numel())


def _halves(t: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = t.shape
    v = t.view(n, k // group, 2, group // 2)
    return v[:, :, 0].reshape(n, k // 2), v[:, :, 1].reshape(n, k // 2)


def _unhalves(h0: torch.Tensor, h1: torch.Tensor, group: int) -> torch.Tensor:
    n, kh = h0.shape
    g = group // 2
    return torch.stack([h0.view(n, kh // g, g), h1.view(n, kh // g, g)], dim=2).reshape(n, 2 * kh)


def pick_eb(w: torch.Tensor) -> int:
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    eh = (bits >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).to(torch.int64)
    return int(hist.unfold(0, 8, 1).sum(1).argmax())


def encode(w: torch.Tensor, group: int = GROUP, plain: bool = False, eb: int | None = None) -> Packed:
    n, k = w.shape
    if w.dtype != torch.bfloat16 or k % group or group % 16:
        raise ValueError(f"cannot encode {tuple(w.shape)} {w.dtype} with group {group}")
    eb = pick_eb(w) if eb is None else eb
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, eh, low = bits >> 15, (bits >> 8) & 0x7F, bits & 0xFF
    c = eh - eb
    ok = (c >= 0) & (c < 8)
    s, c, low = (torch.where(ok, t, torch.zeros_like(t)) for t in (s, c, low))
    l0, l1 = _halves(low, group)
    s0, s1 = _halves(s, group)
    c0, c1 = _halves(c, group)
    a = torch.stack([l0, l1], dim=-1).to(torch.uint8).reshape(n, k).view(torch.int16)
    if plain:
        b = (s0 << 3) | c0 | (((s1 << 3) | c1) << 4)
    else:
        c1n = c1.view(n, k // 8, 4).roll(-1, dims=2).reshape(n, k // 2)
        b = (s0 << 7) | c0 | (s1 << 3) | (c1n << 4)
    esc = (~ok).nonzero()
    shift = 2.0 ** (REBASE - 2 * eb)
    esc_w = (w[esc[:, 0], esc[:, 1]].double() * shift).to(torch.bfloat16)
    if not torch.equal(esc_w.double() / shift, w[esc[:, 0], esc[:, 1]].double()):
        raise ValueError("escape weight not exact after rebase")
    return Packed(a.contiguous(), b.to(torch.uint8).contiguous(), 2.0 ** (2 * eb - REBASE), eb,
                  esc[:, 0].contiguous(), esc[:, 1].contiguous(), esc_w, plain, group)


def decode_ref(p: Packed) -> torch.Tensor:
    """Original bf16 bits (int32) from the planes, without the PTX tricks."""
    n, kh = p.a.shape
    ab = p.a.view(torch.uint8).view(n, kh, 2).to(torch.int32)
    l0, l1 = ab[..., 0], ab[..., 1]
    b = p.b.to(torch.int32)
    if p.plain:
        s0, c0, s1, c1 = (b >> 3) & 1, b & 7, (b >> 7) & 1, (b >> 4) & 7
    else:
        s0, c0, s1 = (b >> 7) & 1, b & 7, (b >> 3) & 1
        c1 = ((b >> 4) & 7).view(n, kh // 4, 4).roll(1, dims=2).reshape(n, kh)
    s, c, low = (_unhalves(x, y, p.group) for x, y in ((s0, s1), (c0, c1), (l0, l1)))
    e = ((c + p.eb) << 1) | (low >> 7)
    bits = (s << 15) | (e << 7) | (low & 0x7F)
    orig = (p.esc_w.double() / (2.0 ** (REBASE - 2 * p.eb))).to(torch.bfloat16)
    bits[p.esc_n, p.esc_k] = orig.view(torch.int16).to(torch.int32) & 0xFFFF
    return bits


def round_trip(w: torch.Tensor, p: Packed) -> None:
    want = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    if not torch.equal(decode_ref(p), want):
        raise AssertionError("compressed round trip is not bit-exact")


def _prmt(a: torch.Tensor, b: torch.Tensor, sel: int) -> torch.Tensor:
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = torch.zeros_like(a)
    for i in range(4):
        out |= src[(sel >> (4 * i)) & 7] << (8 * i)
    return out


def emulate_asm(p: Packed) -> torch.Tensor:
    """Decoded W' bits (int32, [N, K]) via the exact lop3/shf/prmt sequence of the asm kernel."""
    n, kh = p.a.shape
    m32 = 0xFFFFFFFF
    a32 = p.a.view(torch.uint8).view(n, kh // 2, 4).to(torch.int64)
    a32 = a32[..., 0] | (a32[..., 1] << 8) | (a32[..., 2] << 16) | (a32[..., 3] << 24)
    r0, r1 = a32[:, 0::2], a32[:, 1::2]
    bb = p.b.view(n, kh // 4, 4).to(torch.int64)
    b32 = bb[..., 0] | (bb[..., 1] << 8) | (bb[..., 2] << 16) | (bb[..., 3] << 24)
    h1 = (b32 & LOP_MASK) | LOP_BASE
    rot = ((b32 << 4) | (b32 >> 28)) & m32
    h2 = (rot & LOP_MASK) | LOP_BASE
    o0, o1 = _prmt(r0, h1, 0x5240), _prmt(r1, h1, 0x7260)
    p0, p1 = _prmt(r0, h2, 0x5341), _prmt(r1, h2, 0x7361)

    def unpack(lo, hi):
        w = torch.stack([lo & 0xFFFF, lo >> 16, hi & 0xFFFF, hi >> 16], dim=-1)
        return w.reshape(n, kh)

    return _unhalves(unpack(o0, o1), unpack(p0, p1), p.group).to(torch.int32)


def expected_rebased(w: torch.Tensor, p: Packed) -> torch.Tensor:
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    reb = (bits & 0x807F) | ((e - 2 * p.eb + REBASE) << 7)
    reb[p.esc_n, p.esc_k] = 0x3800
    return reb


def esc_lists(p: Packed, block_n: int, e_max: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per row block: packed (n_local << 16 | k) int32 and W' bf16, padded to e_max (pow2 >= 16)."""
    key = (block_n, e_max)
    if key in p.lists:
        return p.lists[key]
    n = p.a.shape[0]
    blocks = n // block_n
    blk = p.esc_n // block_n
    counts = torch.bincount(blk, minlength=blocks)
    need = max(16, triton.next_power_of_2(int(counts.max()) if p.n_esc else 1))
    e = need if e_max is None else e_max
    if need > e:
        raise ValueError(f"block of {block_n} rows has {int(counts.max())} escapes > {e}")
    order = torch.argsort(blk * n * 65536 + p.esc_n * 65536 + p.esc_k)
    blk_s, n_s, k_s, w_s = blk[order], p.esc_n[order], p.esc_k[order], p.esc_w[order]
    start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(p.n_esc, device=blk.device) - start[blk_s]
    nk = torch.full((blocks, e), block_n << 16, dtype=torch.int32, device=p.a.device)
    ew = torch.zeros((blocks, e), dtype=torch.bfloat16, device=p.a.device)
    nk[blk_s, rank] = (((n_s % block_n) << 16) | k_s).to(torch.int32)
    ew[blk_s, rank] = w_s
    p.lists[key] = (nk.contiguous(), ew.contiguous(), e)
    return p.lists[key]


def max_block_escapes(p: Packed, block_n: int) -> int:
    if not p.n_esc:
        return 0
    return int(torch.bincount(p.esc_n // block_n).max())


def nbytes(p: Packed, block_n: int) -> int:
    nk, ew, _ = esc_lists(p, block_n)
    return p.a.numel() * 2 + p.b.numel() + nk.numel() * 4 + ew.numel() * 2


# ------------------------------------------------------------------------------------- kernels

@triton.jit
def _decode(a, b, PLAIN: tl.constexpr):
    if PLAIN:
        ai = a.to(tl.int32) & 0xFFFF
        bi = b.to(tl.int32)
        h0 = ((bi & 0x8) << 4) | (bi & 0x7) | 0x38
        h1 = (bi & 0x80) | ((bi >> 4) & 0x7) | 0x38
        w0 = ((h0 << 8) | (ai & 0xFF)).to(tl.int16).to(tl.bfloat16, bitcast=True)
        w1 = ((h1 << 8) | (ai >> 8)).to(tl.int16).to(tl.bfloat16, bitcast=True)
    else:
        w0, w1 = tl.inline_asm_elementwise(
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
            dtype=(tl.bfloat16, tl.bfloat16),
            is_pure=True,
            pack=4,
        )
    return w0, w1


@triton.jit
def _decode_store(a_ptr, b_ptr, out_ptr, K: tl.constexpr, G: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, PLAIN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    HK: tl.constexpr = BLOCK_K // 2
    HG: tl.constexpr = G // 2
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    jj = tl.arange(0, HK)
    j0 = pid_k * HK
    a = tl.load(a_ptr + offs_n[:, None] * (K // 2) + j0 + jj[None, :])
    b = tl.load(b_ptr + offs_n[:, None] * (K // 2) + j0 + jj[None, :])
    w0, w1 = _decode(a, b, PLAIN)
    kx = pid_k * BLOCK_K + (jj // HG) * G + jj % HG
    o = out_ptr + offs_n[:, None] * K + kx[None, :]
    tl.store(o, w0)
    tl.store(o + HG, w1)


@triton.jit
def _split_finish(acc, ws_ptrs, ws_stride, out_ptrs, mask, cnt_ptr, pid_s, SPLIT: tl.constexpr):
    tl.store(ws_ptrs + pid_s * ws_stride, acc, mask=mask)
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
    if done == SPLIT - 1:
        total = tl.zeros_like(acc)
        for s in tl.static_range(SPLIT):
            total += tl.load(ws_ptrs + s * ws_stride, mask=mask, other=0.0, cache_modifier=".cg")
        tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=mask)
        tl.atomic_xchg(cnt_ptr, 0)


@triton.jit
def _cgemv(x_ptr, a_ptr, b_ptr, enk_ptr, ew_ptr, out_ptr, ws_ptr, cnt_ptr, scale, N,
           K: tl.constexpr, G: tl.constexpr, SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr, E_MAX: tl.constexpr, PLAIN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    HK: tl.constexpr = BLOCK_K // 2
    HG: tl.constexpr = G // 2
    KH: tl.constexpr = K // 2
    k_len: tl.constexpr = K // SPLIT
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    jj = tl.arange(0, HK)
    xoff = (jj // HG) * G + jj % HG
    rows = offs_n[:, None] * KH + jj[None, :]
    acc0 = tl.zeros((BLOCK_N, HK), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N, HK), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        k0 = pid_s * k_len + kk
        a = tl.load(a_ptr + rows + k0 // 2, eviction_policy="evict_first")
        b = tl.load(b_ptr + rows + k0 // 2, eviction_policy="evict_first")
        w0, w1 = _decode(a, b, PLAIN)
        x0 = tl.load(x_ptr + k0 + xoff, eviction_policy="evict_last").to(tl.float32)
        x1 = tl.load(x_ptr + k0 + HG + xoff, eviction_policy="evict_last").to(tl.float32)
        acc0 += w0.to(tl.float32) * x0[None, :]
        acc1 += w1.to(tl.float32) * x1[None, :]
    r = tl.sum(acc0 + acc1, axis=1)
    if E_MAX > 0:
        if pid_s == 0:
            e = tl.arange(0, E_MAX)
            nk = tl.load(enk_ptr + pid_n * E_MAX + e)
            ew = tl.load(ew_ptr + pid_n * E_MAX + e).to(tl.float32)
            xe = tl.load(x_ptr + (nk & 0xFFFF)).to(tl.float32)
            hit = tl.arange(0, BLOCK_N)[:, None] == (nk >> 16)[None, :]
            r += tl.sum(tl.where(hit, ((ew - 3.0517578125e-05) * xe)[None, :], 0.0), axis=1)
    r = r * scale
    if SPLIT == 1:
        tl.store(out_ptr + offs_n, r.to(out_ptr.dtype.element_ty))
    else:
        _split_finish(r, ws_ptr + offs_n, N, out_ptr + offs_n, offs_n < N, cnt_ptr + pid_n, pid_s, SPLIT)


@triton.jit(do_not_specialize=["M"])
def _cgemm(x_ptr, a_ptr, b_ptr, enk_ptr, ew_ptr, out_ptr, ws_ptr, cnt_ptr, scale, M, stride_xm, N,
           K: tl.constexpr, G: tl.constexpr, SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr, E_MAX: tl.constexpr, PLAIN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    HK: tl.constexpr = BLOCK_K // 2
    HG: tl.constexpr = G // 2
    KH: tl.constexpr = K // 2
    k_len: tl.constexpr = K // SPLIT
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    jj = tl.arange(0, HK)
    xoff = (jj // HG) * G + jj % HG
    rows = offs_n[:, None] * KH + jj[None, :]
    xrow = x_ptr + offs_t[:, None] * stride_xm + xoff[None, :]
    acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        k0 = pid_s * k_len + kk
        a = tl.load(a_ptr + rows + k0 // 2, eviction_policy="evict_first")
        b = tl.load(b_ptr + rows + k0 // 2, eviction_policy="evict_first")
        w0, w1 = _decode(a, b, PLAIN)
        x0 = tl.load(xrow + k0, mask=t_mask[:, None], other=0.0, eviction_policy="evict_last")
        x1 = tl.load(xrow + k0 + HG, mask=t_mask[:, None], other=0.0, eviction_policy="evict_last")
        acc = tl.dot(w0, tl.trans(x0), acc)
        acc = tl.dot(w1, tl.trans(x1), acc)
    if E_MAX > 0:
        if pid_s == 0:
            e = tl.arange(0, E_MAX)
            nk = tl.load(enk_ptr + pid_n * E_MAX + e)
            ew = tl.load(ew_ptr + pid_n * E_MAX + e)
            xe = tl.load(x_ptr + offs_t[None, :] * stride_xm + (nk & 0xFFFF)[:, None],
                         mask=t_mask[None, :], other=0.0)
            hit = tl.arange(0, BLOCK_N)[:, None] == (nk >> 16)[None, :]
            a_true = tl.where(hit, ew[None, :], 0.0).to(tl.bfloat16)
            a_dec = tl.where(hit, -3.0517578125e-05, 0.0).to(tl.bfloat16)
            acc = tl.dot(a_true, xe, acc)
            acc = tl.dot(a_dec, xe, acc)
    acc = acc * scale
    out_ptrs = out_ptr + offs_t[None, :] * N + offs_n[:, None]
    if SPLIT == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=t_mask[None, :])
    else:
        ws_ptrs = ws_ptr + offs_t[None, :] * N + offs_n[:, None]
        _split_finish(acc, ws_ptrs, BLOCK_T * N, out_ptrs, t_mask[None, :], cnt_ptr + pid_n, pid_s, SPLIT)


def launch(x: torch.Tensor, p: Packed, n: int, out: torch.Tensor, ws: torch.Tensor, cnt: torch.Tensor,
           t: CTile, e_max: int) -> None:
    k = p.a.shape[1] * 2
    nk, ew, e = esc_lists(p, t.block_n, e_max)
    grid = (n // t.block_n, t.split)
    common = dict(K=k, G=p.group, SPLIT=t.split, BLOCK_N=t.block_n, BLOCK_K=t.block_k, E_MAX=e,
                  PLAIN=p.plain, num_warps=t.warps, num_stages=t.stages)
    if t.kind == "fma":
        _cgemv[grid](x, p.a, p.b, nk, ew, out, ws, cnt, p.scale, n, **common)
    else:
        _cgemm[grid](x, p.a, p.b, nk, ew, out, ws, cnt, p.scale, x.shape[0], x.stride(0), n,
                     BLOCK_T=t.block_t, **common)


def tile_fits(t: CTile, n: int, k: int, group: int = GROUP) -> bool:
    return n % t.block_n == 0 and k % (t.split * t.block_k) == 0 and t.block_k % group == 0 \
        and (k // t.split) % group == 0


# ------------------------------------------------------------------------------ CPU self-tests

def selftest() -> None:
    gen = torch.Generator().manual_seed(5151)
    for n, k in ((256, 8192), (128, 4096)):
        w = (torch.randn((n, k), generator=gen) / k ** 0.5).to(torch.bfloat16)
        w[0, :5] = torch.tensor([0.0, -0.0, 1e-30, 3.0, -1e-9]).to(torch.bfloat16)
        for plain in (False, True):
            p = encode(w, plain=plain)
            round_trip(w, p)
            if not plain:
                got = emulate_asm(p)
                want = expected_rebased(w, p)
                if not torch.equal(got, want):
                    bad = (got != want).nonzero()[:5]
                    raise AssertionError(f"asm emulation mismatch at {bad.tolist()}")
            dec = torch.zeros_like(w)
            print(f"[{n}x{k}] plain={plain} eb={p.eb} window E in [{2 * p.eb}, {2 * p.eb + 15}] "
                  f"escapes {p.n_esc} ({p.n_esc / w.numel():.2e}) max/4rows {max_block_escapes(p, 4)} "
                  f"max/64rows {max_block_escapes(p, 64)} bytes/w {nbytes(p, 32) / w.numel():.4f}", dec.shape)
        nk, ew, e = esc_lists(encode(w), 32)
        assert int((ew != 0).sum()) <= encode(w).n_esc
    print("selftest ok")


def _sig_compile(fn, sig: dict, consts: dict, warps: int, stages: int):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    names = list(fn.arg_names)
    full = {name: sig.get(name, "constexpr") for name in names}
    attrs = {(names.index(nm),): [["tt.divisibility", 16]] for nm, ty in sig.items()
             if ty.startswith("*") or nm in ("N", "stride_xm")}
    src = ASTSource(fn=fn, signature=full, constexprs={(names.index(nm),): v for nm, v in consts.items()},
                    attrs=attrs)
    return triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": warps,
                                                                           "num_stages": stages})


def compile_report() -> None:
    import re

    ptrs = {"x_ptr": "*bf16", "a_ptr": "*i16", "b_ptr": "*u8", "enk_ptr": "*i32", "ew_ptr": "*bf16",
            "out_ptr": "*bf16", "ws_ptr": "*fp32", "cnt_ptr": "*i32", "scale": "fp32", "N": "i32"}
    cases = [
        ("cgemv asm 4x1024 st3", _cgemv, ptrs, dict(K=8192, G=128, SPLIT=1, BLOCK_N=4, BLOCK_K=1024,
                                                    E_MAX=16, PLAIN=False), 4, 3),
        ("cgemv plain 4x1024 st3", _cgemv, ptrs, dict(K=8192, G=128, SPLIT=1, BLOCK_N=4, BLOCK_K=1024,
                                                      E_MAX=16, PLAIN=True), 4, 3),
        ("cgemm asm 32x256 bt16 st5", _cgemm, {**ptrs, "M": "i32", "stride_xm": "i32"},
         dict(K=8192, G=128, SPLIT=1, BLOCK_N=32, BLOCK_K=256, BLOCK_T=16, E_MAX=64, PLAIN=False), 4, 5),
        ("cgemm asm 64x128 bt32 st5", _cgemm, {**ptrs, "M": "i32", "stride_xm": "i32"},
         dict(K=8192, G=128, SPLIT=1, BLOCK_N=64, BLOCK_K=128, BLOCK_T=32, E_MAX=128, PLAIN=False), 4, 5),
    ]
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    import submission as sub

    cases.append(("BASE bf16_gemv 4x1024 st3", sub._bf16_gemv,
                  {"x_ptr": "*bf16", "w_ptr": "*bf16", "out_ptr": "*bf16", "ws_ptr": "*fp32", "cnt_ptr": "*i32"},
                  dict(N=4096, K=8192, SPLIT=1, BLOCK_N=4, BLOCK_K=1024), 4, 3))
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), "")
    for name, fn, sig, consts, warps, stages in cases:
        if only and only not in name:
            continue
        try:
            k = _sig_compile(fn, sig, consts, warps, stages)
        except Exception as exc:
            print(f"== {name}: compile FAILED {type(exc).__name__}: {str(exc)[:2000]}")
            continue
        ttgir, ptx = k.asm["ttgir"], k.asm["ptx"]
        print(f"== {name}: regs {getattr(k, 'n_regs', '?')} spills {getattr(k, 'n_spills', '?')} "
              f"smem {k.metadata.shared}")
        print("   ttgir: convert_layout", ttgir.count("convert_layout"), "local_alloc", ttgir.count("local_alloc"),
              "async_copy", ttgir.count("async_copy_global_to_local"), "inline_asm", ttgir.count("elementwise_inline_asm"),
              "warp_group_dot", ttgir.count("warp_group_dot"), "tt.dot", ttgir.count("tt.dot "))
        if "--ir" in sys.argv:
            for line in ttgir.splitlines():
                if any(t in line for t in ("convert_layout", "#blocked", "#ttg.blocked", "inline_asm", "tt.load", "async_copy", "local_alloc")):
                    print("     |", line.strip()[:260])
        ops = re.findall(r"^\s*([a-z][a-z0-9_]*)(?:\.[\w.]+)?\s", ptx, flags=re.M)
        hist: dict[str, int] = {}
        for o in ops:
            hist[o] = hist.get(o, 0) + 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:22]
        print("   ptx ops:", " ".join(f"{o}:{c}" for o, c in top))
        try:
            from triton.tools.disasm import get_sass

            sass = get_sass(k.asm["cubin"])
            sops = re.findall(r"^[-\w:]+\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", sass, flags=re.M)
            sops = [o.split(".")[0] for o in sops]
            if not sops:
                print("   sass head:", sass[:600])
            sh: dict[str, int] = {}
            for o in sops:
                sh[o] = sh.get(o, 0) + 1
            print("   sass ops:", " ".join(f"{o}:{c}" for o, c in sorted(sh.items(), key=lambda kv: -kv[1])[:26]))
        except Exception as exc:
            print("   sass unavailable:", type(exc).__name__, str(exc)[:200])


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    if "--compile" in sys.argv:
        compile_report()
