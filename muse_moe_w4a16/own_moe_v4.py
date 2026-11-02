# okc: problem=muse_moe_w4a16
import time
from typing import TypedDict

import torch
import triton
import triton.language as tl


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
# Gate/up interleave granularity in prepare; every BN13 below must divide it.
GU_BLOCK = 64
# (BN, BK, num_warps, num_stages); BN is gate rows per CTA for W13 (2*BN rows loaded).
# BK is fixed at 128: the in-chunk byte repack is laid out for 64-byte chunks.
TILES = [(64, 128, 4, 4), (64, 128, 8, 4), (32, 128, 4, 4)]
SPLITS = {1: (8, 16), 8: (4, 8), 16: (2, 4), 32: (1, 2), 128: (1,), 256: (1,)}
BMS = {1: (16,), 8: (16,), 16: (16,), 32: (16,), 128: (16, 32), 256: (16, 32)}
DEFAULT = {1: (16, 0, 8, False, False), 8: (16, 0, 4, True, False), 16: (16, 0, 2, True, False),
           32: (16, 0, 1, True, False), 128: (16, 0, 1, True, False), 256: (32, 0, 1, True, False)}
SWEEP_BUDGET_S = 150.0
# bs >= 128: experts with at least HOT_MIN tokens get HOT_BM-token tiles in a second launch.
HOT_BM = 64
HOT_MIN = 48
_DEQ_ASM = tl.constexpr("""
{
.reg .b32 L0, L1, H0, H1, Z, t, x, s0, s1, s2, s3, sl, sh, l, h, v, g;
mov.b32 L0, 0xC0800000;
mov.b32 L1, 0xC0804000;
mov.b32 H0, 0x3F3F3F00;
mov.b32 H1, 0x40404040;
mov.b32 Z, 0;
and.b32 t, $4, 0x07070707;
shl.b32 x, t, 4;
or.b32 s0, t, x;
shr.b32 s1, s0, 16;
shr.b32 t, $4, 4;
and.b32 t, t, 0x07070707;
shl.b32 x, t, 4;
or.b32 s2, t, x;
shr.b32 s3, s2, 16;
and.b32 sl, $4, 0x08080808;
shl.b32 sl, sl, 4;
and.b32 sh, $4, 0x80808080;
prmt.b32 l, L0, L1, s0;
prmt.b32 h, H0, H1, s0;
prmt.b32 v, l, h, 0x7250;
prmt.b32 g, sl, Z, 0x1404;
or.b32 v, v, g;
mul.rn.bf16x2 $0, v, $5;
prmt.b32 l, L0, L1, s1;
prmt.b32 h, H0, H1, s1;
prmt.b32 v, l, h, 0x7250;
prmt.b32 g, sl, Z, 0x3424;
or.b32 v, v, g;
mul.rn.bf16x2 $1, v, $6;
prmt.b32 l, L0, L1, s2;
prmt.b32 h, H0, H1, s2;
prmt.b32 v, l, h, 0x7250;
prmt.b32 g, sh, Z, 0x1404;
or.b32 v, v, g;
mul.rn.bf16x2 $2, v, $5;
prmt.b32 l, L0, L1, s3;
prmt.b32 h, H0, H1, s3;
prmt.b32 v, l, h, 0x7250;
prmt.b32 g, sh, Z, 0x3424;
or.b32 v, v, g;
mul.rn.bf16x2 $3, v, $6;
}
""")
# Same decode; the 4 e4m3 scale bytes arrive raw and are widened to bf16 in-register (positive normal scales only).
_DEQ_ASM_FS = tl.constexpr("""
{
.reg .b32 c01, c23;
.reg .b32 L0, L1, H0, H1, Z, t, x, s0, s1, s2, s3, sl, sh, l, h, v, g;
mov.b32 L0, 0xC0800000;
mov.b32 L1, 0xC0804000;
mov.b32 H0, 0x3F3F3F00;
mov.b32 H1, 0x40404040;
mov.b32 Z, 0;
prmt.b32 c01, $5, Z, 0x4140;
and.b32 c01, c01, 0x007F007F;
shl.b32 c01, c01, 4;
add.u32 c01, c01, 0x3C003C00;
prmt.b32 c23, $5, Z, 0x4342;
and.b32 c23, c23, 0x007F007F;
shl.b32 c23, c23, 4;
add.u32 c23, c23, 0x3C003C00;
and.b32 t, $4, 0x07070707;
shl.b32 x, t, 4;
or.b32 s0, t, x;
shr.b32 s1, s0, 16;
shr.b32 t, $4, 4;
and.b32 t, t, 0x07070707;
shl.b32 x, t, 4;
or.b32 s2, t, x;
shr.b32 s3, s2, 16;
and.b32 sl, $4, 0x08080808;
shl.b32 sl, sl, 4;
and.b32 sh, $4, 0x80808080;
prmt.b32 l, L0, L1, s0;
prmt.b32 h, H0, H1, s0;
prmt.b32 v, l, h, 0x7250;
prmt.b32 g, sl, Z, 0x1404;
or.b32 v, v, g;
mul.rn.bf16x2 $0, v, c01;
prmt.b32 l, L0, L1, s1;
prmt.b32 h, H0, H1, s1;
prmt.b32 v, l, h, 0x7250;
prmt.b32 g, sl, Z, 0x3424;
or.b32 v, v, g;
mul.rn.bf16x2 $1, v, c23;
prmt.b32 l, L0, L1, s2;
prmt.b32 h, H0, H1, s2;
prmt.b32 v, l, h, 0x7250;
prmt.b32 g, sh, Z, 0x1404;
or.b32 v, v, g;
mul.rn.bf16x2 $2, v, c01;
prmt.b32 l, L0, L1, s3;
prmt.b32 h, H0, H1, s3;
prmt.b32 v, l, h, 0x7250;
prmt.b32 g, sh, Z, 0x3424;
or.b32 v, v, g;
mul.rn.bf16x2 $3, v, c23;
}
""")


@triton.jit
def _dequant(b3, s, R: tl.constexpr, FAST: tl.constexpr):
    # b3[r, c0, j] holds original byte c0 + 4j of a 64-byte chunk; scale index is j // 2.
    if FAST:
        su = s.to(tl.uint8, bitcast=True)
        sb = tl.broadcast_to(su[:, None, :, None], (R, 4, 8, 2)).reshape(R, 4, 16)
        lo, hi = tl.inline_asm_elementwise(_DEQ_ASM_FS, "=r,=r,=r,=r,r,r", [b3, sb],
                                           dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)
    else:
        sf = s.to(tl.bfloat16)
        sb = tl.broadcast_to(sf[:, None, :, None], (R, 4, 8, 2)).reshape(R, 4, 16)
        lo, hi = tl.inline_asm_elementwise(_DEQ_ASM, "=r,=r,=r,=r,r,r,r", [b3, sb],
                                           dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)
    return tl.permute(tl.join(lo, hi), (0, 2, 1, 3)).reshape(R, 128)


# scr layout: [0,N) counts, [N,2N) starts, [2N,3N) active experts, 3N total rows, 3N+1 active count.
@triton.jit
def _emit_tiles(scr_ptr, nt, te_ptr, tr_ptr, tend_ptr, NLOC: tl.constexpr, BMX: tl.constexpr,
                TB: tl.constexpr, TMAX: tl.constexpr):
    tinc = tl.cumsum(nt, 0)
    ex = tl.arange(0, NLOC)
    t = tl.arange(0, TB)
    et = tl.sum((tinc[None, :] <= t[:, None]).to(tl.int32), axis=1)
    hit = (et[:, None] == ex[None, :]).to(tl.int32)
    s_e = tl.sum(hit * tl.load(scr_ptr + NLOC + ex, volatile=True)[None, :], axis=1)
    c_e = tl.sum(hit * tl.load(scr_ptr + ex, volatile=True)[None, :], axis=1)
    ts_e = tl.sum(hit * (tinc - nt)[None, :], axis=1)
    tm = t < TMAX
    tl.store(te_ptr + t, et, mask=tm)
    tl.store(tr_ptr + t, s_e + (t - ts_e) * BMX, mask=tm)
    tl.store(tend_ptr + t, s_e + c_e, mask=tm)


@triton.jit
def _route_kernel(ids_ptr, scr_ptr, tok_ptr, inv_ptr, te_ptr, tr_ptr, tend_ptr, he_ptr, hr_ptr, hend_ptr, lo, P,
                  NLOC: tl.constexpr, TOPK: tl.constexpr, PB: tl.constexpr, BM: tl.constexpr,
                  TB: tl.constexpr, TMAX: tl.constexpr, BMB: tl.constexpr, HOT: tl.constexpr,
                  TBH: tl.constexpr, TMAXH: tl.constexpr):
    offs = tl.arange(0, PB)
    pm = offs < P
    eid = tl.load(ids_ptr + offs, mask=pm, other=-1) - lo
    loc = pm & (eid >= 0) & (eid < NLOC)
    ex = tl.arange(0, NLOC)
    tl.store(scr_ptr + ex, tl.zeros([NLOC], dtype=tl.int32))
    tl.debug_barrier()
    rank = tl.atomic_add(scr_ptr + eid, 1, mask=loc)
    tl.debug_barrier()
    cnt = tl.load(scr_ptr + ex, volatile=True)
    start = tl.cumsum(cnt, 0) - cnt
    act = (cnt > 0).to(tl.int32)
    slot = tl.cumsum(act, 0) - act
    tl.store(scr_ptr + NLOC + ex, start)
    tl.store(scr_ptr + 2 * NLOC + slot, ex, mask=cnt > 0)
    tl.store(scr_ptr + 3 * NLOC + ex, tl.zeros([NLOC], dtype=tl.int32) + tl.sum(cnt, 0), mask=ex == 0)
    tl.store(scr_ptr + 3 * NLOC + ex, tl.zeros([NLOC], dtype=tl.int32) + tl.sum(act, 0), mask=ex == 1)
    tl.debug_barrier()
    pos = tl.load(scr_ptr + NLOC + eid, mask=loc, other=0, volatile=True) + rank
    tl.store(inv_ptr + offs, pos, mask=loc)
    tl.store(tok_ptr + pos, offs // TOPK, mask=loc)
    hot = cnt >= HOT
    _emit_tiles(scr_ptr, tl.where(hot, 0, (cnt + BM - 1) // BM), te_ptr, tr_ptr, tend_ptr, NLOC, BM, TB, TMAX)
    if TMAXH > 0:
        _emit_tiles(scr_ptr, tl.where(hot, (cnt + BMB - 1) // BMB, 0), he_ptr, hr_ptr, hend_ptr, NLOC, BMB, TBH, TMAXH)


@triton.jit
def _zero_rows_kernel(buf_ptr, scr_ptr, NLOC: tl.constexpr, W: tl.constexpr):
    r = tl.program_id(0)
    if r >= tl.load(scr_ptr + 3 * NLOC):
        return
    c = tl.arange(0, W)
    tl.store(buf_ptr + r.to(tl.int64) * W + c, tl.zeros([W], dtype=tl.float32))


@triton.jit
def _w13_tile(x_ptr, wbase, sbase, tok_ptr, h_ptr, col, gs, row0, rend, k0,
              H: tl.constexpr, I: tl.constexpr, SPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, FAST: tl.constexpr):
    rows = row0 + tl.arange(0, BM)
    rmask = rows < rend
    tok = tl.load(tok_ptr + rows, mask=rmask, other=0).to(tl.int64)
    xp = x_ptr + tok[None, :] * H + (k0 + tl.arange(0, BK))[:, None]
    acc = tl.zeros((2 * BN, BM), dtype=tl.float32)
    for k in range(0, H // SPLIT, BK):
        xt = tl.load(xp + k, mask=rmask[None, :], other=0.0)
        w = _dequant(tl.load(wbase + k // 2), tl.load(sbase + k // 16), 2 * BN, FAST)
        acc = tl.dot(w, xt, acc)
    acc = acc * gs
    hp = h_ptr + rows.to(tl.int64)[None, :] * (2 * I) + col[:, None]
    if SPLIT > 1:
        tl.atomic_add(hp, acc, mask=rmask[None, :], sem="relaxed")
    else:
        tl.store(hp, acc, mask=rmask[None, :])


@triton.jit
def _w13_kernel(x_ptr, w_ptr, sf_ptr, gs_ptr, tok_ptr, scr_ptr, te_ptr, tr_ptr, tend_ptr, h_ptr,
                NLOC: tl.constexpr, H: tl.constexpr, I: tl.constexpr, SPLIT: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GU: tl.constexpr,
                TILED: tl.constexpr, FAST: tl.constexpr):
    s = tl.program_id(0)
    nb = tl.program_id(1)
    ks = tl.program_id(2)
    if TILED:
        e = tl.load(te_ptr + s)
        if e >= NLOC:
            return
    else:
        if s >= tl.load(scr_ptr + 3 * NLOC + 1):
            return
        e = tl.load(scr_ptr + 2 * NLOC + s)
    j = tl.arange(0, 2 * BN)
    n = nb * BN + j % BN
    is_up = (j >= BN).to(tl.int32)
    r = (n // GU) * (2 * GU) + is_up * GU + n % GU
    col = n + is_up * I
    e64 = e.to(tl.int64)
    k0 = ks * (H // SPLIT)
    q = tl.arange(0, 4)[:, None] * 16 + tl.arange(0, 16)[None, :]
    wbase = w_ptr + e64 * (2 * I * (H // 2)) + r[:, None, None] * (H // 2) + k0 // 2 + q[None, :, :]
    sbase = sf_ptr + e64 * (2 * I * (H // 16)) + r[:, None] * (H // 16) + (k0 // 16 + tl.arange(0, BK // 16))[None, :]
    gs = tl.load(gs_ptr + e)
    if TILED:
        _w13_tile(x_ptr, wbase, sbase, tok_ptr, h_ptr, col, gs, tl.load(tr_ptr + s), tl.load(tend_ptr + s), k0,
                  H, I, SPLIT, BM, BN, BK, FAST)
    else:
        start = tl.load(scr_ptr + NLOC + e)
        end = start + tl.load(scr_ptr + e)
        for m0 in range(start, end, BM):
            _w13_tile(x_ptr, wbase, sbase, tok_ptr, h_ptr, col, gs, m0, end, k0, H, I, SPLIT, BM, BN, BK, FAST)


@triton.jit
def _swiglu_kernel(h_ptr, a_ptr, y_ptr, scr_ptr, NLOC: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
                   ZERO_Y: tl.constexpr):
    r = tl.program_id(0)
    if r >= tl.load(scr_ptr + 3 * NLOC):
        return
    r64 = r.to(tl.int64)
    c = tl.arange(0, I)
    g = tl.load(h_ptr + r64 * (2 * I) + c)
    u = tl.load(h_ptr + r64 * (2 * I) + I + c)
    tl.store(a_ptr + r64 * I + c, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16))
    if ZERO_Y:
        ch = tl.arange(0, H)
        tl.store(y_ptr + r64 * H + ch, tl.zeros([H], dtype=tl.float32))


@triton.jit
def _w2_tile(a_ptr, wbase, sbase, y_ptr, n, gs, row0, rend, k0,
             H: tl.constexpr, I: tl.constexpr, SPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr, FAST: tl.constexpr):
    rows = row0 + tl.arange(0, BM)
    rmask = rows < rend
    r64 = rows.to(tl.int64)
    ap = a_ptr + r64[None, :] * I + (k0 + tl.arange(0, BK))[:, None]
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for k in range(0, I // SPLIT, BK):
        at = tl.load(ap + k, mask=rmask[None, :], other=0.0)
        w = _dequant(tl.load(wbase + k // 2), tl.load(sbase + k // 16), BN, FAST)
        acc = tl.dot(w, at, acc)
    acc = acc * gs
    yp = y_ptr + r64[None, :] * H + n[:, None]
    if SPLIT > 1:
        tl.atomic_add(yp, acc, mask=rmask[None, :], sem="relaxed")
    else:
        tl.store(yp, acc, mask=rmask[None, :])


@triton.jit
def _w2_kernel(a_ptr, w_ptr, sf_ptr, gs_ptr, scr_ptr, te_ptr, tr_ptr, tend_ptr, y_ptr,
               NLOC: tl.constexpr, H: tl.constexpr, I: tl.constexpr, SPLIT: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               TILED: tl.constexpr, FAST: tl.constexpr):
    s = tl.program_id(0)
    nb = tl.program_id(1)
    ks = tl.program_id(2)
    if TILED:
        e = tl.load(te_ptr + s)
        if e >= NLOC:
            return
    else:
        if s >= tl.load(scr_ptr + 3 * NLOC + 1):
            return
        e = tl.load(scr_ptr + 2 * NLOC + s)
    n = nb * BN + tl.arange(0, BN)
    e64 = e.to(tl.int64)
    k0 = ks * (I // SPLIT)
    q = tl.arange(0, 4)[:, None] * 16 + tl.arange(0, 16)[None, :]
    wbase = w_ptr + e64 * (H * (I // 2)) + n[:, None, None] * (I // 2) + k0 // 2 + q[None, :, :]
    sbase = sf_ptr + e64 * (H * (I // 16)) + n[:, None] * (I // 16) + (k0 // 16 + tl.arange(0, BK // 16))[None, :]
    gs = tl.load(gs_ptr + e)
    if TILED:
        _w2_tile(a_ptr, wbase, sbase, y_ptr, n, gs, tl.load(tr_ptr + s), tl.load(tend_ptr + s), k0,
                 H, I, SPLIT, BM, BN, BK, FAST)
    else:
        start = tl.load(scr_ptr + NLOC + e)
        end = start + tl.load(scr_ptr + e)
        for m0 in range(start, end, BM):
            _w2_tile(a_ptr, wbase, sbase, y_ptr, n, gs, m0, end, k0, H, I, SPLIT, BM, BN, BK, FAST)


@triton.jit
def _combine_kernel(y_ptr, ids_ptr, tw_ptr, inv_ptr, pw_ptr, out_ptr, lo, eps,
                    NLOC: tl.constexpr, H: tl.constexpr, K: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.arange(0, H)
    scale = 1.0 + tl.load(pw_ptr + cols).to(tl.float32)
    acc = tl.zeros((H,), dtype=tl.float32)
    for s in tl.static_range(K):
        eid = tl.load(ids_ptr + t * K + s) - lo
        if (eid >= 0) & (eid < NLOC):
            pos = tl.load(inv_ptr + t * K + s).to(tl.int64)
            y = tl.load(y_ptr + pos * H + cols) * scale
            rn = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * (rn * tl.load(tw_ptr + t * K + s))
    tl.store(out_ptr + t.to(tl.int64) * H + cols, acc.to(tl.bfloat16))


def _interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    e, two_i, c = t.shape
    i = two_i // 2
    return t.view(e, 2, i // GU_BLOCK, GU_BLOCK, c).permute(0, 2, 1, 3, 4).contiguous().view(e, two_i, c)


def _chunk_repack(w: torch.Tensor) -> torch.Tensor:
    # Within each 64-byte K chunk: byte c0 + 4j moves to position 16 * c0 + j.
    *lead, kb = w.shape
    return w.view(*lead, kb // 64, 16, 4).transpose(-1, -2).contiguous().view(*lead, kb)


def _chunk_unpack(w: torch.Tensor) -> torch.Tensor:
    *lead, kb = w.shape
    return w.view(*lead, kb // 64, 4, 16).transpose(-1, -2).contiguous().view(*lead, kb)


def _bucket(m: int) -> int:
    for b in sorted(SPLITS):
        if m <= b:
            return b
    return max(SPLITS)


def _run(data, state, cfg):
    bm, tile_idx, split, tiled, hot = cfg
    bn, bk, warps, stages = TILES[tile_idx]
    fast = state["fast"]
    c = data["config"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    tw = data["topk_weights"].contiguous()
    m, h = x.shape
    i_dim = c["d_expert"]
    lo, nloc = c["local_start"], c["local_count"]
    dev = x.device
    p = ids.numel()
    tmax = min(triton.cdiv(p, bm), triton.cdiv(m, bm) * nloc) + nloc
    slots = tmax if tiled else min(nloc, p)
    scr = torch.empty(5 * nloc, dtype=torch.int32, device=dev)
    tok = torch.empty(p, dtype=torch.int32, device=dev)
    inv = torch.empty(p, dtype=torch.int32, device=dev)
    tiles = torch.empty((3, tmax), dtype=torch.int32, device=dev)
    tmaxh = triton.cdiv(p, HOT_BM) + nloc if hot else 0
    htiles = torch.empty((3, max(tmaxh, 1)), dtype=torch.int32, device=dev)
    _route_kernel[(1,)](ids, scr, tok, inv, tiles[0], tiles[1], tiles[2], htiles[0], htiles[1], htiles[2], lo, p,
                        NLOC=nloc, TOPK=TOP_K, PB=triton.next_power_of_2(p), BM=bm,
                        TB=triton.next_power_of_2(tmax), TMAX=tmax, BMB=HOT_BM,
                        HOT=HOT_MIN if hot else 1 << 30, TBH=triton.next_power_of_2(max(tmaxh, 1)), TMAXH=tmaxh,
                        num_warps=4)
    launches = [(bm, slots, tiles)] + ([(HOT_BM, tmaxh, htiles)] if hot else [])
    hbuf = torch.empty((p, 2 * i_dim), dtype=torch.float32, device=dev)
    if split > 1:
        _zero_rows_kernel[(p,)](hbuf, scr, NLOC=nloc, W=2 * i_dim, num_warps=8)
    sl = slice(lo, lo + nloc)
    for lbm, lslots, lt in launches:
        _w13_kernel[(lslots, i_dim // bn, split)](
            x, state["w13"][sl], state["w13_sf"][sl], state["w13_gs"][sl], tok, scr, lt[0], lt[1], lt[2],
            hbuf, NLOC=nloc, H=h, I=i_dim, SPLIT=split, BM=lbm, BN=bn, BK=bk, GU=GU_BLOCK, TILED=tiled, FAST=fast,
            num_warps=warps, num_stages=stages)
    inter = torch.empty((p, i_dim), dtype=torch.bfloat16, device=dev)
    y = torch.empty((p, h), dtype=torch.float32, device=dev)
    _swiglu_kernel[(p,)](hbuf, inter, y, scr, NLOC=nloc, I=i_dim, H=h, ZERO_Y=split > 1, num_warps=8)
    for lbm, lslots, lt in launches:
        _w2_kernel[(lslots, h // bn, split)](
            inter, state["w2"][sl], state["w2_sf"][sl], state["w2_gs"][sl], scr, lt[0], lt[1], lt[2], y,
            NLOC=nloc, H=h, I=i_dim, SPLIT=split, BM=lbm, BN=bn, BK=bk, TILED=tiled, FAST=fast,
            num_warps=warps, num_stages=stages)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
    _combine_kernel[(m,)](y, ids, tw, inv, state["post_norm_w"], out, lo, EPS,
                          NLOC=nloc, H=h, K=TOP_K, num_warps=8)
    return out


def _synthetic(bs, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    n_exp, nloc = 256, 32
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    pop = 0.03 * torch.randn(n_exp, generator=g, device=dev) / (-torch.rand(n_exp, generator=g, device=dev).clamp(min=1e-12).log()).sqrt()
    scores = torch.rand((bs, n_exp), generator=g, device=dev) + pop
    top = scores.topk(TOP_K, dim=-1)
    ids = top.indices
    gpu_load = torch.stack([((ids // nloc) == r).sum() for r in range(n_exp // nloc)])
    ids = (ids - nloc * gpu_load.argmax()) % n_exp
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": n_exp,
                       "local_start": 0, "local_count": nloc, "top_k": TOP_K, "sharding": "ep8"}}


def _time(data, state, cfg, flush, reps=10):
    _run(data, state, cfg)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(data, state, cfg)
    graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        flush.zero_()
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
    for bs in sorted(SPLITS):
        data = _synthetic(bs, dev, 1234 + bs)
        best = None
        for bm in BMS[bs]:
            for ti in range(len(TILES)):
                modes = [(td, False) for td in (False, True)] + ([(True, True)] if bs >= 128 else [])
                for sp, (tiled, hot) in [(sp, md) for sp in SPLITS[bs] for md in modes]:
                    if time.perf_counter() - t0 > SWEEP_BUDGET_S:
                        break
                    cfg = (bm, ti, sp, tiled, hot)
                    try:
                        t = _time(data, state, cfg, flush)
                    except Exception as exc:  # noqa: BLE001
                        print(f"sweep bs={bs} cfg={cfg} failed: {type(exc).__name__}: {exc}")
                        continue
                    if best is None or t < best[0]:
                        best = (t, cfg)
        if best is not None:
            chosen[bs] = best[1]
        print(f"sweep bs={bs} -> {chosen[bs]} {best[0] * 1e3 if best else float('nan'):.1f} us")
    del flush
    print(f"sweep done in {time.perf_counter() - t0:.1f} s")
    return chosen


def prepare(weights: Weights):
    # Lossless row permutation: per GU_BLOCK of expert width, gate rows then up rows.
    state = {
        "w13": _chunk_repack(_interleave_gate_up(weights["w13"])),
        "w13_sf": _interleave_gate_up(weights["w13_sf"].view(torch.uint8)).view(torch.float8_e4m3fn),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": _chunk_repack(weights["w2"]),
        "w2_sf": weights["w2_sf"].contiguous(),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "post_norm_w": weights["post_norm_w"].contiguous(),
    }
    # The in-register scale widening is exact only for positive normal e4m3 scales.
    state["fast"] = all(bool(((t.view(torch.uint8) & 0x78) != 0).all() and ((t.view(torch.uint8) & 0x80) == 0).all())
                        for t in (weights["w13_sf"], weights["w2_sf"]))
    print(f"fast scale path: {state['fast']}")
    state["cfg"] = _sweep(state, weights["w13"].device)
    return state


def kernel_fn(data: Inputs, state):
    return _run(data, state, state["cfg"][_bucket(data["hidden_states"].shape[0])])
