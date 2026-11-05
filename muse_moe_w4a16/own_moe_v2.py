# okc: problem=muse_moe_w4a16
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
BN = 64
BK = 128
# bs -> (BM, split-K for W13, split-K for W2, num_stages)
CONFIGS = {1: (16, 8, 8, 4), 8: (16, 4, 4, 4), 16: (16, 2, 2, 4), 32: (16, 2, 2, 4),
           128: (16, 1, 1, 4), 256: (32, 1, 1, 4)}

# Byte LUT decode of 4 packed bytes (8 E2M1 codes) into bf16 pairs, times the block scale
# (exact in bf16). prmt picks the low/high byte of each bf16 magnitude from 8-entry tables;
# the sign bit is moved separately.
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


@triton.jit
def _dequant(b, s, R: tl.constexpr, K: tl.constexpr):
    sb = tl.broadcast_to(s.to(tl.bfloat16)[:, :, None], (R, K // 16, 8)).reshape(R, K // 2)
    lo, hi = tl.inline_asm_elementwise(_DEQ_ASM, "=r,=r,=r,=r,r,r,r", [b, sb],
                                       dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)
    return tl.join(lo, hi).reshape(R, K)


@triton.jit
def _route_kernel(ids_ptr, scr_ptr, tok_ptr, inv_ptr, te_ptr, tr_ptr, tend_ptr, lo, P,
                  NLOC: tl.constexpr, BM: tl.constexpr, TOPK: tl.constexpr,
                  PB: tl.constexpr, TB: tl.constexpr, TMAX: tl.constexpr):
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
    nt = (cnt + BM - 1) // BM
    tinc = tl.cumsum(nt, 0)
    tl.store(scr_ptr + NLOC + ex, start)
    tl.store(scr_ptr + 2 * NLOC + ex, tinc - nt)
    tl.store(scr_ptr + 3 * NLOC + ex, tl.zeros([NLOC], dtype=tl.int32) + tl.sum(cnt, 0))
    tl.debug_barrier()
    pos = tl.load(scr_ptr + NLOC + eid, mask=loc, other=0, volatile=True) + rank
    tl.store(inv_ptr + offs, pos, mask=loc)
    tl.store(tok_ptr + pos, offs // TOPK, mask=loc)
    t = tl.arange(0, TB)
    et = tl.sum((tinc[None, :] <= t[:, None]).to(tl.int32), axis=1)
    ec = tl.minimum(et, NLOC - 1)
    s_e = tl.load(scr_ptr + NLOC + ec, volatile=True)
    c_e = tl.load(scr_ptr + ec, volatile=True)
    ts_e = tl.load(scr_ptr + 2 * NLOC + ec, volatile=True)
    tm = t < TMAX
    tl.store(te_ptr + t, et, mask=tm)
    tl.store(tr_ptr + t, s_e + (t - ts_e) * BM, mask=tm)
    tl.store(tend_ptr + t, s_e + c_e, mask=tm)


@triton.jit
def _zero_rows_kernel(buf_ptr, scr_ptr, NLOC: tl.constexpr, W: tl.constexpr):
    r = tl.program_id(0)
    if r >= tl.load(scr_ptr + 3 * NLOC):
        return
    c = tl.arange(0, W)
    tl.store(buf_ptr + r.to(tl.int64) * W + c, tl.zeros([W], dtype=tl.float32))


@triton.jit
def _w13_kernel(x_ptr, w_ptr, sf_ptr, gs_ptr, tok_ptr, te_ptr, tr_ptr, tend_ptr, h_ptr,
                NLOC: tl.constexpr, H: tl.constexpr, I: tl.constexpr, SPLIT: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    ks = tl.program_id(2)
    e = tl.load(te_ptr + t)
    if e >= NLOC:
        return
    row0 = tl.load(tr_ptr + t)
    rend = tl.load(tend_ptr + t)
    rows = row0 + tl.arange(0, BM)
    rmask = rows < rend
    tok = tl.load(tok_ptr + rows, mask=rmask, other=0).to(tl.int64)
    r = nb * (2 * BN) + tl.arange(0, 2 * BN)
    e64 = e.to(tl.int64)
    KS = H // SPLIT
    k0 = ks * KS
    wp = w_ptr + e64 * (2 * I * (H // 2)) + r[:, None] * (H // 2) + (k0 // 2 + tl.arange(0, BK // 2))[None, :]
    sp = sf_ptr + e64 * (2 * I * (H // 16)) + r[:, None] * (H // 16) + (k0 // 16 + tl.arange(0, BK // 16))[None, :]
    xp = x_ptr + tok[None, :] * H + (k0 + tl.arange(0, BK))[:, None]
    acc = tl.zeros((2 * BN, BM), dtype=tl.float32)
    for k in range(0, KS, BK):
        xt = tl.load(xp + k, mask=rmask[None, :], other=0.0)
        w = _dequant(tl.load(wp + k // 2), tl.load(sp + k // 16), 2 * BN, BK)
        acc = tl.dot(w, xt, acc)
    acc = acc * tl.load(gs_ptr + e)
    j = tl.arange(0, 2 * BN)
    col = tl.where(j < BN, nb * BN + j, I + nb * BN + j - BN)
    hp = h_ptr + rows.to(tl.int64)[None, :] * (2 * I) + col[:, None]
    if SPLIT == 1:
        tl.store(hp, acc, mask=rmask[None, :])
    else:
        tl.atomic_add(hp, acc, mask=rmask[None, :], sem="relaxed")


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
def _w2_kernel(a_ptr, w_ptr, sf_ptr, gs_ptr, te_ptr, tr_ptr, tend_ptr, y_ptr,
               NLOC: tl.constexpr, H: tl.constexpr, I: tl.constexpr, SPLIT: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    ks = tl.program_id(2)
    e = tl.load(te_ptr + t)
    if e >= NLOC:
        return
    row0 = tl.load(tr_ptr + t)
    rend = tl.load(tend_ptr + t)
    rows = row0 + tl.arange(0, BM)
    rmask = rows < rend
    n = nb * BN + tl.arange(0, BN)
    e64 = e.to(tl.int64)
    KS = I // SPLIT
    k0 = ks * KS
    wp = w_ptr + e64 * (H * (I // 2)) + n[:, None] * (I // 2) + (k0 // 2 + tl.arange(0, BK // 2))[None, :]
    sp = sf_ptr + e64 * (H * (I // 16)) + n[:, None] * (I // 16) + (k0 // 16 + tl.arange(0, BK // 16))[None, :]
    ap = a_ptr + rows.to(tl.int64)[None, :] * I + (k0 + tl.arange(0, BK))[:, None]
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for k in range(0, KS, BK):
        at = tl.load(ap + k, mask=rmask[None, :], other=0.0)
        w = _dequant(tl.load(wp + k // 2), tl.load(sp + k // 16), BN, BK)
        acc = tl.dot(w, at, acc)
    acc = acc * tl.load(gs_ptr + e)
    yp = y_ptr + rows.to(tl.int64)[None, :] * H + n[:, None]
    if SPLIT == 1:
        tl.store(yp, acc, mask=rmask[None, :])
    else:
        tl.atomic_add(yp, acc, mask=rmask[None, :], sem="relaxed")


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
    return t.view(e, 2, i // BN, BN, c).permute(0, 2, 1, 3, 4).contiguous().view(e, two_i, c)


def prepare(weights: Weights):
    # Pure row permutation: per BN block of the expert width, BN gate rows then BN up rows.
    return {
        "w13": _interleave_gate_up(weights["w13"]),
        "w13_sf": _interleave_gate_up(weights["w13_sf"].view(torch.uint8)).view(torch.float8_e4m3fn),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": weights["w2"].contiguous(),
        "w2_sf": weights["w2_sf"].contiguous(),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "post_norm_w": weights["post_norm_w"].contiguous(),
    }


def _config(m: int):
    for b in sorted(CONFIGS):
        if m <= b:
            return CONFIGS[b]
    return CONFIGS[max(CONFIGS)]


def kernel_fn(data: Inputs, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    tw = data["topk_weights"].contiguous()
    m, h = x.shape
    i_dim = cfg["d_expert"]
    lo, nloc = cfg["local_start"], cfg["local_count"]
    bm, s13, s2, stages = _config(m)
    dev = x.device
    p = ids.numel()
    tmax = min(triton.cdiv(p, bm), triton.cdiv(m, bm) * nloc) + nloc
    scr = torch.empty(4 * nloc, dtype=torch.int32, device=dev)
    tok = torch.empty(p, dtype=torch.int32, device=dev)
    inv = torch.empty(p, dtype=torch.int32, device=dev)
    tiles = torch.empty((3, tmax), dtype=torch.int32, device=dev)
    _route_kernel[(1,)](ids, scr, tok, inv, tiles[0], tiles[1], tiles[2], lo, p,
                        NLOC=nloc, BM=bm, TOPK=TOP_K, PB=triton.next_power_of_2(p),
                        TB=triton.next_power_of_2(tmax), TMAX=tmax, num_warps=4)
    hbuf = torch.empty((p, 2 * i_dim), dtype=torch.float32, device=dev)
    if s13 > 1:
        _zero_rows_kernel[(p,)](hbuf, scr, NLOC=nloc, W=2 * i_dim, num_warps=8)
    sl = slice(lo, lo + nloc)
    _w13_kernel[(tmax, i_dim // BN, s13)](x, state["w13"][sl], state["w13_sf"][sl], state["w13_gs"][sl],
                                          tok, tiles[0], tiles[1], tiles[2], hbuf,
                                          NLOC=nloc, H=h, I=i_dim, SPLIT=s13, BM=bm, BN=BN, BK=BK,
                                          num_warps=4, num_stages=stages)
    inter = torch.empty((p, i_dim), dtype=torch.bfloat16, device=dev)
    y = torch.empty((p, h), dtype=torch.float32, device=dev)
    _swiglu_kernel[(p,)](hbuf, inter, y, scr, NLOC=nloc, I=i_dim, H=h, ZERO_Y=s2 > 1, num_warps=8)
    _w2_kernel[(tmax, h // BN, s2)](inter, state["w2"][sl], state["w2_sf"][sl], state["w2_gs"][sl],
                                    tiles[0], tiles[1], tiles[2], y,
                                    NLOC=nloc, H=h, I=i_dim, SPLIT=s2, BM=bm, BN=BN, BK=BK,
                                    num_warps=4, num_stages=stages)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
    _combine_kernel[(m,)](y, ids, tw, inv, state["post_norm_w"], out, lo, EPS,
                          NLOC=nloc, H=h, K=TOP_K, num_warps=8)
    return out
