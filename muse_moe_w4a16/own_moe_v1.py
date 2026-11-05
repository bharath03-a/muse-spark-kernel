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


@triton.jit
def _dequant(b, s, BN: tl.constexpr, BK: tl.constexpr):
    # E2M1 nibble -> bf16 bits directly; times the e4m3 block scale is exact in bf16.
    b = b.to(tl.int32)
    c = tl.join(b & 15, b >> 4).reshape(BN, BK)
    m = c & 7
    bits = tl.where(m < 2, m * (126 << 7), (((m >> 1) + 126) << 7) | ((m & 1) << 6)) | ((c & 8) << 12)
    v = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    v = v.reshape(BN, BK // 16, 16) * s.to(tl.bfloat16)[:, :, None]
    return v.reshape(BN, BK)


@triton.jit
def _w13_kernel(x_ptr, w_ptr, sf_ptr, gs_ptr, tok_ptr, te_ptr, tr_ptr, tend_ptr, out_ptr,
                NLOC: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(te_ptr + t)
    if e >= NLOC:
        return
    row0 = tl.load(tr_ptr + t)
    rend = tl.load(tend_ptr + t)
    rows = row0 + tl.arange(0, BM)
    rmask = rows < rend
    tok = tl.load(tok_ptr + rows, mask=rmask, other=0).to(tl.int64)
    n = nb * BN + tl.arange(0, BN)
    e64 = e.to(tl.int64)
    wg = w_ptr + e64 * (2 * I * (H // 2)) + n[:, None] * (H // 2) + tl.arange(0, BK // 2)[None, :]
    wu = wg + I * (H // 2)
    sg = sf_ptr + e64 * (2 * I * (H // 16)) + n[:, None] * (H // 16) + tl.arange(0, BK // 16)[None, :]
    su = sg + I * (H // 16)
    xp = x_ptr + tok[:, None] * H + tl.arange(0, BK)[None, :]
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, H, BK):
        xa = tl.load(xp + k, mask=rmask[:, None], other=0.0)
        g = _dequant(tl.load(wg + k // 2), tl.load(sg + k // 16), BN, BK)
        u = _dequant(tl.load(wu + k // 2), tl.load(su + k // 16), BN, BK)
        accg = tl.dot(xa, tl.trans(g), accg)
        accu = tl.dot(xa, tl.trans(u), accu)
    gs = tl.load(gs_ptr + e)
    accg = accg * gs
    accu = accu * gs
    inter = accg / (1.0 + tl.exp(-accg)) * accu
    op = out_ptr + rows.to(tl.int64)[:, None] * I + n[None, :]
    tl.store(op, inter.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _w2_kernel(a_ptr, w_ptr, sf_ptr, gs_ptr, te_ptr, tr_ptr, tend_ptr, out_ptr,
               NLOC: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(te_ptr + t)
    if e >= NLOC:
        return
    row0 = tl.load(tr_ptr + t)
    rend = tl.load(tend_ptr + t)
    rows = row0 + tl.arange(0, BM)
    rmask = rows < rend
    n = nb * BN + tl.arange(0, BN)
    e64 = e.to(tl.int64)
    wp = w_ptr + e64 * (H * (I // 2)) + n[:, None] * (I // 2) + tl.arange(0, BK // 2)[None, :]
    sp = sf_ptr + e64 * (H * (I // 16)) + n[:, None] * (I // 16) + tl.arange(0, BK // 16)[None, :]
    ap = a_ptr + rows.to(tl.int64)[:, None] * I + tl.arange(0, BK)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, I, BK):
        a = tl.load(ap + k, mask=rmask[:, None], other=0.0)
        w = _dequant(tl.load(wp + k // 2), tl.load(sp + k // 16), BN, BK)
        acc = tl.dot(a, tl.trans(w), acc)
    acc = acc * tl.load(gs_ptr + e)
    op = out_ptr + rows.to(tl.int64)[:, None] * H + n[None, :]
    tl.store(op, acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _combine_kernel(y_ptr, ids_ptr, tw_ptr, inv_ptr, pw_ptr, out_ptr, lo, eps,
                    NLOC: tl.constexpr, H: tl.constexpr, K: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.arange(0, H)
    scale = 1.0 + tl.load(pw_ptr + cols).to(tl.float32)
    acc = tl.zeros((H,), dtype=tl.float32)
    for s in tl.static_range(K):
        eid = tl.load(ids_ptr + t * K + s) - lo
        is_local = (eid >= 0) & (eid < NLOC)
        if is_local:
            pos = tl.load(inv_ptr + t * K + s).to(tl.int64)
            y = tl.load(y_ptr + pos * H + cols).to(tl.float32) * scale
            r = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * (r * tl.load(tw_ptr + t * K + s))
    tl.store(out_ptr + t.to(tl.int64) * H + cols, acc.to(tl.bfloat16))


def prepare(weights: Weights):
    # Assumes the default ep8 shard (experts 0-31); full slicing happens per call otherwise.
    return {
        "w13": weights["w13"].contiguous(),
        "w13_sf": weights["w13_sf"].contiguous(),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": weights["w2"].contiguous(),
        "w2_sf": weights["w2_sf"].contiguous(),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "post_norm_w": weights["post_norm_w"].contiguous(),
    }


def _route(ids, lo, nloc, bm, m):
    p = ids.numel()
    flat = ids.reshape(-1).to(torch.int64) - lo
    local = (flat >= 0) & (flat < nloc)
    key = torch.where(local, flat, torch.full_like(flat, nloc))
    order = torch.argsort(key, stable=True)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(p, device=ids.device)
    tok = (order // TOP_K).to(torch.int32)
    counts = (key[:, None] == torch.arange(nloc, device=ids.device)[None, :]).sum(0)
    starts = torch.cumsum(counts, 0) - counts
    ntiles = (counts + bm - 1) // bm
    tcum = torch.cumsum(ntiles, 0)
    tmax = min(triton.cdiv(p, bm), triton.cdiv(m, bm) * nloc) + nloc
    t = torch.arange(tmax, device=ids.device)
    te = torch.searchsorted(tcum, t, right=True)
    ec = te.clamp(max=nloc - 1)
    row0 = starts[ec] + (t - (tcum[ec] - ntiles[ec])) * bm
    rend = starts[ec] + counts[ec]
    return tok, inv, te.to(torch.int32), row0.to(torch.int32), rend.to(torch.int32), tmax


def kernel_fn(data: Inputs, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    tw = data["topk_weights"].contiguous()
    m, h = x.shape
    i_dim = cfg["d_expert"]
    lo, nloc = cfg["local_start"], cfg["local_count"]
    bm = 16 if m <= 32 else 64
    bn1, bk1 = 64, 128
    bn2, bk2 = 64, 128
    tok, inv, te, row0, rend, tmax = _route(ids, lo, nloc, bm, m)
    w13 = state["w13"][lo:lo + nloc]
    w13_sf = state["w13_sf"][lo:lo + nloc]
    w2 = state["w2"][lo:lo + nloc]
    w2_sf = state["w2_sf"][lo:lo + nloc]
    gs13 = state["w13_gs"][lo:lo + nloc]
    gs2 = state["w2_gs"][lo:lo + nloc]
    p = ids.numel()
    inter = torch.empty((p, i_dim), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((p, h), dtype=torch.bfloat16, device=x.device)
    _w13_kernel[(tmax, i_dim // bn1)](x, w13, w13_sf, gs13, tok, te, row0, rend, inter,
                                      NLOC=nloc, H=h, I=i_dim, BM=bm, BN=bn1, BK=bk1,
                                      num_warps=4, num_stages=3)
    _w2_kernel[(tmax, h // bn2)](inter, w2, w2_sf, gs2, te, row0, rend, y,
                                 NLOC=nloc, H=h, I=i_dim, BM=bm, BN=bn2, BK=bk2,
                                 num_warps=4, num_stages=3)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=x.device)
    _combine_kernel[(m,)](y, ids, tw, inv, state["post_norm_w"], out, lo, EPS,
                          NLOC=nloc, H=h, K=TOP_K, num_warps=8)
    return out
