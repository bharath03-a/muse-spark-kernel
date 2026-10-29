# okc: problem=muse_moe_w4a16
"""Routed experts for Muse Spark Open on H200: a Triton NVFP4 grouped GEMM.

- Weights stay in their NVFP4 form (packed E2M1 + E4M3 block scales + FP32 global scale).
  Each tile is decoded in registers: e2m1 * block_scale is exact in BF16 (<= 6 significant
  bits), so the tensor-core inputs are the exact dequantized weights; the global scale is
  applied to the FP32 accumulator.
- GEMM1 computes the gate and up tiles together and writes silu(gate) * up directly
  (no [pairs, 2I] intermediate, no separate activation kernel).
- GEMM2 writes each (token, expert) row; SGLang's Triton post-expert norm weighted sum
  finishes the token.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import SMALL_NUMEL_LIMIT, moe_align_small_numel
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
POST_NORM_EPS = 1e-8


@triton.jit
def _decode_nvfp4(packed, scale_f32, BN: tl.constexpr, BK: tl.constexpr):
    """packed uint8 [BN, BK//2], scale f32 [BN, BK//16] -> bf16 [BN, BK] (exact)."""
    p = packed.to(tl.int32)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    nib = tl.reshape(tl.join(lo, hi), (BN, BK))  # element 2i = low nibble
    c = nib & 7
    bits = tl.where(c >= 2, 0x3F00 + (c << 6), tl.where(c == 1, 0x3F00, 0))
    bits = bits | ((nib & 8) << 12)
    val = bits.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    s = tl.reshape(tl.broadcast_to(scale_f32[:, :, None], (BN, BK // 16, 16)), (BN, BK))
    return (val * s).to(tl.bfloat16)


@triton.jit
def _moe_nvfp4_gemm_kernel(
    a_ptr, out_ptr,
    w_ptr, sf_ptr, gs_ptr,
    sorted_ids_ptr, expert_ids_ptr, n_padded_ptr,
    numel,                      # valid pair count (padding id == numel)
    N, K,                       # output cols (per half for SWIGLU), reduction dim
    stride_am, stride_om,
    stride_we, stride_wn,       # packed weight: [E, rows, K/2]
    stride_se, stride_sn,       # scales: [E, rows, K/16]
    A_ROW_DIV: tl.constexpr,    # GEMM1: token = pair // top_k; GEMM2: 1
    SWIGLU: tl.constexpr,       # GEMM1: gate rows [n], up rows [N + n]
    NUM_LOCAL: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_padded = tl.load(n_padded_ptr)
    if pid_m * BM >= n_padded:
        return
    e = tl.load(expert_ids_ptr + pid_m)
    if e < 0 or e >= NUM_LOCAL:
        return
    e64 = e.to(tl.int64)

    offs_m = pid_m * BM + tl.arange(0, BM)
    pair = tl.load(sorted_ids_ptr + offs_m)
    row_ok = pair < numel
    a_row = (pair // A_ROW_DIV).to(tl.int64)

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_kp = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 16)

    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :]
    w_base = w_ptr + e64 * stride_we
    s_base = sf_ptr + e64 * stride_se
    wg_ptrs = w_base + offs_n[:, None].to(tl.int64) * stride_wn + offs_kp[None, :]
    sg_ptrs = s_base + offs_n[:, None].to(tl.int64) * stride_sn + offs_ks[None, :]
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    if SWIGLU:
        wu_ptrs = wg_ptrs + N * stride_wn
        su_ptrs = sg_ptrs + N * stride_sn
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=row_ok[:, None], other=0.0)
        wg = _decode_nvfp4(tl.load(wg_ptrs), tl.load(sg_ptrs).to(tl.float32), BN, BK)
        acc_g = tl.dot(a, tl.trans(wg), acc_g)
        if SWIGLU:
            wu = _decode_nvfp4(tl.load(wu_ptrs), tl.load(su_ptrs).to(tl.float32), BN, BK)
            acc_u = tl.dot(a, tl.trans(wu), acc_u)
            wu_ptrs += BK // 2
            su_ptrs += BK // 16
        a_ptrs += BK
        wg_ptrs += BK // 2
        sg_ptrs += BK // 16

    gs = tl.load(gs_ptr + e)
    if SWIGLU:
        g = acc_g * gs
        u = acc_u * gs
        res = g / (1.0 + tl.exp(-g)) * u
    else:
        res = acc_g * gs
    o_ptrs = out_ptr + pair.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
    tl.store(o_ptrs, res.to(out_ptr.dtype.element_ty), mask=row_ok[:, None])


def _tiles(pairs):
    """(BM, BN1, BK1, warps1, stages1, BN2, BK2, warps2, stages2) by routed-pair count."""
    if pairs <= 64:
        return 16, 32, 256, 4, 3, 64, 256, 4, 3
    if pairs <= 512:
        return 16, 64, 128, 4, 4, 64, 128, 4, 4
    if pairs <= 4096:
        return 64, 64, 64, 4, 4, 128, 64, 8, 4
    return 128, 64, 64, 8, 3, 128, 64, 8, 3


def prepare(weights):
    device = weights["w13"].device
    start, count = 0, 32
    sl = slice(start, start + count)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
    state = {
        "w13": weights["w13"][sl].contiguous(),
        "w13_sf": weights["w13_sf"][sl].contiguous(),
        "w13_gs": weights["w13_gs"][sl].float().contiguous(),
        "w2": weights["w2"][sl].contiguous(),
        "w2_sf": weights["w2_sf"][sl].contiguous(),
        "w2_gs": weights["w2_gs"][sl].float().contiguous(),
        "expert_map": expert_map,
        "count": count,
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }
    torch.cuda.synchronize()
    return state


def _align(local_ids, block_m, num_exp):
    numel = local_ids.numel()
    if numel < num_exp + 1:
        padded = numel * block_m
    else:
        padded = numel + (num_exp + 1) * (block_m - 1)
    padded = (padded + 3) // 4 * 4
    dev = local_ids.device
    sorted_ids = torch.empty((padded,), dtype=torch.int32, device=dev)
    expert_ids = torch.empty((triton.cdiv(padded, block_m),), dtype=torch.int32, device=dev)
    n_padded = torch.empty((1,), dtype=torch.int32, device=dev)
    if numel <= SMALL_NUMEL_LIMIT and num_exp + 1 > 64:
        moe_align_small_numel(local_ids, num_exp + 1, block_m, sorted_ids, expert_ids, n_padded)
    else:
        cumsum = torch.empty((num_exp + 2,), dtype=torch.int32, device=dev)
        sgl_moe_align_block_size(local_ids, num_exp + 1, block_m, sorted_ids, expert_ids, n_padded,
                                 cumsum, True, False)
    return sorted_ids, expert_ids, n_padded


def moe_forward(x, topk_weights, local_ids, st, norm_w):
    """x bf16 [M, H]; local_ids int32 [M, 8] in [0, 32) or -1."""
    num_tokens, H = x.shape
    I = st["w2"].shape[2] * 2
    pairs = num_tokens * TOP_K
    BM, BN1, BK1, W1, S1, BN2, BK2, W2, S2 = _tiles(pairs)
    sorted_ids, expert_ids, n_padded = _align(local_ids, BM, N_EXPERTS)
    m_blocks = expert_ids.numel()

    inter = torch.empty((pairs, I), dtype=torch.bfloat16, device=x.device)
    w13, s13 = st["w13"], st["w13_sf"]
    _moe_nvfp4_gemm_kernel[(m_blocks, I // BN1)](
        x, inter, w13, s13, st["w13_gs"], sorted_ids, expert_ids, n_padded,
        pairs, I, H, x.stride(0), inter.stride(0),
        w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
        A_ROW_DIV=TOP_K, SWIGLU=True, NUM_LOCAL=st["count"],
        BM=BM, BN=BN1, BK=BK1, num_warps=W1, num_stages=S1)

    y = torch.empty((pairs, H), dtype=torch.bfloat16, device=x.device)
    w2, s2 = st["w2"], st["w2_sf"]
    _moe_nvfp4_gemm_kernel[(m_blocks, H // BN2)](
        inter, y, w2, s2, st["w2_gs"], sorted_ids, expert_ids, n_padded,
        pairs, H, I, inter.stride(0), y.stride(0),
        w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1),
        A_ROW_DIV=1, SWIGLU=False, NUM_LOCAL=st["count"],
        BM=BM, BN=BN2, BK=BK2, num_warps=W2, num_stages=S2)

    out = torch.empty((num_tokens, H), dtype=x.dtype, device=x.device)
    _post_expert_norm_weighted_sum_kernel[(num_tokens,)](
        y, norm_w, topk_weights, out, local_ids,
        TOPK=TOP_K, H=H, eps=POST_NORM_EPS, APPLY_NORM=True, HAS_WEIGHT=True,
        WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=triton.next_power_of_2(H),
        num_warps=8, num_stages=2)
    return out


def kernel_fn(data, state):
    x = data["hidden_states"].contiguous()
    local_ids = state["expert_map"][data["topk_ids"]].contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    return moe_forward(x, topk_weights, local_ids, state, state["norm_w"])
