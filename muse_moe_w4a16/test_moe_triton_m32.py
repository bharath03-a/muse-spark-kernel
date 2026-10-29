# okc: problem=muse_moe_w4a16
"""Hybrid MoE: Ultra-fast Triton NVFP4 GEMV for M <= 32, Tuned Marlin for M > 32."""

import types
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import (
    SMALL_NUMEL_LIMIT,
    moe_align_small_numel,
)
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

N_EXPERTS = 256
TOP_K = 8
POST_NORM_EPS = 1e-8
MAX_TOKENS = 512
D_EXPERT = 4096
D_HIDDEN = 4096

BLOCK_SIZE_M_MAP = {
    1: 8,
    4: 8,
    8: 8,
    16: 8,
    32: 16,
    64: 16,
    128: 32,
    256: 32,
}


@triton.jit
def _fused_nvfp4_swiglu_gemv_kernel(
    x_ptr,           # [M, K]
    topk_ids_ptr,    # [P] (P = M * TOP_K)
    w13_ptr,         # [32, 2 * I, K // 2]
    w13_sf_ptr,      # [32, 2 * I, K // 16]
    w13_gs_ptr,      # [32]
    out_ptr,         # [P, I]
    local_ids_ptr,   # [P]
    local_start: int,
    local_count: int,
    I: tl.constexpr, # 4096
    K: tl.constexpr, # 4096
    BLOCK_I: tl.constexpr, # 32
    BLOCK_K: tl.constexpr, # 128
):
    pid_p = tl.program_id(0) # pair in [0, P)
    pid_i = tl.program_id(1) # row block in [0, I // BLOCK_I)

    exp_id = tl.load(topk_ids_ptr + pid_p)
    if exp_id < local_start or exp_id >= (local_start + local_count):
        if pid_i == 0:
            tl.store(local_ids_ptr + pid_p, -1)
        return

    local_idx = exp_id - local_start
    if pid_i == 0:
        tl.store(local_ids_ptr + pid_p, local_idx)

    token_idx = pid_p // 8 # TOP_K = 8
    x_tok_ptr = x_ptr + token_idx * K

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < I

    gate_rows = offs_i
    up_rows = offs_i + I

    acc_gate = tl.zeros((BLOCK_I,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_I,), dtype=tl.float32)

    gs = tl.load(w13_gs_ptr + local_idx)

    stride_w_exp = 2 * I * (K // 2)
    stride_sf_exp = 2 * I * (K // 16)
    w_exp_ptr = w13_ptr + local_idx * stride_w_exp
    sf_exp_ptr = w13_sf_ptr + local_idx * stride_sf_exp

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_val = tl.load(x_tok_ptr + offs_k).to(tl.float32)

        offs_k_byte = offs_k // 2
        is_hi = (offs_k % 2) == 1

        # Gate
        gate_bytes = tl.load(w_exp_ptr + gate_rows[:, None] * (K // 2) + offs_k_byte[None, :])
        gate_codes = tl.where(is_hi[None, :], (gate_bytes >> 4) & 0x0F, gate_bytes & 0x0F)
        gate_sign = (gate_codes & 0x08) != 0
        gate_mag_code = gate_codes & 0x07
        gate_mag = tl.where(gate_mag_code < 5, gate_mag_code.to(tl.float32) * 0.5,
                   tl.where(gate_mag_code == 5, 3.0,
                   tl.where(gate_mag_code == 6, 4.0, 6.0)))
        gate_val = tl.where(gate_sign, -gate_mag, gate_mag)

        offs_k_sf = offs_k // 16
        gate_sf = tl.load(sf_exp_ptr + gate_rows[:, None] * (K // 16) + offs_k_sf[None, :]).to(tl.float32)
        gate_dequant = gate_val * gate_sf

        # Up
        up_bytes = tl.load(w_exp_ptr + up_rows[:, None] * (K // 2) + offs_k_byte[None, :])
        up_codes = tl.where(is_hi[None, :], (up_bytes >> 4) & 0x0F, up_bytes & 0x0F)
        up_sign = (up_codes & 0x08) != 0
        up_mag_code = up_codes & 0x07
        up_mag = tl.where(up_mag_code < 5, up_mag_code.to(tl.float32) * 0.5,
                 tl.where(up_mag_code == 5, 3.0,
                 tl.where(up_mag_code == 6, 4.0, 6.0)))
        up_val = tl.where(up_sign, -up_mag, up_mag)

        up_sf = tl.load(sf_exp_ptr + up_rows[:, None] * (K // 16) + offs_k_sf[None, :]).to(tl.float32)
        up_dequant = up_val * up_sf

        acc_gate += tl.sum(gate_dequant * x_val[None, :], axis=1)
        acc_up += tl.sum(up_dequant * x_val[None, :], axis=1)

    gate_final = acc_gate * gs
    up_final = acc_up * gs

    silu_gate = gate_final / (1.0 + tl.exp(-gate_final))
    swiglu = silu_gate * up_final

    out_row_ptr = out_ptr + pid_p * I
    tl.store(out_row_ptr + offs_i, swiglu.to(out_row_ptr.dtype.element_ty), mask=mask_i)


@triton.jit
def _fused_nvfp4_down_gemv_kernel(
    y_inter_ptr,     # [P, I]
    topk_ids_ptr,    # [P]
    w2_ptr,          # [32, H, I // 2]
    w2_sf_ptr,       # [32, H, I // 16]
    w2_gs_ptr,       # [32]
    out_ptr,         # [P, H]
    local_start: int,
    local_count: int,
    H: tl.constexpr, # 4096
    I: tl.constexpr, # 4096
    BLOCK_H: tl.constexpr, # 32
    BLOCK_I: tl.constexpr, # 128
):
    pid_p = tl.program_id(0) # pair in [0, P)
    pid_h = tl.program_id(1) # row block in [0, H // BLOCK_H)

    exp_id = tl.load(topk_ids_ptr + pid_p)
    if exp_id < local_start or exp_id >= (local_start + local_count):
        out_row_ptr = out_ptr + pid_p * H
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        tl.store(out_row_ptr + offs_h, 0.0, mask=offs_h < H)
        return

    local_idx = exp_id - local_start
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    gs = tl.load(w2_gs_ptr + local_idx)

    stride_w2_exp = H * (I // 2)
    stride_sf2_exp = H * (I // 16)
    w2_exp_ptr = w2_ptr + local_idx * stride_w2_exp
    sf2_exp_ptr = w2_sf_ptr + local_idx * stride_sf2_exp
    y_row_ptr = y_inter_ptr + pid_p * I

    for i_start in range(0, I, BLOCK_I):
        offs_i = i_start + tl.arange(0, BLOCK_I)
        y_val = tl.load(y_row_ptr + offs_i).to(tl.float32)

        offs_i_byte = offs_i // 2
        is_hi = (offs_i % 2) == 1

        w2_bytes = tl.load(w2_exp_ptr + offs_h[:, None] * (I // 2) + offs_i_byte[None, :])
        w2_codes = tl.where(is_hi[None, :], (w2_bytes >> 4) & 0x0F, w2_bytes & 0x0F)
        w2_sign = (w2_codes & 0x08) != 0
        w2_mag_code = w2_codes & 0x07
        w2_mag = tl.where(w2_mag_code < 5, w2_mag_code.to(tl.float32) * 0.5,
                 tl.where(w2_mag_code == 5, 3.0,
                 tl.where(w2_mag_code == 6, 4.0, 6.0)))
        w2_val = tl.where(w2_sign, -w2_mag, w2_mag)

        offs_i_sf = offs_i // 16
        w2_sf = tl.load(sf2_exp_ptr + offs_h[:, None] * (I // 16) + offs_i_sf[None, :]).to(tl.float32)
        w2_dequant = w2_val * w2_sf

        acc += tl.sum(w2_dequant * y_val[None, :], axis=1)

    res = acc * gs
    out_row_ptr = out_ptr + pid_p * H
    tl.store(out_row_ptr + offs_h, res.to(out_row_ptr.dtype.element_ty), mask=mask_h)


def _marlin_layer(weights, start, count):
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    starts = [0]
    counts = [32, 256]
    banks = {}
    for start in starts:
        for count in counts:
            if start + count > N_EXPERTS:
                continue
            expert_map = None
            if count != N_EXPERTS:
                expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
                expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
            layer = _marlin_layer(weights, start, count)
            banks[(start, count)] = {
                "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
                "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
                "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
            }
    torch.cuda.synchronize()

    size_n = 4096
    size_k = 4096
    gemm1_n = 2 * size_n
    cache2_buf = torch.empty((MAX_TOKENS * TOP_K, size_n), device=device, dtype=torch.bfloat16)
    cache13_buf = torch.empty((MAX_TOKENS * TOP_K * max(gemm1_n, size_k),), device=device, dtype=torch.bfloat16)

    max_padded = 32768
    align_sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    align_expert_ids = torch.empty((max_padded // 8,), dtype=torch.int32, device=device)
    align_n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    align_cumsum = torch.empty((N_EXPERTS + 2,), dtype=torch.int32, device=device)

    # Triton fast buffers for M <= 32
    max_m_small = 32
    max_p_small = max_m_small * TOP_K
    swiglu_buf = torch.empty((max_p_small, 4096), dtype=torch.bfloat16, device=device)
    y_buf = torch.empty((max_m_small, TOP_K, 4096), dtype=torch.bfloat16, device=device)
    local_ids_buf = torch.empty((max_m_small, TOP_K), dtype=torch.int32, device=device)
    out_buf = torch.empty((max_m_small, 4096), dtype=torch.bfloat16, device=device)

    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "cache2_buf": cache2_buf,
        "cache13_buf": cache13_buf,
        "align_sorted_ids": align_sorted_ids,
        "align_expert_ids": align_expert_ids,
        "align_n_padded": align_n_padded,
        "align_cumsum": align_cumsum,
        # Raw weights for Triton
        "raw_w13": weights["w13"].contiguous(),
        "raw_w13_sf": weights["w13_sf"].contiguous(),
        "raw_w13_gs": weights["w13_gs"].contiguous(),
        "raw_w2": weights["w2"].contiguous(),
        "raw_w2_sf": weights["w2_sf"].contiguous(),
        "raw_w2_gs": weights["w2_gs"].contiguous(),
        "swiglu_buf": swiglu_buf,
        "y_buf": y_buf,
        "local_ids_buf": local_ids_buf,
        "out_buf": out_buf,
    }


def kernel_fn(data, state):
    x = data["hidden_states"]
    num_tokens, size_k = x.shape
    topk_ids = data["topk_ids"]
    topk_weights = data["topk_weights"]
    cfg = data["config"]
    local_start = cfg["local_start"]
    local_count = cfg["local_count"]

    # Fast path: Triton NVFP4 GEMV for M <= 32
    if num_tokens <= 32:
        P = num_tokens * TOP_K
        swiglu_buf = state["swiglu_buf"][:P]
        y_buf = state["y_buf"][:num_tokens]
        local_ids_buf = state["local_ids_buf"][:num_tokens]

        _fused_nvfp4_swiglu_gemv_kernel[(P, 4096 // 32)](
            x,
            topk_ids.view(-1),
            state["raw_w13"],
            state["raw_w13_sf"],
            state["raw_w13_gs"],
            swiglu_buf,
            local_ids_buf.view(-1),
            local_start=local_start,
            local_count=local_count,
            I=4096,
            K=4096,
            BLOCK_I=32,
            BLOCK_K=128,
            num_warps=4,
        )

        _fused_nvfp4_down_gemv_kernel[(P, 4096 // 32)](
            swiglu_buf,
            topk_ids.view(-1),
            state["raw_w2"],
            state["raw_w2_sf"],
            state["raw_w2_gs"],
            y_buf.view(-1, 4096),
            local_start=local_start,
            local_count=local_count,
            H=4096,
            I=4096,
            BLOCK_H=32,
            BLOCK_I=128,
            num_warps=4,
        )

        out = state["out_buf"][:num_tokens]
        _post_expert_norm_weighted_sum_kernel[(num_tokens,)](
            y_buf,
            state["norm_w"],
            topk_weights,
            out,
            local_ids_buf,
            TOPK=TOP_K,
            H=4096,
            eps=POST_NORM_EPS,
            APPLY_NORM=True,
            HAS_WEIGHT=True,
            WEIGHT_FIRST=True,
            HAS_IDS=True,
            BLOCK_H=4096,
            num_warps=8,
            num_stages=2,
        )
        return out

    # Large batch path: Tuned Marlin MoE for M > 32
    bank = state["banks"][(local_start, local_count)]
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = topk_ids
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = topk_weights.contiguous()

    block_size_m = BLOCK_SIZE_M_MAP.get(num_tokens, 32)
    numel = local_ids.numel()
    num_exp = N_EXPERTS if expert_map is not None else bank["count"]
    if numel < num_exp + 1:
        max_num_tokens_padded = numel * block_size_m
    else:
        max_num_tokens_padded = numel + (num_exp + 1) * (block_size_m - 1)
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4
    max_num_m_blocks = max_num_tokens_padded // block_size_m

    sorted_ids = state["align_sorted_ids"][:max_num_tokens_padded]
    block_experts = state["align_expert_ids"][:max_num_m_blocks]
    n_padded = state["align_n_padded"]

    if numel <= SMALL_NUMEL_LIMIT and (num_exp + 1) > 64:
        moe_align_small_numel(
            local_ids,
            num_exp + 1,
            block_size_m,
            sorted_ids,
            block_experts,
            n_padded,
        )
    else:
        sgl_moe_align_block_size(
            local_ids,
            num_exp + 1,
            block_size_m,
            sorted_ids,
            block_experts,
            n_padded,
            state["align_cumsum"],
            True,
            False,
        )

    total_pairs = num_tokens * TOP_K
    cache2 = state["cache2_buf"][:total_pairs, :size_n]
    cache1 = state["cache13_buf"][: total_pairs * gemm1_n].view(-1, gemm1_n)
    cache3 = state["cache13_buf"][: total_pairs * size_k].view(-1, size_k)

    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]

    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=total_pairs, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)

    out = torch.empty((num_tokens, size_k), dtype=x.dtype, device=x.device)
    block_h = triton.next_power_of_2(size_k)
    _post_expert_norm_weighted_sum_kernel[(num_tokens,)](
        y,
        state["norm_w"],
        topk_weights,
        out,
        local_ids,
        TOPK=TOP_K,
        H=size_k,
        eps=POST_NORM_EPS,
        APPLY_NORM=True,
        HAS_WEIGHT=True,
        WEIGHT_FIRST=True,
        HAS_IDS=True,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out
