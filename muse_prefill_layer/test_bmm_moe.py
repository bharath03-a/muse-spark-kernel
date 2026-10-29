# okc: problem=muse_prefill_layer
"""Prefill layer with hybrid MoE:
- T <= 1000: Marlin MoE (optimal for small batch, 745 us)
- T > 1000: Fused NVFP4 dequantize + cuBLAS batched GEMM (eliminates Marlin's 20-wave reload bottleneck on 16K tokens)
"""

from typing import TypedDict
import types
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import (
    SMALL_NUMEL_LIMIT,
    moe_align_small_numel,
)
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    attn_residual_gate: torch.Tensor
    pre_ffn_norm: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    pooler: torch.Tensor
    pre_expert_norm: torch.Tensor
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_expert_norm: torch.Tensor
    unpooler: torch.Tensor
    post_ffn_norm: torch.Tensor
    ffn_residual_gate: torch.Tensor
    next_pre_attn_norm: torch.Tensor


class Config(TypedDict):
    batch: int
    q_len: int
    tokens: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float
    n_experts: int
    top_k: int
    local_experts: int


class Inputs(TypedDict):
    stream: torch.Tensor
    attn_input: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    local_start: torch.Tensor
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
ROUTER_FP32_MAX_TOKENS = 8
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
MAX_TOKENS = 16384
MAX_EXP_TOKENS = 1536  # Safe maximum tokens per expert for 16K prefill


def _cos_sin(theta, d, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _gemm(x, w_t):
    return _bf16_gemm_dispatch_impl(x, w_t)


def _marlin_bank(weights, device):
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map,
        "size_n": layer.intermediate_size_per_partition,
    }


@triton.jit
def _dequant_nvfp4_kernel(
    packed_ptr,
    sf_ptr,
    gs_ptr,
    out_ptr,
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_en = tl.program_id(0)
    pid_k = tl.program_id(1)

    pid_e = pid_en // (N // BLOCK_N)
    pid_n = pid_en % (N // BLOCK_N)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    offs_packed = pid_e * (N * (K // 2)) + offs_n[:, None] * (K // 2) + (offs_k[None, :] // 2)
    bytes_val = tl.load(packed_ptr + offs_packed)

    is_hi = (offs_k[None, :] % 2) == 1
    codes = tl.where(is_hi, (bytes_val >> 4) & 0x0F, bytes_val & 0x0F)

    sign = (codes & 0x08) != 0
    mag_code = codes & 0x07

    mag = tl.where(mag_code < 5, mag_code.to(tl.float32) * 0.5,
          tl.where(mag_code == 5, 3.0,
          tl.where(mag_code == 6, 4.0, 6.0)))

    val = tl.where(sign, -mag, mag)

    offs_sf = pid_e * (N * (K // 16)) + offs_n[:, None] * (K // 16) + (offs_k[None, :] // 16)
    sf_val = tl.load(sf_ptr + offs_sf).to(tl.float32)
    gs = tl.load(gs_ptr + pid_e)

    res = val * sf_val * gs
    # Write to out: [E, K, N] for optimal row-major torch.bmm (X @ W)
    offs_out = pid_e * (K * N) + offs_k[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + offs_out, res.to(out_ptr.dtype.element_ty))


@triton.jit
def _fused_prepare_kv_kernel(
    qkv_ptr,
    positions_ptr,
    cos_sin_ptr,
    cache_loc_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_val,
    v_scale_val,
    has_rope: tl.constexpr,
    scaled: tl.constexpr,
    stride_qkv_tok,
    stride_q_tok,
    stride_k_cache_page,
    stride_k_cache_ps,
    stride_k_cache_head,
    stride_v_cache_page,
    stride_v_cache_ps,
    stride_v_cache_head,
    page_size,
    HEAD_DIM: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    eps: tl.constexpr = 1e-5

    offs_d = tl.arange(0, HEAD_DIM)
    offs_d2 = tl.arange(0, HEAD_DIM // 2)

    if head_idx < num_q_heads:
        q_offset = tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + offs_d
        q_vals = tl.load(qkv_ptr + q_offset).to(tl.float32)
        rms = tl.sqrt(tl.sum(q_vals * q_vals) / HEAD_DIM + eps)

        if has_rope:
            pos = tl.load(positions_ptr + tok_idx)
            cos_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + offs_d2)
            sin_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + HEAD_DIM // 2 + offs_d2)

            q_low = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + offs_d2).to(tl.float32)
            q_high = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + HEAD_DIM // 2 + offs_d2).to(tl.float32)

            q_rot_low = q_low * cos_val - q_high * sin_val
            q_rot_high = q_low * sin_val + q_high * cos_val

            scale = 5.473 / rms
            tl.store(q_out_ptr + tok_idx * stride_q_tok + head_idx * HEAD_DIM + offs_d2, (q_rot_low * scale).to(tl.float8e4m3fn))
            tl.store(q_out_ptr + tok_idx * stride_q_tok + head_idx * HEAD_DIM + HEAD_DIM // 2 + offs_d2, (q_rot_high * scale).to(tl.float8e4m3fn))
        else:
            scale = 5.473 / rms
            tl.store(q_out_ptr + tok_idx * stride_q_tok + head_idx * HEAD_DIM + offs_d, (q_vals * scale).to(tl.float8e4m3fn))

    else:
        kv_head_idx = head_idx - num_q_heads
        k_offset = tok_idx * stride_qkv_tok + (num_q_heads + kv_head_idx) * HEAD_DIM + offs_d
        v_offset = tok_idx * stride_qkv_tok + (num_q_heads + num_kv_heads + kv_head_idx) * HEAD_DIM + offs_d

        k_vals = tl.load(qkv_ptr + k_offset).to(tl.float32)
        v_vals = tl.load(qkv_ptr + v_offset).to(tl.float32)

        rms_k = tl.sqrt(tl.sum(k_vals * k_vals) / HEAD_DIM + eps)

        if has_rope:
            pos = tl.load(positions_ptr + tok_idx)
            cos_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + offs_d2)
            sin_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + HEAD_DIM // 2 + offs_d2)

            k_low = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + (num_q_heads + kv_head_idx) * HEAD_DIM + offs_d2).to(tl.float32)
            k_high = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + (num_q_heads + kv_head_idx) * HEAD_DIM + HEAD_DIM // 2 + offs_d2).to(tl.float32)

            k_rot_low = k_low * cos_val - k_high * sin_val
            k_rot_high = k_low * sin_val + k_high * cos_val

            norm_k_low = k_rot_low / rms_k
            norm_k_high = k_rot_high / rms_k

            if scaled:
                norm_k_low = norm_k_low / k_scale_val
                norm_k_high = norm_k_high / k_scale_val
                v_vals = v_vals / v_scale_val

            k_fp8_low = norm_k_low.to(tl.float8e4m3fn)
            k_fp8_high = norm_k_high.to(tl.float8e4m3fn)
            v_fp8 = v_vals.to(tl.float8e4m3fn)

            cache_row = tl.load(cache_loc_ptr + tok_idx)
            page_idx = cache_row // page_size
            page_offset = cache_row % page_size

            k_dst_low = (
                k_cache_ptr
                + page_idx * stride_k_cache_page
                + page_offset * stride_k_cache_ps
                + kv_head_idx * stride_k_cache_head
                + offs_d2
            )
            k_dst_high = (
                k_cache_ptr
                + page_idx * stride_k_cache_page
                + page_offset * stride_k_cache_ps
                + kv_head_idx * stride_k_cache_head
                + HEAD_DIM // 2 + offs_d2
            )
            v_dst = (
                v_cache_ptr
                + page_idx * stride_v_cache_page
                + page_offset * stride_v_cache_ps
                + kv_head_idx * stride_v_cache_head
                + offs_d
            )

            tl.store(k_dst_low, k_fp8_low)
            tl.store(k_dst_high, k_fp8_high)
            tl.store(v_dst, v_fp8)
        else:
            norm_k = k_vals / rms_k
            if scaled:
                norm_k = norm_k / k_scale_val
                v_vals = v_vals / v_scale_val

            k_fp8 = norm_k.to(tl.float8e4m3fn)
            v_fp8 = v_vals.to(tl.float8e4m3fn)

            cache_row = tl.load(cache_loc_ptr + tok_idx)
            page_idx = cache_row // page_size
            page_offset = cache_row % page_size

            k_dst = (
                k_cache_ptr
                + page_idx * stride_k_cache_page
                + page_offset * stride_k_cache_ps
                + kv_head_idx * stride_k_cache_head
                + offs_d
            )
            v_dst = (
                v_cache_ptr
                + page_idx * stride_v_cache_page
                + page_offset * stride_v_cache_ps
                + kv_head_idx * stride_v_cache_head
                + offs_d
            )

            tl.store(k_dst, k_fp8)
            tl.store(v_dst, v_fp8)


@triton.jit
def _fast_pre_expert_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x,
    stride_out,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=0) / H + eps)
    normed = (x / rms) * (1.0 + w)
    tl.store(out_ptr + pid * stride_out + offs, normed.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_swiglu_kernel(
    in_ptr,   # [E, M, 2 * I]
    out_ptr,  # [E, M, I]
    total_elements,
    I: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Map flat offset to (row, col)
    row = offs // I
    col = offs % I

    gate = tl.load(in_ptr + row * (2 * I) + col, mask=mask).to(tl.float32)
    up = tl.load(in_ptr + row * (2 * I) + I + col, mask=mask).to(tl.float32)

    # silu(gate) * up = gate * sigmoid(gate) * up
    silu_gate = gate / (1.0 + tl.exp(-gate))
    res = silu_gate * up
    tl.store(out_ptr + row * I + col, res.to(out_ptr.dtype.element_ty), mask=mask)


def prepare(weights):
    device = weights["qkv_gate"].device
    dense = {k: v.contiguous() for k, v in weights.items() if v.ndim == 2}
    router = weights["router"]
    r_lo = (router - router.to(torch.bfloat16)).to(torch.bfloat16)
    r_hi = (router - r_lo).to(torch.bfloat16)

    freqs = _cos_sin(500000, 64, device)
    bank = _marlin_bank(weights, device)

    max_padded = MAX_TOKENS * TOP_K + (N_EXPERTS + 1) * 64
    align_sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    align_expert_ids = torch.empty((max_padded // 8,), dtype=torch.int32, device=device)
    align_n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    align_cumsum = torch.empty((N_EXPERTS + 2,), dtype=torch.int32, device=device)

    size_n = bank["size_n"]
    size_k = D_EXPERT
    gemm1_n = 2 * size_n

    cache2_buf = torch.empty((MAX_TOKENS * TOP_K, size_n), device=device, dtype=torch.bfloat16)
    cache13_buf = torch.empty((MAX_TOKENS * TOP_K * max(gemm1_n, size_k),), device=device, dtype=torch.bfloat16)
    q_out_buf = torch.empty((MAX_TOKENS, 16, 64), dtype=FP8, device=device)
    expert_out_buf = torch.empty((MAX_TOKENS, size_k), dtype=torch.bfloat16, device=device)
    pre_expert_buf = torch.empty((MAX_TOKENS, size_k), dtype=torch.bfloat16, device=device)

    # Preallocated buffers for Batched GEMM path on 16K prefill
    w13_bf16 = torch.empty((LOCAL_EXPERTS, D_EXPERT, 2 * D_EXPERT), dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.empty((LOCAL_EXPERTS, D_EXPERT, D_EXPERT), dtype=torch.bfloat16, device=device)

    workspace = marlin_make_workspace(device, max_blocks_per_sm=32)
    bank["workspace"] = workspace

    state = {
        "dense": dense,
        "cos_sin": freqs,
        "router_hi_t": r_hi.t().contiguous(),
        "router_lo_t": r_lo.t().contiguous(),
        "router": weights["router"],
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "o_norm_gate": FusedRMSNormGate(64, eps=RMS_EPS),
        "attn_boundary": FusedPostNormResidualPreNorm(weights["attn_residual_gate"], weights["pre_ffn_norm"]),
        "ffn_boundary": FusedPostNormResidualPreNorm(weights["ffn_residual_gate"], weights["post_ffn_norm"], weights["next_pre_attn_norm"]),
        "post_expert_w": (weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16),
        "pre_expert_w": weights["pre_expert_norm"],
        "bank": bank,
        "side_stream": torch.cuda.Stream(device),
        "align_sorted_ids": align_sorted_ids,
        "align_expert_ids": align_expert_ids,
        "align_n_padded": align_n_padded,
        "align_cumsum": align_cumsum,
        "cache2_buf": cache2_buf,
        "cache13_buf": cache13_buf,
        "q_out_buf": q_out_buf,
        "expert_out_buf": expert_out_buf,
        "pre_expert_buf": pre_expert_buf,
        "w13_raw": weights["w13"],
        "w13_sf": weights["w13_sf"],
        "w13_gs": weights["w13_gs"],
        "w2_raw": weights["w2"],
        "w2_sf": weights["w2_sf"],
        "w2_gs": weights["w2_gs"],
        "w13_bf16": w13_bf16,
        "w2_bf16": w2_bf16,
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _optimized_experts(x, topk_weights, topk_ids, state):
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break

    numel = local_ids.numel()
    num_exp = N_EXPERTS
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

    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]

    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=total_pairs, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)

    out = state["expert_out_buf"][:num_tokens]
    block_h = triton.next_power_of_2(size_k)
    _post_expert_norm_weighted_sum_kernel[(num_tokens,)](
        y,
        state["post_expert_w"],
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


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]

    # 1. QKV projection
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])

    # 2. Fused QK-norm + RoPE + FP8 quantize + KV-cache write
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k_scale_val = cfg["kv_scale"]
    v_scale_val = cfg["kv_scale"]

    q_out = state["q_out_buf"][:tokens]
    grid = (tokens, hq + hkv)
    _fused_prepare_kv_kernel[grid](
        qkv,
        data["positions"],
        state["cos_sin"],
        data["cache_loc"],
        q_out,
        k_cache,
        v_cache,
        k_scale_val,
        v_scale_val,
        has_rope=bool(cfg["rope_theta"]),
        scaled=scaled,
        stride_qkv_tok=qkv.stride(0),
        stride_q_tok=q_out.stride(0),
        stride_k_cache_page=k_cache.stride(0),
        stride_k_cache_ps=k_cache.stride(1),
        stride_k_cache_head=k_cache.stride(2),
        stride_v_cache_page=v_cache.stride(0),
        stride_v_cache_ps=v_cache.stride(1),
        stride_v_cache_head=v_cache.stride(2),
        page_size=cfg["page_size"],
        HEAD_DIM=64,
        num_q_heads=hq,
        num_kv_heads=hkv,
    )

    gate = qkv[:, hq * d + 2 * hkv * d:]
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))

    # 3. FlashAttention-3
    out = flash_attn_with_kvcache(
        q=q_out,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]

    # 4. Output norm & gate + O projection
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemm(o, dense["o"])

    # 5. Attention residual boundary + Router
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)

    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            assert not tensor.isnan().any()

    # 6. Fast Triton pre_expert_norm
    normed_expert_input = state["pre_expert_buf"][:tokens]
    _fast_pre_expert_norm_kernel[(tokens,)](
        expert_input,
        state["pre_expert_w"],
        normed_expert_input,
        expert_input.stride(0),
        normed_expert_input.stride(0),
        H=D_EXPERT,
        eps=RMS_EPS,
        BLOCK_H=D_EXPERT,
    )

    # 7. MoE experts
    y = _optimized_experts(normed_expert_input, topk_weights, topk_ids, state)

    # 8. Unpooler
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        f = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device=y.device)
        f[:, :UNPOOL_SLICE] = _gemm(y, dense["unpooler"][:UNPOOL_SLICE])
    else:
        f = _gemm(y, dense["unpooler"])

    # 9. FFN residual boundary
    next_attn_input, stream, _ = state["ffn_boundary"](stream, f)
    return next_attn_input, stream, topk_ids
