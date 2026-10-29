# okc: problem=muse_prefill_layer
"""Prefill layer with hybrid MoE:
- tokens < 4000: Marlin MoE
- tokens >= 4000: Fast NVFP4 Dequantize + cuBLAS Batched GEMM (eliminates Marlin 20-wave reload bottleneck)
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
from torch import nn

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
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
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
ROPE_THETA = 500_000.0
MAX_TOKENS = 16384
M_MAX = 1024


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


@triton.jit
def _fast_pre_expert_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x_tok,
    stride_out_tok,
    N: tl.constexpr,
    eps: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + tok_idx * stride_x_tok + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    y = (x / rms) * (1.0 + w)
    tl.store(out_ptr + tok_idx * stride_out_tok + offs, y.to(out_ptr.dtype.element_ty))


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

            idx_even = offs_d2 * 2
            idx_odd = offs_d2 * 2 + 1

            q_even = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + idx_even).to(tl.float32) / rms
            q_odd = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + idx_odd).to(tl.float32) / rms

            rot_even = q_even * cos_val - q_odd * sin_val
            rot_odd = q_even * sin_val + q_odd * cos_val

            out_offset = tok_idx * stride_q_tok + head_idx * HEAD_DIM
            tl.store(q_out_ptr + out_offset + idx_even, rot_even.to(q_out_ptr.dtype.element_ty))
            tl.store(q_out_ptr + out_offset + idx_odd, rot_odd.to(q_out_ptr.dtype.element_ty))
        else:
            q_norm = q_vals / rms
            out_offset = tok_idx * stride_q_tok + head_idx * HEAD_DIM + offs_d
            tl.store(q_out_ptr + out_offset, q_norm.to(q_out_ptr.dtype.element_ty))

    else:
        kv_h = head_idx - num_q_heads
        loc = tl.load(cache_loc_ptr + tok_idx)
        page_idx = loc // page_size
        offset_in_page = loc % page_size

        k_start = (num_q_heads + kv_h) * HEAD_DIM
        v_start = (num_q_heads + num_kv_heads + kv_h) * HEAD_DIM

        k_offset = tok_idx * stride_qkv_tok + k_start + offs_d
        k_vals = tl.load(qkv_ptr + k_offset).to(tl.float32)
        rms = tl.sqrt(tl.sum(k_vals * k_vals) / HEAD_DIM + eps)

        if has_rope:
            pos = tl.load(positions_ptr + tok_idx)
            cos_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + offs_d2)
            sin_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + HEAD_DIM // 2 + offs_d2)

            idx_even = offs_d2 * 2
            idx_odd = offs_d2 * 2 + 1

            k_even = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + k_start + idx_even).to(tl.float32) / rms
            k_odd = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + k_start + idx_odd).to(tl.float32) / rms

            rot_k_even = k_even * cos_val - k_odd * sin_val
            rot_k_odd = k_even * sin_val + k_odd * cos_val

            if scaled:
                rot_k_even = rot_k_even / k_scale_val
                rot_k_odd = rot_k_odd / k_scale_val

            k_dst_offset = page_idx * stride_k_cache_page + offset_in_page * stride_k_cache_ps + kv_h * stride_k_cache_head
            tl.store(k_cache_ptr + k_dst_offset + idx_even, rot_k_even.to(k_cache_ptr.dtype.element_ty))
            tl.store(k_cache_ptr + k_dst_offset + idx_odd, rot_k_odd.to(k_cache_ptr.dtype.element_ty))
        else:
            k_norm = k_vals / rms
            if scaled:
                k_norm = k_norm / k_scale_val
            k_dst_offset = page_idx * stride_k_cache_page + offset_in_page * stride_k_cache_ps + kv_h * stride_k_cache_head + offs_d
            tl.store(k_cache_ptr + k_dst_offset, k_norm.to(k_cache_ptr.dtype.element_ty))

        v_offset = tok_idx * stride_qkv_tok + v_start + offs_d
        v_vals = tl.load(qkv_ptr + v_offset).to(tl.float32)
        if scaled:
            v_vals = v_vals / v_scale_val
        v_dst_offset = page_idx * stride_v_cache_page + offset_in_page * stride_v_cache_ps + kv_h * stride_v_cache_head + offs_d
        tl.store(v_cache_ptr + v_dst_offset, v_vals.to(v_cache_ptr.dtype.element_ty))


@triton.jit
def _dequant_weights_kernel(
    packed_ptr,  # [E, N, K // 2]
    sf_ptr,      # [E, N, K // 16]
    gs_ptr,      # [E]
    out_ptr,     # [E, K, N]
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_en = tl.program_id(0)
    pid_k = tl.program_id(1)

    pid_e = pid_en // (N // BLOCK_N)
    pid_n = pid_en % (N // BLOCK_N)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    offs_packed = pid_e * (N * (K // 2)) + offs_n * (K // 2) + (offs_k // 2)
    bytes_val = tl.load(packed_ptr + offs_packed)

    is_hi = (offs_k % 2) == 1
    codes = tl.where(is_hi, (bytes_val >> 4) & 0x0F, bytes_val & 0x0F)

    sign = (codes & 0x08) != 0
    mag_code = codes & 0x07

    mag = tl.where(mag_code < 5, mag_code.to(tl.float32) * 0.5,
          tl.where(mag_code == 5, 3.0,
          tl.where(mag_code == 6, 4.0, 6.0)))

    val = tl.where(sign, -mag, mag)

    offs_sf = pid_e * (N * (K // 16)) + offs_n * (K // 16) + (offs_k // 16)
    sf_val = tl.load(sf_ptr + offs_sf).to(tl.float32)
    gs = tl.load(gs_ptr + pid_e)

    res = val * sf_val * gs

    offs_out = pid_e * (K * N) + offs_k * N + offs_n
    tl.store(out_ptr + offs_out, res.to(out_ptr.dtype.element_ty))


@triton.jit
def _count_and_map_kernel(
    topk_ids_ptr,      # [total_pairs]
    map_tok_ptr,       # [32 * M_MAX]
    map_slot_ptr,      # [32 * M_MAX]
    scatter_idx_ptr,   # [32 * M_MAX]
    counts_ptr,        # [32]
    total_pairs,
    M_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_pairs

    exp = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    tok = offs // 8
    slot = offs % 8

    is_local = (exp >= 0) & (exp < 32) & mask

    pos = tl.atomic_add(counts_ptr + exp, 1, mask=is_local)

    valid_store = is_local & (pos < M_MAX)
    dest = exp * M_MAX + pos

    tl.store(map_tok_ptr + dest, tok, mask=valid_store)
    tl.store(map_slot_ptr + dest, slot, mask=valid_store)
    tl.store(scatter_idx_ptr + dest, tok, mask=valid_store)


@triton.jit
def _copy_to_batch_kernel(
    x_ptr,            # [tokens + 1, 4096] (row tokens is all zeros)
    scatter_idx_ptr,  # [32 * M_MAX]
    x_batch_ptr,      # [32 * M_MAX, 4096]
    total_rows: tl.constexpr,
    D: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= total_rows:
        return

    tok = tl.load(scatter_idx_ptr + row_idx)
    src_off = tok * D
    dst_off = row_idx * D

    for chunk in tl.static_range(4):
        k_offs = chunk * 1024 + tl.arange(0, 1024)
        vals = tl.load(x_ptr + src_off + k_offs)
        tl.store(x_batch_ptr + dst_off + k_offs, vals)



@triton.jit
def _fused_swiglu_kernel(
    in_ptr,   # [32, M_MAX, 8192]
    out_ptr,  # [32, M_MAX, 4096]
    total_elements,
    I: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    row = offs // I
    col = offs % I

    gate = tl.load(in_ptr + row * (2 * I) + col, mask=mask).to(tl.float32)
    up = tl.load(in_ptr + row * (2 * I) + I + col, mask=mask).to(tl.float32)

    silu_gate = gate / (1.0 + tl.exp(-gate))
    res = silu_gate * up
    tl.store(out_ptr + row * I + col, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gather_post_norm_kernel(
    y_ptr,               # [32, M_MAX, 4096]
    post_w_ptr,          # [4096] (already 1.0 + w in FP32)
    topk_weights_ptr,    # [tokens, 8]
    map_tok_ptr,         # [32, M_MAX]
    map_slot_ptr,        # [32, M_MAX]
    counts_ptr,          # [32]
    out_ptr,             # [tokens, 4096] float32
    M_MAX: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    e = pid // M_MAX
    pos = pid % M_MAX

    count = tl.load(counts_ptr + e)
    if pos < count:
        tok = tl.load(map_tok_ptr + e * M_MAX + pos)
        slot = tl.load(map_slot_ptr + e * M_MAX + pos)
        wt = tl.load(topk_weights_ptr + tok * 8 + slot).to(tl.float32)

        offs = tl.arange(0, 1024)
        base_y = (e * M_MAX + pos) * D
        v0 = tl.load(y_ptr + base_y + offs).to(tl.float32) * tl.load(post_w_ptr + offs)
        v1 = tl.load(y_ptr + base_y + 1024 + offs).to(tl.float32) * tl.load(post_w_ptr + 1024 + offs)
        v2 = tl.load(y_ptr + base_y + 2048 + offs).to(tl.float32) * tl.load(post_w_ptr + 2048 + offs)
        v3 = tl.load(y_ptr + base_y + 3072 + offs).to(tl.float32) * tl.load(post_w_ptr + 3072 + offs)

        sum_sq = tl.sum(v0 * v0) + tl.sum(v1 * v1) + tl.sum(v2 * v2) + tl.sum(v3 * v3)
        rms = tl.sqrt(sum_sq / 4096.0 + eps)
        inv_rms_wt = wt / rms

        base_out = tok * D
        tl.atomic_add(out_ptr + base_out + offs, v0 * inv_rms_wt)
        tl.atomic_add(out_ptr + base_out + 1024 + offs, v1 * inv_rms_wt)
        tl.atomic_add(out_ptr + base_out + 2048 + offs, v2 * inv_rms_wt)
        tl.atomic_add(out_ptr + base_out + 3072 + offs, v3 * inv_rms_wt)


def _marlin_bank(weights):
    device = weights["w13"].device
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
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=32),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)

    # Marlin buffers
    size_n = weights["w13"].shape[1] // 2
    size_k = D_EXPERT
    gemm1_n = 2 * size_n
    cache2_buf = torch.empty((MAX_TOKENS * TOP_K, size_n), device=device, dtype=torch.bfloat16)
    cache13_buf = torch.empty((MAX_TOKENS * TOP_K * max(gemm1_n, size_k),), device=device, dtype=torch.bfloat16)

    max_padded = 262144
    align_sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    align_expert_ids = torch.empty((max_padded // 8,), dtype=torch.int32, device=device)
    align_n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    align_cumsum = torch.empty((N_EXPERTS + 2,), dtype=torch.int32, device=device)

    q_out_buf = torch.empty((MAX_TOKENS, 16, 64), dtype=FP8, device=device)
    pre_expert_norm_buf = torch.zeros((MAX_TOKENS + 1, D_EXPERT), dtype=torch.bfloat16, device=device)
    expert_out_buf = torch.empty((MAX_TOKENS, D_EXPERT), dtype=torch.bfloat16, device=device)

    # BMM MoE buffers
    x_batch = torch.empty((LOCAL_EXPERTS, M_MAX, D_EXPERT), dtype=torch.bfloat16, device=device)
    w13_bf16 = torch.empty((LOCAL_EXPERTS, D_EXPERT, 2 * D_EXPERT), dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.empty((LOCAL_EXPERTS, D_EXPERT, D_EXPERT), dtype=torch.bfloat16, device=device)
    y13_buf = torch.empty((LOCAL_EXPERTS, M_MAX, 2 * D_EXPERT), dtype=torch.bfloat16, device=device)
    swiglu_buf = torch.empty((LOCAL_EXPERTS, M_MAX, D_EXPERT), dtype=torch.bfloat16, device=device)
    y2_buf = torch.empty((LOCAL_EXPERTS, M_MAX, D_EXPERT), dtype=torch.bfloat16, device=device)
    map_tok = torch.empty((LOCAL_EXPERTS, M_MAX), dtype=torch.int32, device=device)
    map_slot = torch.empty((LOCAL_EXPERTS, M_MAX), dtype=torch.int32, device=device)
    scatter_idx = torch.empty((LOCAL_EXPERTS * M_MAX,), dtype=torch.int32, device=device)
    counts = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=device)
    out_accum = torch.empty((MAX_TOKENS, D_EXPERT), dtype=torch.float32, device=device)
    post_expert_w_fp32 = post_expert._get_effective_weight().detach().float().contiguous()

    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous(),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_w": weights["pre_expert_norm"].contiguous(),
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "post_expert_w_fp32": post_expert_w_fp32,
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
        "cache2_buf": cache2_buf,
        "cache13_buf": cache13_buf,
        "align_sorted_ids": align_sorted_ids,
        "align_expert_ids": align_expert_ids,
        "align_n_padded": align_n_padded,
        "align_cumsum": align_cumsum,
        "q_out_buf": q_out_buf,
        "pre_expert_norm_buf": pre_expert_norm_buf,
        "expert_out_buf": expert_out_buf,
        "w13_raw": weights["w13"],
        "w13_sf": weights["w13_sf"],
        "w13_gs": weights["w13_gs"],
        "w2_raw": weights["w2"],
        "w2_sf": weights["w2_sf"],
        "w2_gs": weights["w2_gs"],
        "m_max": M_MAX,
        "x_batch": x_batch,
        "w13_bf16": w13_bf16,
        "w2_bf16": w2_bf16,
        "y13_buf": y13_buf,
        "swiglu_buf": swiglu_buf,
        "y2_buf": y2_buf,
        "map_tok": map_tok,
        "map_slot": map_slot,
        "scatter_idx": scatter_idx,
        "counts": counts,
        "out_accum": out_accum,
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _bmm_experts(x, topk_weights, topk_ids, state, num_tokens):
    D = D_EXPERT
    m_max = state["m_max"]
    total_pairs = num_tokens * TOP_K

    # 1. Zero state buffers & fill scatter_idx with dummy index (num_tokens)
    state["counts"].zero_()
    state["scatter_idx"].fill_(num_tokens)
    state["out_accum"][:num_tokens].zero_()

    # 2. Dequantize weights
    _dequant_weights_kernel[(LOCAL_EXPERTS * (2 * D // 64), D // 64)](
        state["w13_raw"], state["w13_sf"], state["w13_gs"], state["w13_bf16"],
        LOCAL_EXPERTS, 2 * D, D, 64, 64, num_warps=4)
    _dequant_weights_kernel[(LOCAL_EXPERTS * (D // 64), D // 64)](
        state["w2_raw"], state["w2_sf"], state["w2_gs"], state["w2_bf16"],
        LOCAL_EXPERTS, D, D, 64, 64, num_warps=4)

    # 3. Vectorized count and map
    grid_cm = (triton.cdiv(total_pairs, 128),)
    _count_and_map_kernel[grid_cm](
        topk_ids.view(-1), state["map_tok"], state["map_slot"], state["scatter_idx"], state["counts"],
        total_pairs, M_MAX=m_max, BLOCK_SIZE=128, num_warps=4)

    # 4. Copy rows to batch
    total_rows = LOCAL_EXPERTS * m_max
    _copy_to_batch_kernel[(total_rows,)](
        x, state["scatter_idx"], state["x_batch"],
        total_rows=total_rows, D=D, num_warps=4)

    # 5. First GEMM: Y13 = X @ W13
    torch.bmm(state["x_batch"], state["w13_bf16"], out=state["y13_buf"])

    # 6. Fused SwiGLU
    total_swiglu = LOCAL_EXPERTS * m_max * D
    _fused_swiglu_kernel[(triton.cdiv(total_swiglu, 1024),)](
        state["y13_buf"], state["swiglu_buf"], total_swiglu, I=D, BLOCK_SIZE=1024, num_warps=4)

    # 7. Second GEMM: Y2 = SwiGLU @ W2
    torch.bmm(state["swiglu_buf"], state["w2_bf16"], out=state["y2_buf"])

    # 8. Gather & Post-Expert Norm
    _gather_post_norm_kernel[(LOCAL_EXPERTS * m_max,)](
        state["y2_buf"], state["post_expert_w_fp32"], topk_weights,
        state["map_tok"], state["map_slot"], state["counts"],
        state["out_accum"], M_MAX=m_max, D=D, eps=POST_NORM_EPS, num_warps=4)

    # 9. Cast out to bfloat16
    return state["out_accum"][:num_tokens].to(torch.bfloat16)


def _marlin_experts(x, topk_weights, topk_ids, state, num_tokens):
    bank = state["bank"]
    size_k = D_EXPERT
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


def _optimized_experts(x, topk_weights, topk_ids, state):
    num_tokens = x.shape[0]
    if num_tokens >= 4000:
        return _bmm_experts(x, topk_weights, topk_ids, state, num_tokens)
    else:
        return _marlin_experts(x, topk_weights, topk_ids, state, num_tokens)


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
            tensor.record_stream(main_stream)

    # 6. Fast Triton pre-expert norm + Optimized MoE experts
    normed_expert_in = state["pre_expert_norm_buf"][:tokens]
    _fast_pre_expert_norm_kernel[(tokens,)](
        expert_input,
        state["pre_expert_w"],
        normed_expert_in,
        stride_x_tok=expert_input.stride(0),
        stride_out_tok=normed_expert_in.stride(0),
        N=D_EXPERT,
        eps=RMS_EPS,
        num_warps=8,
        num_stages=2,
    )

    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    expert_output = _optimized_experts(normed_expert_in, topk_weights, local_ids, state)

    # 7. Unpooler
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # 8. Post-FFN boundary
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
