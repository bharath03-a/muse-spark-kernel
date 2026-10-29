# okc: problem=muse_prefill_layer
"""Optimized prefill layer for Muse Spark Open on H200 (TP8/EP8):
- Fused attention prepare kernel
- Large Marlin MoE workspace (max_blocks_per_sm=32) avoiding serialization across waves
- Fast Triton pre_expert_norm kernel (replaces slow PyTorch vectorized_layer_norm)
- Preallocated expert output buffer (eliminates 134 MB allocation per step)
"""

from typing import TypedDict
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import (
    _jit_moe_wna16_marlin_module,
    moe_wna16_marlin_gemm,
)
try:
    from sglang.kernels.ops.moe.moe_wna16_marlin import _MAX_THREAD_N
except ImportError:
    _MAX_THREAD_N = 128
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
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
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


def _fast_mm(x, w_t, out_buf):
    return torch.mm(x, w_t, out=out_buf[:x.shape[0]])


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
        if has_rope:
            pos = tl.load(positions_ptr + tok_idx)
            cos_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + offs_d2)
            sin_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + HEAD_DIM // 2 + offs_d2)

            idx_even = offs_d2 * 2
            idx_odd = offs_d2 * 2 + 1

            q_even = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + idx_even).to(tl.float32)
            q_odd = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + idx_odd).to(tl.float32)

            rms = tl.sqrt((tl.sum(q_even * q_even) + tl.sum(q_odd * q_odd)) / HEAD_DIM + eps)
            inv_rms = 1.0 / rms
            q_even = q_even * inv_rms
            q_odd = q_odd * inv_rms

            rot_even = q_even * cos_val - q_odd * sin_val
            rot_odd = q_even * sin_val + q_odd * cos_val

            out_offset = tok_idx * stride_q_tok + head_idx * HEAD_DIM
            tl.store(q_out_ptr + out_offset + idx_even, rot_even.to(q_out_ptr.dtype.element_ty))
            tl.store(q_out_ptr + out_offset + idx_odd, rot_odd.to(q_out_ptr.dtype.element_ty))
        else:
            q_offset = tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + offs_d
            q_vals = tl.load(qkv_ptr + q_offset).to(tl.float32)
            rms = tl.sqrt(tl.sum(q_vals * q_vals) / HEAD_DIM + eps)
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

        if has_rope:
            pos = tl.load(positions_ptr + tok_idx)
            cos_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + offs_d2)
            sin_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + HEAD_DIM // 2 + offs_d2)

            idx_even = offs_d2 * 2
            idx_odd = offs_d2 * 2 + 1

            k_even = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + k_start + idx_even).to(tl.float32)
            k_odd = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + k_start + idx_odd).to(tl.float32)

            rms = tl.sqrt((tl.sum(k_even * k_even) + tl.sum(k_odd * k_odd)) / HEAD_DIM + eps)
            inv_rms = 1.0 / rms
            k_even = k_even * inv_rms
            k_odd = k_odd * inv_rms

            rot_k_even = k_even * cos_val - k_odd * sin_val
            rot_k_odd = k_even * sin_val + k_odd * cos_val

            if scaled:
                rot_k_even = rot_k_even / k_scale_val
                rot_k_odd = rot_k_odd / k_scale_val

            k_dst_offset = page_idx * stride_k_cache_page + offset_in_page * stride_k_cache_ps + kv_h * stride_k_cache_head
            tl.store(k_cache_ptr + k_dst_offset + idx_even, rot_k_even.to(k_cache_ptr.dtype.element_ty))
            tl.store(k_cache_ptr + k_dst_offset + idx_odd, rot_k_odd.to(k_cache_ptr.dtype.element_ty))
        else:
            k_offset = tok_idx * stride_qkv_tok + k_start + offs_d
            k_vals = tl.load(qkv_ptr + k_offset).to(tl.float32)
            rms = tl.sqrt(tl.sum(k_vals * k_vals) / HEAD_DIM + eps)
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
    w13_num_groups = layer.w13_weight_scale.size(1)
    w13_group_size = 4096 // w13_num_groups if w13_num_groups > 1 else -1
    w2_num_groups = layer.w2_weight_scale.size(1)
    w2_group_size = layer.intermediate_size_per_partition // w2_num_groups if w2_num_groups > 1 else -1
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w13_num_groups": w13_num_groups, "w13_group_size": w13_group_size,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "w2_num_groups": w2_num_groups, "w2_group_size": w2_group_size,
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
    pre_expert = _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS)

    # Preallocated MoE scratch buffers for up to MAX_TOKENS (16384)
    size_n = weights["w13"].shape[1] // 2
    size_k = D_EXPERT
    gemm1_n = 2 * size_n
    cache2_buf = torch.empty((MAX_TOKENS * TOP_K, size_n), device=device, dtype=torch.bfloat16)
    cache13_buf = torch.empty((MAX_TOKENS * TOP_K * max(gemm1_n, size_k),), device=device, dtype=torch.bfloat16)

    # Alignment preallocated buffers
    max_padded = 262144
    align_sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    align_expert_ids = torch.empty((max_padded // 8,), dtype=torch.int32, device=device)
    align_n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    align_cumsum = torch.empty((N_EXPERTS + 2,), dtype=torch.int32, device=device)

    # Preallocated Q_out buffer for attention
    q_out_buf = torch.empty((MAX_TOKENS, 16, 64), dtype=FP8, device=device)
    attn_out_bf16_buf = torch.empty((MAX_TOKENS, 16, 64), dtype=torch.bfloat16, device=device)
    pre_expert_norm_buf = torch.empty((MAX_TOKENS, D_EXPERT), dtype=torch.bfloat16, device=device)
    expert_out_buf = torch.empty((MAX_TOKENS, D_EXPERT), dtype=torch.bfloat16, device=device)

    # Preallocated GEMM buffers
    qkv_buf = torch.empty((MAX_TOKENS, 2304), dtype=torch.bfloat16, device=device)
    attn_out_buf = torch.empty((MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    pooler_buf = torch.empty((MAX_TOKENS, D_EXPERT), dtype=torch.bfloat16, device=device)
    unpooler_buf = torch.empty((MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    unpooler_slice_buf = torch.empty((MAX_TOKENS, UNPOOL_SLICE), dtype=torch.bfloat16, device=device)
    padded_unpool_buf = torch.zeros((MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)

    dense = {
        "qkv_gate_t": weights["qkv_gate"].t().contiguous(),
        "o_t": weights["o"].t().contiguous(),
        "pooler_t": weights["pooler"].t().contiguous(),
        "unpooler_t": weights["unpooler"].t().contiguous(),
        "unpooler_slice_t": weights["unpooler"][:UNPOOL_SLICE].t().contiguous(),
    }

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    marlin_mod_ep = _jit_moe_wna16_marlin_module(torch.bfloat16, True, False)
    empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=device)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    c_tmp_buf = torch.empty(16 * 1024 * 1024, dtype=torch.float32, device=device)

    state = {
        "dense": dense,
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
        "pre_expert_norm": pre_expert,
        "pre_expert_w": weights["pre_expert_norm"].contiguous(),
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert, pre_expert),
        "cache2_buf": cache2_buf,
        "cache13_buf": cache13_buf,
        "align_sorted_ids": align_sorted_ids,
        "align_expert_ids": align_expert_ids,
        "align_n_padded": align_n_padded,
        "align_cumsum": align_cumsum,
        "q_out_buf": q_out_buf,
        "attn_out_bf16_buf": attn_out_bf16_buf,
        "pre_expert_norm_buf": pre_expert_norm_buf,
        "expert_out_buf": expert_out_buf,
        "qkv_buf": qkv_buf,
        "attn_out_buf": attn_out_buf,
        "pooler_buf": pooler_buf,
        "unpooler_buf": unpooler_buf,
        "unpooler_slice_buf": unpooler_slice_buf,
        "padded_unpool_buf": padded_unpool_buf,
        "sms": sms,
        "marlin_mod_ep": marlin_mod_ep,
        "empty_bf16": empty_bf16,
        "empty_i32": empty_i32,
        "empty_f32": empty_f32,
        "c_tmp_buf": c_tmp_buf,
        "b_q_type_id": scalar_types.float4_e2m1f.id,
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _fast_marlin(
    a, c, b_q_weight, b_scales, b_global_scale,
    sorted_ids, block_experts, n_padded, topk_weights,
    moe_block_size, top_k, is_ep,
    size_m, size_n, size_k,
    num_groups, group_size,
    module, workspace, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
    use_fp32_reduce=True,
):
    if use_fp32_reduce:
        max_c_tmp_size = min(
            size_n * sorted_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = c_tmp_buf[:max_c_tmp_size]
    else:
        c_tmp = empty_f32

    module.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        empty_bf16,
        b_scales,
        b_global_scale,
        empty_bf16,
        empty_i32,
        empty_i32,
        workspace,
        sorted_ids,
        block_experts,
        n_padded,
        topk_weights,
        empty_bf16,
        c_tmp,
        moe_block_size,
        top_k,
        False,  # mul_topk_weights
        is_ep,
        b_q_type_id,
        size_m,
        size_n,
        size_k,
        False,  # has_act_order
        False,  # has_bias
        True,   # is_k_full
        False,  # has_zp
        num_groups,
        group_size,
        False,  # use_atomic_add
        use_fp32_reduce,
        False,  # is_zp_float
    )
    return c


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

    use_fp32_reduce = True
    ws = bank["workspace"]
    module = state["marlin_mod_ep"]
    empty_bf16 = state["empty_bf16"]
    empty_i32 = state["empty_i32"]
    empty_f32 = state["empty_f32"]
    c_tmp_buf = state["c_tmp_buf"]
    sms = state["sms"]
    b_q_type_id = state["b_q_type_id"]

    _fast_marlin(
        x, cache1, bank["w13"], bank["w13_s"], bank["w13_gs"],
        sorted_ids, block_experts, n_padded, topk_weights,
        block_size_m, TOP_K, True,
        num_tokens, gemm1_n, size_k,
        bank["w13_num_groups"], bank["w13_group_size"],
        module, ws, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
        use_fp32_reduce=use_fp32_reduce,
    )
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    _fast_marlin(
        cache2, cache3, bank["w2"], bank["w2_s"], bank["w2_gs"],
        sorted_ids, block_experts, n_padded, topk_weights,
        block_size_m, 1, True,
        total_pairs, size_k, size_n,
        bank["w2_num_groups"], bank["w2_group_size"],
        module, ws, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
        use_fp32_reduce=use_fp32_reduce,
    )
    y = cache3.view(num_tokens, TOP_K, size_k)

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
    qkv = _fast_mm(data["attn_input"], dense["qkv_gate_t"], state["qkv_buf"])

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
        num_warps=2,
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
        out=state["attn_out_bf16_buf"][:tokens],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]

    # 4. Output norm & gate + O projection
    o = state["o_norm_gate"](out.reshape(tokens, hq * d), gate)
    attn_output = _fast_mm(o, dense["o_t"], state["attn_out_buf"])

    # 5. Attention residual boundary + Router & Pooler
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)

    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _fast_mm(ffn_input, dense["pooler_t"], state["pooler_buf"])
    normed_expert_in = state["pre_expert_norm_buf"][:tokens]
    _fast_pre_expert_norm_kernel[(tokens,)](
        expert_input,
        state["pre_expert_w"],
        normed_expert_in,
        stride_x_tok=expert_input.stride(0),
        stride_out_tok=normed_expert_in.stride(0),
        N=D_EXPERT,
        eps=RMS_EPS,
    )
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

    # 6. Optimized MoE experts
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    expert_output = _optimized_experts(normed_expert_in, topk_weights, local_ids, state)

    # 7. Unpooler
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _fast_mm(expert_output, dense["unpooler_slice_t"], state["unpooler_slice_buf"])
        state["padded_unpool_buf"][:tokens, :UNPOOL_SLICE].copy_(unpooled)
        ffn_output = state["padded_unpool_buf"][:tokens]
    else:
        ffn_output = _fast_mm(expert_output, dense["unpooler_t"], state["unpooler_buf"])

    # 8. Post-FFN boundary
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
