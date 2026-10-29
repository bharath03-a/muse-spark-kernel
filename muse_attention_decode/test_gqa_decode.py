# okc: problem=muse_attention_decode
"""High-performance Attention for Muse Spark Open on H200:
- Fused prepare KV (QK-Norm + RoPE + FP8 Quantize + Cache write)
- Custom Single-KV-Load GQA Decode kernel for full attention (q_len=1, window=0)
  Loads KV cache ONCE for all 8 query heads in registers, eliminating FA3's 4x KV reload bottleneck.
- Fused Split-K combine + Output RMSNorm + Sigmoid Gate
- FlashAttention-3 for sliding window and speculative verify blocks (q_len=8)
"""

from typing import TypedDict
import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8
    positions: torch.Tensor    # int64 [B*L], each new token's position
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


FP8 = torch.float8_e4m3fn
_ROPE = {}
ROPE_POSITIONS = 1 << 21
_STATE = {}
MAX_TOKENS = 512


def _cos_sin(theta, d, device):
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


@triton.jit
def _fused_prepare_kv_kernel(
    qkv_ptr,
    positions_ptr,
    cos_sin_ptr,
    cache_loc_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
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

        k_scale_val = tl.load(k_scale_ptr) if scaled else 1.0
        v_scale_val = tl.load(v_scale_ptr) if scaled else 1.0

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

            k_even = (tl.load(qkv_ptr + tok_idx * stride_qkv_tok + k_start + idx_even).to(tl.float32) / rms)
            k_odd = (tl.load(qkv_ptr + tok_idx * stride_qkv_tok + k_start + idx_odd).to(tl.float32) / rms)

            rot_even = (k_even * cos_val - k_odd * sin_val) / k_scale_val
            rot_odd = (k_even * sin_val + k_odd * cos_val) / k_scale_val

            k_cache_dst = (page_idx * stride_k_cache_page +
                           offset_in_page * stride_k_cache_ps +
                           kv_h * stride_k_cache_head)
            tl.store(k_cache_ptr + k_cache_dst + idx_even, rot_even.to(k_cache_ptr.dtype.element_ty))
            tl.store(k_cache_ptr + k_cache_dst + idx_odd, rot_odd.to(k_cache_ptr.dtype.element_ty))
        else:
            k_scaled = (k_vals / rms) / k_scale_val
            k_cache_dst = (page_idx * stride_k_cache_page +
                           offset_in_page * stride_k_cache_ps +
                           kv_h * stride_k_cache_head + offs_d)
            tl.store(k_cache_ptr + k_cache_dst, k_scaled.to(k_cache_ptr.dtype.element_ty))

        v_offset = tok_idx * stride_qkv_tok + v_start + offs_d
        v_vals = tl.load(qkv_ptr + v_offset).to(tl.float32) / v_scale_val
        v_cache_dst = (page_idx * stride_v_cache_page +
                       offset_in_page * stride_v_cache_ps +
                       kv_h * stride_v_cache_head + offs_d)
        tl.store(v_cache_ptr + v_cache_dst, v_vals.to(v_cache_ptr.dtype.element_ty))


@triton.jit
def _gqa_decode_stage1_kernel(
    q_ptr,              # [batch, num_q_heads, 64]
    k_cache_ptr,        # [pages, page_size, hkv, 64]
    v_cache_ptr,        # [pages, page_size, hkv, 64]
    page_table_ptr,     # [batch, max_pages]
    cache_seqlens_ptr,  # [batch]
    partial_acc_ptr,    # [batch, NUM_SPLITS, num_q_heads, 64]
    partial_m_ptr,      # [batch, NUM_SPLITS, num_q_heads]
    partial_l_ptr,      # [batch, NUM_SPLITS, num_q_heads]
    k_scale_ptr,
    v_scale_ptr,
    scaled: tl.constexpr,
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_k_page,
    stride_k_ps,
    stride_k_head,
    stride_v_page,
    stride_v_ps,
    stride_v_head,
    stride_pt_b,
    stride_pacc_b,
    stride_pacc_s,
    stride_pacc_h,
    stride_pm_b,
    stride_pm_s,
    stride_pl_b,
    stride_pl_s,
    page_size,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % NUM_SPLITS
    pid_bk = pid // NUM_SPLITS
    kv_h = pid_bk % num_kv_heads
    b = pid_bk // num_kv_heads

    seq_len = tl.load(cache_seqlens_ptr + b)
    chunk = (seq_len + NUM_SPLITS - 1) // NUM_SPLITS
    t_start = split_id * chunk
    t_end = tl.minimum(t_start + chunk, seq_len)

    offs_d = tl.arange(0, HEAD_DIM)
    group: tl.constexpr = 8
    q_start = kv_h * group

    k_scale_val = tl.load(k_scale_ptr) if scaled else 1.0
    v_scale_val = tl.load(v_scale_ptr) if scaled else 1.0

    # 1. Load 8 Q vectors into registers
    q0 = tl.load(q_ptr + b * stride_q_b + (q_start + 0) * stride_q_h + offs_d).to(tl.float32) * sm_scale
    q1 = tl.load(q_ptr + b * stride_q_b + (q_start + 1) * stride_q_h + offs_d).to(tl.float32) * sm_scale
    q2 = tl.load(q_ptr + b * stride_q_b + (q_start + 2) * stride_q_h + offs_d).to(tl.float32) * sm_scale
    q3 = tl.load(q_ptr + b * stride_q_b + (q_start + 3) * stride_q_h + offs_d).to(tl.float32) * sm_scale
    q4 = tl.load(q_ptr + b * stride_q_b + (q_start + 4) * stride_q_h + offs_d).to(tl.float32) * sm_scale
    q5 = tl.load(q_ptr + b * stride_q_b + (q_start + 5) * stride_q_h + offs_d).to(tl.float32) * sm_scale
    q6 = tl.load(q_ptr + b * stride_q_b + (q_start + 6) * stride_q_h + offs_d).to(tl.float32) * sm_scale
    q7 = tl.load(q_ptr + b * stride_q_b + (q_start + 7) * stride_q_h + offs_d).to(tl.float32) * sm_scale

    acc0 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    acc1 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    acc2 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    acc3 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    acc4 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    acc5 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    acc6 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    acc7 = tl.zeros([HEAD_DIM], dtype=tl.float32)

    m0 = -1e9
    m1 = -1e9
    m2 = -1e9
    m3 = -1e9
    m4 = -1e9
    m5 = -1e9
    m6 = -1e9
    m7 = -1e9

    l0 = 0.0
    l1 = 0.0
    l2 = 0.0
    l3 = 0.0
    l4 = 0.0
    l5 = 0.0
    l6 = 0.0
    l7 = 0.0

    offs_n = tl.arange(0, BLOCK_N)

    # 2. Main loop over tokens in this split
    curr_t = t_start
    while curr_t < t_end:
        tok_idx = curr_t + offs_n
        mask_tok = tok_idx < t_end
        tok_safe = tl.where(mask_tok, tok_idx, 0)

        p_idx = tl.load(page_table_ptr + b * stride_pt_b + (tok_safe // page_size), mask=mask_tok, other=0).to(tl.int64)
        off_in_page = (tok_safe % page_size).to(tl.int64)

        k_offs = p_idx[:, None] * stride_k_page + off_in_page[:, None] * stride_k_ps + kv_h * stride_k_head + offs_d[None, :]
        k_val = tl.load(k_cache_ptr + k_offs, mask=mask_tok[:, None], other=0.0).to(tl.float32) * k_scale_val

        v_offs = p_idx[:, None] * stride_v_page + off_in_page[:, None] * stride_v_ps + kv_h * stride_v_head + offs_d[None, :]
        v_val = tl.load(v_cache_ptr + v_offs, mask=mask_tok[:, None], other=0.0).to(tl.float32) * v_scale_val

        # Head 0
        s0 = tl.sum(k_val * q0[None, :], axis=1)
        s0 = tl.where(mask_tok, s0, -1e9)
        m0_new = tl.maximum(m0, tl.max(s0, axis=0))
        alpha0 = tl.exp(m0 - m0_new)
        p0 = tl.exp(s0 - m0_new)
        p0 = tl.where(mask_tok, p0, 0.0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        acc0 = acc0 * alpha0 + tl.sum(p0[:, None] * v_val, axis=0)
        m0 = m0_new

        # Head 1
        s1 = tl.sum(k_val * q1[None, :], axis=1)
        s1 = tl.where(mask_tok, s1, -1e9)
        m1_new = tl.maximum(m1, tl.max(s1, axis=0))
        alpha1 = tl.exp(m1 - m1_new)
        p1 = tl.exp(s1 - m1_new)
        p1 = tl.where(mask_tok, p1, 0.0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        acc1 = acc1 * alpha1 + tl.sum(p1[:, None] * v_val, axis=0)
        m1 = m1_new

        # Head 2
        s2 = tl.sum(k_val * q2[None, :], axis=1)
        s2 = tl.where(mask_tok, s2, -1e9)
        m2_new = tl.maximum(m2, tl.max(s2, axis=0))
        alpha2 = tl.exp(m2 - m2_new)
        p2 = tl.exp(s2 - m2_new)
        p2 = tl.where(mask_tok, p2, 0.0)
        l2 = l2 * alpha2 + tl.sum(p2, axis=0)
        acc2 = acc2 * alpha2 + tl.sum(p2[:, None] * v_val, axis=0)
        m2 = m2_new

        # Head 3
        s3 = tl.sum(k_val * q3[None, :], axis=1)
        s3 = tl.where(mask_tok, s3, -1e9)
        m3_new = tl.maximum(m3, tl.max(s3, axis=0))
        alpha3 = tl.exp(m3 - m3_new)
        p3 = tl.exp(s3 - m3_new)
        p3 = tl.where(mask_tok, p3, 0.0)
        l3 = l3 * alpha3 + tl.sum(p3, axis=0)
        acc3 = acc3 * alpha3 + tl.sum(p3[:, None] * v_val, axis=0)
        m3 = m3_new

        # Head 4
        s4 = tl.sum(k_val * q4[None, :], axis=1)
        s4 = tl.where(mask_tok, s4, -1e9)
        m4_new = tl.maximum(m4, tl.max(s4, axis=0))
        alpha4 = tl.exp(m4 - m4_new)
        p4 = tl.exp(s4 - m4_new)
        p4 = tl.where(mask_tok, p4, 0.0)
        l4 = l4 * alpha4 + tl.sum(p4, axis=0)
        acc4 = acc4 * alpha4 + tl.sum(p4[:, None] * v_val, axis=0)
        m4 = m4_new

        # Head 5
        s5 = tl.sum(k_val * q5[None, :], axis=1)
        s5 = tl.where(mask_tok, s5, -1e9)
        m5_new = tl.maximum(m5, tl.max(s5, axis=0))
        alpha5 = tl.exp(m5 - m5_new)
        p5 = tl.exp(s5 - m5_new)
        p5 = tl.where(mask_tok, p5, 0.0)
        l5 = l5 * alpha5 + tl.sum(p5, axis=0)
        acc5 = acc5 * alpha5 + tl.sum(p5[:, None] * v_val, axis=0)
        m5 = m5_new

        # Head 6
        s6 = tl.sum(k_val * q6[None, :], axis=1)
        s6 = tl.where(mask_tok, s6, -1e9)
        m6_new = tl.maximum(m6, tl.max(s6, axis=0))
        alpha6 = tl.exp(m6 - m6_new)
        p6 = tl.exp(s6 - m6_new)
        p6 = tl.where(mask_tok, p6, 0.0)
        l6 = l6 * alpha6 + tl.sum(p6, axis=0)
        acc6 = acc6 * alpha6 + tl.sum(p6[:, None] * v_val, axis=0)
        m6 = m6_new

        # Head 7
        s7 = tl.sum(k_val * q7[None, :], axis=1)
        s7 = tl.where(mask_tok, s7, -1e9)
        m7_new = tl.maximum(m7, tl.max(s7, axis=0))
        alpha7 = tl.exp(m7 - m7_new)
        p7 = tl.exp(s7 - m7_new)
        p7 = tl.where(mask_tok, p7, 0.0)
        l7 = l7 * alpha7 + tl.sum(p7, axis=0)
        acc7 = acc7 * alpha7 + tl.sum(p7[:, None] * v_val, axis=0)
        m7 = m7_new

        curr_t += BLOCK_N

    # 3. Store split partial outputs
    pacc_base = b * stride_pacc_b + split_id * stride_pacc_s
    pm_base = b * stride_pm_b + split_id * stride_pm_s
    pl_base = b * stride_pl_b + split_id * stride_pl_s

    tl.store(partial_acc_ptr + pacc_base + (q_start + 0) * stride_pacc_h + offs_d, acc0)
    tl.store(partial_m_ptr + pm_base + (q_start + 0), m0)
    tl.store(partial_l_ptr + pl_base + (q_start + 0), l0)

    tl.store(partial_acc_ptr + pacc_base + (q_start + 1) * stride_pacc_h + offs_d, acc1)
    tl.store(partial_m_ptr + pm_base + (q_start + 1), m1)
    tl.store(partial_l_ptr + pl_base + (q_start + 1), l1)

    tl.store(partial_acc_ptr + pacc_base + (q_start + 2) * stride_pacc_h + offs_d, acc2)
    tl.store(partial_m_ptr + pm_base + (q_start + 2), m2)
    tl.store(partial_l_ptr + pl_base + (q_start + 2), l2)

    tl.store(partial_acc_ptr + pacc_base + (q_start + 3) * stride_pacc_h + offs_d, acc3)
    tl.store(partial_m_ptr + pm_base + (q_start + 3), m3)
    tl.store(partial_l_ptr + pl_base + (q_start + 3), l3)

    tl.store(partial_acc_ptr + pacc_base + (q_start + 4) * stride_pacc_h + offs_d, acc4)
    tl.store(partial_m_ptr + pm_base + (q_start + 4), m4)
    tl.store(partial_l_ptr + pl_base + (q_start + 4), l4)

    tl.store(partial_acc_ptr + pacc_base + (q_start + 5) * stride_pacc_h + offs_d, acc5)
    tl.store(partial_m_ptr + pm_base + (q_start + 5), m5)
    tl.store(partial_l_ptr + pl_base + (q_start + 5), l5)

    tl.store(partial_acc_ptr + pacc_base + (q_start + 6) * stride_pacc_h + offs_d, acc6)
    tl.store(partial_m_ptr + pm_base + (q_start + 6), m6)
    tl.store(partial_l_ptr + pl_base + (q_start + 6), l6)

    tl.store(partial_acc_ptr + pacc_base + (q_start + 7) * stride_pacc_h + offs_d, acc7)
    tl.store(partial_m_ptr + pm_base + (q_start + 7), m7)
    tl.store(partial_l_ptr + pl_base + (q_start + 7), l7)


@triton.jit
def _gqa_decode_combine_norm_gate_kernel(
    partial_acc_ptr,  # [batch, NUM_SPLITS, num_q_heads, 64]
    partial_m_ptr,    # [batch, NUM_SPLITS, num_q_heads]
    partial_l_ptr,    # [batch, NUM_SPLITS, num_q_heads]
    gate_ptr,         # [batch, num_q_heads * 64]
    out_ptr,          # [batch, num_q_heads * 64]
    stride_pacc_b,
    stride_pacc_s,
    stride_pacc_h,
    stride_pm_b,
    stride_pm_s,
    stride_pl_b,
    stride_pl_s,
    stride_gate_b,
    stride_out_b,
    num_q_heads: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    q_h = pid % num_q_heads
    b = pid // num_q_heads

    offs_d = tl.arange(0, HEAD_DIM)

    # 1. Global max across splits
    m_max = -1e9
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(partial_m_ptr + b * stride_pm_b + s * stride_pm_s + q_h)
        m_max = tl.maximum(m_max, m_s)

    # 2. Rescale & sum
    l_sum = 0.0
    acc_sum = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(partial_m_ptr + b * stride_pm_b + s * stride_pm_s + q_h)
        l_s = tl.load(partial_l_ptr + b * stride_pl_b + s * stride_pl_s + q_h)
        weight = tl.where(m_s > -1e8, tl.exp(m_s - m_max), 0.0)
        l_sum += l_s * weight
        acc_s = tl.load(partial_acc_ptr + b * stride_pacc_b + s * stride_pacc_s + q_h * stride_pacc_h + offs_d)
        acc_sum += acc_s * weight

    l_sum_safe = tl.maximum(l_sum, 1e-12)
    a_head = acc_sum / l_sum_safe

    # 3. RMSNorm + Sigmoid Gate (Fused)
    rms = tl.sqrt(tl.sum(a_head * a_head) / HEAD_DIM + 1e-5)
    gate_vals = tl.load(gate_ptr + b * stride_gate_b + q_h * HEAD_DIM + offs_d).to(tl.float32)
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate_vals))

    out_vals = (a_head / rms) * sigmoid_gate
    tl.store(out_ptr + b * stride_out_b + q_h * HEAD_DIM + offs_d, out_vals.to(out_ptr.dtype.element_ty))


def _get_buffers(batch, num_q_heads, max_splits, device):
    key = (batch, num_q_heads, max_splits, str(device))
    buf = _STATE.get(key)
    if buf is None:
        buf = {
            "q_out": torch.empty((MAX_TOKENS, num_q_heads, 64), dtype=FP8, device=device),
            "partial_acc": torch.empty((batch, max_splits, num_q_heads, 64), dtype=torch.float32, device=device),
            "partial_m": torch.empty((batch, max_splits, num_q_heads), dtype=torch.float32, device=device),
            "partial_l": torch.empty((batch, max_splits, num_q_heads), dtype=torch.float32, device=device),
            "out_buf": torch.empty((batch, num_q_heads * 64), dtype=torch.bfloat16, device=device),
        }
        _STATE[key] = buf
    return buf


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    has_rope = cfg["rope_theta"] != 0.0
    scaled = cfg["kv_scale"] != 1.0

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    cos_sin = _cos_sin(cfg["rope_theta"], d, device) if has_rope else qkv

    bufs = _get_buffers(batch, hq, 64, device)
    q_out = bufs["q_out"][:tokens]

    # 1. Fused prepare KV
    grid = (tokens, hq + hkv)
    _fused_prepare_kv_kernel[grid](
        qkv,
        data["positions"],
        cos_sin,
        data["cache_loc"],
        q_out,
        k_cache,
        v_cache,
        data["k_scale"],
        data["v_scale"],
        has_rope=has_rope,
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
        HEAD_DIM=d,
        num_q_heads=hq,
        num_kv_heads=hkv,
        num_warps=2,
    )

    window = cfg["window"]

    # 2. Dispatch: Custom single-load GQA decode for full attention decode steps
    if length == 1 and not window:
        total_streams = batch * hkv
        if total_streams >= 128:
            num_splits = 1
        elif total_streams >= 64:
            num_splits = 2
        elif total_streams >= 32:
            num_splits = 4
        elif total_streams >= 16:
            num_splits = 8
        elif total_streams >= 8:
            num_splits = 16
        elif total_streams >= 4:
            num_splits = 32
        else:
            num_splits = 64

        pacc, pm, pl = bufs["partial_acc"], bufs["partial_m"], bufs["partial_l"]
        out_buf = bufs["out_buf"]

        sm_scale = cfg["sm_scale"] * cfg["qk_scale_factor"]

        grid1 = (total_streams * num_splits,)
        _gqa_decode_stage1_kernel[grid1](
            q_out,
            k_cache,
            v_cache,
            data["page_table"],
            data["cache_seqlens"],
            pacc,
            pm,
            pl,
            data["k_scale"],
            data["v_scale"],
            scaled=scaled,
            sm_scale=sm_scale,
            stride_q_b=q_out.stride(0),
            stride_q_h=q_out.stride(1),
            stride_k_page=k_cache.stride(0),
            stride_k_ps=k_cache.stride(1),
            stride_k_head=k_cache.stride(2),
            stride_v_page=v_cache.stride(0),
            stride_v_ps=v_cache.stride(1),
            stride_v_head=v_cache.stride(2),
            stride_pt_b=data["page_table"].stride(0),
            stride_pacc_b=pacc.stride(0),
            stride_pacc_s=pacc.stride(1),
            stride_pacc_h=pacc.stride(2),
            stride_pm_b=pm.stride(0),
            stride_pm_s=pm.stride(1),
            stride_pl_b=pl.stride(0),
            stride_pl_s=pl.stride(1),
            page_size=cfg["page_size"],
            num_q_heads=hq,
            num_kv_heads=hkv,
            NUM_SPLITS=num_splits,
            BLOCK_N=64,
            HEAD_DIM=d,
            num_warps=4,
        )

        gate = qkv[:, (hq + 2 * hkv) * d:]
        grid2 = (batch * hq,)
        _gqa_decode_combine_norm_gate_kernel[grid2](
            pacc,
            pm,
            pl,
            gate,
            out_buf,
            stride_pacc_b=pacc.stride(0),
            stride_pacc_s=pacc.stride(1),
            stride_pacc_h=pacc.stride(2),
            stride_pm_b=pm.stride(0),
            stride_pm_s=pm.stride(1),
            stride_pl_b=pl.stride(0),
            stride_pl_s=pl.stride(1),
            stride_gate_b=gate.stride(0),
            stride_out_b=out_buf.stride(0),
            num_q_heads=hq,
            NUM_SPLITS=num_splits,
            HEAD_DIM=d,
            num_warps=2,
        )
        return out_buf.view(batch, 1, hq * d)

    # 3. FlashAttention-3 for sliding window layers and speculative verify blocks (q_len=8)
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))

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

    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=1e-5, has_norm=True)
    return o.view(batch, length, hq * d)
