# okc: problem=muse_attention_decode
"""Fused sliding decode attention kernel in Triton + FA3 fallback for full attention."""

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
        # Q head
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
        # KV head
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
def _triton_sliding_attention_kernel(
    q_ptr,                # [tokens, Hq, HEAD_DIM] bfloat16
    qkv_ptr,              # [tokens, (2*Hq + 2*Hkv)*HEAD_DIM] for gate logits
    k_cache_ptr,          # [pages, page_size, Hkv, HEAD_DIM] float8_e4m3fn
    v_cache_ptr,          # [pages, page_size, Hkv, HEAD_DIM] float8_e4m3fn
    page_table_ptr,       # [batch, max_pages] int32
    seq_lens_ptr,         # [batch] int32
    out_ptr,              # [tokens, Hq * HEAD_DIM] bfloat16
    k_scale_ptr,
    v_scale_ptr,
    softmax_scale,
    window_size,          # 2048
    q_len,                # 1 or 8
    page_size,            # 1 or 64
    stride_q_tok,
    stride_q_head,
    stride_qkv_tok,
    stride_k_page,
    stride_k_ps,
    stride_k_head,
    stride_v_page,
    stride_v_ps,
    stride_v_head,
    stride_pt_b,
    stride_out_tok,
    scaled: tl.constexpr,
    HEAD_DIM: tl.constexpr,      # 64
    BLOCK_N: tl.constexpr,       # 64
    num_q_heads: tl.constexpr,   # 16
    num_kv_heads: tl.constexpr,  # 2
):
    # Grid: (tokens, num_q_heads)
    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    b = tok_idx // q_len
    j = tok_idx % q_len

    group: tl.constexpr = num_q_heads // num_kv_heads  # 8
    kv_h = head_idx // group                           # 0 or 1

    offs_d = tl.arange(0, HEAD_DIM)

    # Load query vector [HEAD_DIM]
    q_ptrs = q_ptr + tok_idx * stride_q_tok + head_idx * stride_q_head + offs_d
    q_vec = tl.load(q_ptrs).to(tl.float32)

    # Token position
    seq_len = tl.load(seq_lens_ptr + b)
    pos = seq_len + j

    # Sliding window range [start, end)
    start = tl.maximum(pos - window_size + 1, 0)
    end = pos + 1

    k_scale_val = tl.load(k_scale_ptr) if scaled else 1.0
    v_scale_val = tl.load(v_scale_ptr) if scaled else 1.0
    scaled_sm_scale = softmax_scale * k_scale_val

    # Online softmax accumulators
    m_prev = -1e9
    l_prev = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    gate_offset = (num_q_heads + 2 * num_kv_heads) * HEAD_DIM + head_idx * HEAD_DIM

    offs_n = tl.arange(0, BLOCK_N)
    tok_start = start

    while tok_start < end:
        tok_indices = tok_start + offs_n
        mask = tok_indices < end
        mask_2d = mask[:, None] & (offs_d[None, :] < HEAD_DIM)

        # Page table lookup
        page_idx = tok_indices // page_size
        offset_in_page = tok_indices % page_size

        pt_offsets = b * stride_pt_b + page_idx
        phys_page = tl.load(page_table_ptr + pt_offsets, mask=mask, other=0).to(tl.int64)

        # Row offsets in k_cache and v_cache
        k_row_offsets = phys_page * stride_k_page + offset_in_page * stride_k_ps + kv_h * stride_k_head
        v_row_offsets = phys_page * stride_v_page + offset_in_page * stride_v_ps + kv_h * stride_v_head

        # Load K chunk [BLOCK_N, HEAD_DIM]
        k_ptrs = k_cache_ptr + k_row_offsets[:, None] + offs_d[None, :]
        k_chunk = tl.load(k_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

        # Load V chunk [BLOCK_N, HEAD_DIM]
        v_ptrs = v_cache_ptr + v_row_offsets[:, None] + offs_d[None, :]
        v_chunk = tl.load(v_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

        # Dot product: q_vec [HEAD_DIM] with k_chunk [BLOCK_N, HEAD_DIM]
        logits = tl.sum(q_vec[None, :] * k_chunk, axis=1) * scaled_sm_scale
        logits = tl.where(mask, logits, -1e9)

        # Online softmax update
        m_chunk = tl.max(logits, axis=0)
        m_new = tl.maximum(m_prev, m_chunk)

        alpha = tl.exp(m_prev - m_new)
        p = tl.exp(logits - m_new)
        p = tl.where(mask, p, 0.0)

        l_prev = l_prev * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v_chunk, axis=0)
        m_prev = m_new

        tok_start += BLOCK_N

    # Normalize softmax output
    out_vec = acc / (l_prev + 1e-10)
    if scaled:
        out_vec = out_vec * v_scale_val

    # Fused per-head RMSNorm: eps = 1e-5
    rms = tl.sqrt(tl.sum(out_vec * out_vec, axis=0) / HEAD_DIM + 1e-5)
    out_normed = out_vec / rms

    # Fused gate: sigmoid(gate)
    gate_ptrs = qkv_ptr + tok_idx * stride_qkv_tok + gate_offset + offs_d
    gate_vals = tl.load(gate_ptrs).to(tl.float32)
    sigmoid_gate = tl.sigmoid(gate_vals)

    final_out = out_normed * sigmoid_gate

    # Store directly to out tensor
    out_dst = out_ptr + tok_idx * stride_out_tok + head_idx * HEAD_DIM + offs_d
    tl.store(out_dst, final_out.to(out_ptr.dtype.element_ty))


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

    window = cfg["window"]

    # If sliding window: use bfloat16 for q_out; if full attention: use FP8 for FA3
    q_dtype = torch.bfloat16 if window else FP8
    q_out = torch.empty((tokens, hq, d), dtype=q_dtype, device=device)

    # 1. Prepare KV + Q
    grid_prep = (tokens, hq + hkv)
    _fused_prepare_kv_kernel[grid_prep](
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

    if window:
        # 2a. Sliding attention: fused Triton kernel (attention + RMSNorm + sigmoid gate)
        out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=device)
        grid_attn = (tokens, hq)
        _triton_sliding_attention_kernel[grid_attn](
            q_out,
            qkv,
            k_cache,
            v_cache,
            data["page_table"],
            data["seq_lens"],
            out,
            data["k_scale"],
            data["v_scale"],
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
            window_size=window,
            q_len=length,
            page_size=cfg["page_size"],
            stride_q_tok=q_out.stride(0),
            stride_q_head=q_out.stride(1),
            stride_qkv_tok=qkv.stride(0),
            stride_k_page=k_cache.stride(0),
            stride_k_ps=k_cache.stride(1),
            stride_k_head=k_cache.stride(2),
            stride_v_page=v_cache.stride(0),
            stride_v_ps=v_cache.stride(1),
            stride_v_head=v_cache.stride(2),
            stride_pt_b=data["page_table"].stride(0),
            stride_out_tok=out.stride(0),
            scaled=scaled,
            HEAD_DIM=d,
            BLOCK_N=64,
            num_q_heads=hq,
            num_kv_heads=hkv,
            num_warps=2,
        )
        return out.view(batch, length, hq * d)

    else:
        # 2b. Full attention: FA3 + rms_norm_gate
        descale = {}
        if scaled:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                           v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))

        fa_out = flash_attn_with_kvcache(
            q=q_out,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"],
            cu_seqlens_q=data["cu_seqlens_q"],
            max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
            causal=True,
            window_size=(-1, -1),
            scheduler_metadata=data["scheduler_metadata"],
            **descale,
        )
        if isinstance(fa_out, tuple):
            fa_out = fa_out[0]

        gate = qkv[:, (hq + 2 * hkv) * d:]
        o = rms_norm_gate(fa_out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=1e-5, has_norm=True)
        return o.view(batch, length, hq * d)
