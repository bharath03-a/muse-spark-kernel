# okc: problem=muse_attention_decode
"""Test get_scheduler_metadata with pack_gqa=True for verify and sliding layers."""

from typing import TypedDict
import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
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
_BUFFERS = {}
_OUT_BUFFERS = {}


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
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    eps: tl.constexpr = 1e-5

    offs_d = tl.arange(0, HEAD_DIM)
    offs_d2 = tl.arange(0, HEAD_DIM // 2)
    tok_offset = tok_idx * stride_qkv_tok

    if head_idx < num_q_heads:
        head_offset = head_idx * HEAD_DIM
        out_offset = tok_idx * stride_q_tok + head_offset
        if has_rope:
            pos = tl.load(positions_ptr + tok_idx)
            pos_offset = pos * HEAD_DIM
            cos_val = tl.load(cos_sin_ptr + pos_offset + offs_d2)
            sin_val = tl.load(cos_sin_ptr + pos_offset + (HEAD_DIM // 2) + offs_d2)

            idx_even = offs_d2 * 2
            idx_odd = offs_d2 * 2 + 1

            q_even = tl.load(qkv_ptr + tok_offset + head_offset + idx_even).to(tl.float32)
            q_odd = tl.load(qkv_ptr + tok_offset + head_offset + idx_odd).to(tl.float32)

            sum_sq = tl.sum(q_even * q_even) + tl.sum(q_odd * q_odd)
            rms = tl.sqrt(sum_sq / HEAD_DIM + eps)
            inv_rms = 1.0 / rms
            q_even = q_even * inv_rms
            q_odd = q_odd * inv_rms

            rot_even = q_even * cos_val - q_odd * sin_val
            rot_odd = q_even * sin_val + q_odd * cos_val

            tl.store(q_out_ptr + out_offset + idx_even, rot_even.to(q_out_ptr.dtype.element_ty))
            tl.store(q_out_ptr + out_offset + idx_odd, rot_odd.to(q_out_ptr.dtype.element_ty))
        else:
            q_vals = tl.load(qkv_ptr + tok_offset + head_offset + offs_d).to(tl.float32)
            rms = tl.sqrt(tl.sum(q_vals * q_vals) / HEAD_DIM + eps)
            q_norm = q_vals / rms
            tl.store(q_out_ptr + out_offset + offs_d, q_norm.to(q_out_ptr.dtype.element_ty))

    else:
        kv_head_idx = head_idx - num_q_heads
        k_offset = num_q_heads * HEAD_DIM + kv_head_idx * HEAD_DIM
        v_offset = (num_q_heads + num_kv_heads) * HEAD_DIM + kv_head_idx * HEAD_DIM

        loc = tl.load(cache_loc_ptr + tok_idx)
        page_idx = loc // PAGE_SIZE
        offset_in_page = loc % PAGE_SIZE

        if has_rope:
            pos = tl.load(positions_ptr + tok_idx)
            pos_offset = pos * HEAD_DIM
            cos_val = tl.load(cos_sin_ptr + pos_offset + offs_d2)
            sin_val = tl.load(cos_sin_ptr + pos_offset + (HEAD_DIM // 2) + offs_d2)

            idx_even = offs_d2 * 2
            idx_odd = offs_d2 * 2 + 1

            k_even = tl.load(qkv_ptr + tok_offset + k_offset + idx_even).to(tl.float32)
            k_odd = tl.load(qkv_ptr + tok_offset + k_offset + idx_odd).to(tl.float32)

            sum_sq = tl.sum(k_even * k_even) + tl.sum(k_odd * k_odd)
            rms = tl.sqrt(sum_sq / HEAD_DIM + eps)
            inv_rms = 1.0 / rms
            k_even = k_even * inv_rms
            k_odd = k_odd * inv_rms

            rot_k_even = k_even * cos_val - k_odd * sin_val
            rot_k_odd = k_even * sin_val + k_odd * cos_val

            if scaled:
                k_scale = tl.load(k_scale_ptr)
                inv_k_scale = 1.0 / k_scale
                rot_k_even = rot_k_even * inv_k_scale
                rot_k_odd = rot_k_odd * inv_k_scale

            k_dest = page_idx * stride_k_cache_page + offset_in_page * stride_k_cache_ps + kv_head_idx * stride_k_cache_head
            tl.store(k_cache_ptr + k_dest + idx_even, rot_k_even.to(k_cache_ptr.dtype.element_ty))
            tl.store(k_cache_ptr + k_dest + idx_odd, rot_k_odd.to(k_cache_ptr.dtype.element_ty))
        else:
            k_vals = tl.load(qkv_ptr + tok_offset + k_offset + offs_d).to(tl.float32)
            rms = tl.sqrt(tl.sum(k_vals * k_vals) / HEAD_DIM + eps)
            k_norm = k_vals / rms
            if scaled:
                k_scale = tl.load(k_scale_ptr)
                k_norm = k_norm / k_scale
            k_dest = page_idx * stride_k_cache_page + offset_in_page * stride_k_cache_ps + kv_head_idx * stride_k_cache_head
            tl.store(k_cache_ptr + k_dest + offs_d, k_norm.to(k_cache_ptr.dtype.element_ty))

        v_vals = tl.load(qkv_ptr + tok_offset + v_offset + offs_d).to(tl.float32)
        if scaled:
            v_scale = tl.load(v_scale_ptr)
            v_vals = v_vals / v_scale
        v_dest = page_idx * stride_v_cache_page + offset_in_page * stride_v_cache_ps + kv_head_idx * stride_v_cache_head
        tl.store(v_cache_ptr + v_dest + offs_d, v_vals.to(v_cache_ptr.dtype.element_ty))


def _get_q_out(tokens, hq, d, device):
    key = (tokens, hq, d)
    buf = _BUFFERS.get(key)
    if buf is None:
        buf = _BUFFERS[key] = torch.empty((tokens, hq, d), dtype=FP8, device=device)
    return buf


def _get_attn_out(tokens, hq, d, device):
    key = (tokens, hq, d)
    buf = _OUT_BUFFERS.get(key)
    if buf is None:
        buf = _OUT_BUFFERS[key] = torch.empty((tokens, hq, d), dtype=torch.bfloat16, device=device)
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

    q_out = _get_q_out(tokens, hq, d, device)

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
        PAGE_SIZE=cfg["page_size"],
        HEAD_DIM=d,
        num_q_heads=hq,
        num_kv_heads=hkv,
        num_warps=1,
    )

    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))

    attn_out = _get_attn_out(tokens, hq, d, device)

    # Compute custom scheduler metadata for verify blocks and missing metadata
    scheduler_metadata = data["scheduler_metadata"]
    if scheduler_metadata is None and not window:
        max_k = data["page_table"].shape[1] * cfg["page_size"]
        scheduler_metadata = get_scheduler_metadata(
            batch_size=batch,
            max_seqlen_q=length,
            max_seqlen_k=max_k,
            num_heads=hq,
            num_heads_k=hkv,
            headdim=d,
            cache_seqlens=data["cache_seqlens"],
            qkv_dtype=FP8,
            cu_seqlens_q=data["cu_seqlens_q"],
            page_size=cfg["page_size"],
            causal=True,
            num_splits=0,
            pack_gqa=True,
        )

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
        pack_gqa=True,
        scheduler_metadata=scheduler_metadata,
        out=attn_out,
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]

    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d), gate, None, head_dim=d, eps=1e-5, has_norm=True)
    return o.view(batch, length, hq * d)
