# okc: problem=muse_attention_decode
"""Test FlashInfer decode for muse_attention_decode."""

from typing import TypedDict
import torch
import triton
import triton.language as tl
import flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float

class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
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

_WORKSPACE = None

def _get_workspace(device):
    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return _WORKSPACE

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

    q_out = torch.empty((tokens, hq, d), dtype=torch.bfloat16, device=device)

    # 1. Fused prepare KV & Q
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
    sm_scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    window_left = (window - 1) if window > 0 else -1

    # 2. Try FlashInfer for decode (length == 1) if not capturing or test
    use_flashinfer = (length == 1 and not scaled)
    if use_flashinfer:
        try:
            ws = _get_workspace(device)
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                ws,
                kv_layout="NHD",
                use_tensor_cores=True,
            )
            cache_seqlens = data["cache_seqlens"]
            page_size = cfg["page_size"]
            num_pages_per_seq = (cache_seqlens + page_size - 1) // page_size
            indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
            indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)

            indices_list = [data["page_table"][b, :int(num_pages_per_seq[b].item())] for b in range(batch)]
            indices = torch.cat(indices_list, dim=0).contiguous()
            last_page_len = (cache_seqlens - 1) % page_size + 1

            wrapper.plan(
                indptr=indptr,
                indices=indices,
                last_page_len=last_page_len,
                num_qo_heads=hq,
                num_kv_heads=hkv,
                head_dim=d,
                page_size=page_size,
            )
            out = wrapper.forward(
                q_out.view(batch, hq, d),
                (k_cache, v_cache),
                window_left=window_left,
                sm_scale=sm_scale,
            )
        except Exception as e:
            use_flashinfer = False

    if not use_flashinfer:
        descale = {}
        if scaled:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                           v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
        out = flash_attn_with_kvcache(
            q=q_out.to(FP8),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"],
            cu_seqlens_q=data["cu_seqlens_q"],
            max_seqlen_q=length,
            softmax_scale=sm_scale,
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
