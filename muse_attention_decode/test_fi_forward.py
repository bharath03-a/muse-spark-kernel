# okc: problem=muse_attention_decode
"""Test FlashInfer BatchDecode wrapper forward call and correctness."""

from typing import TypedDict
import torch
import triton
import triton.language as tl
import flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

FP8 = torch.float8_e4m3fn
_ROPE = {}
ROPE_POSITIONS = 1 << 21
_BUFFERS = {}
_OUT_BUFFERS = {}
_WRAPPERS = {}


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


def kernel_fn(data):
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
    page_size = cfg["page_size"]
    sm_scale = cfg["sm_scale"] * cfg["qk_scale_factor"]

    # Test FlashInfer BatchDecode for q_len=1
    if length == 1:
        try:
            key = (batch, page_size)
            if key not in _WRAPPERS:
                workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
                _WRAPPERS[key] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                    use_tensor_cores=True,
                )
            wrapper = _WRAPPERS[key]
            
            seq_lens = data["cache_seqlens"]
            indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
            indptr[1:] = torch.cumsum(seq_lens, 0)
            
            indices_list = []
            for b in range(batch):
                num_pages = (int(seq_lens[b].item()) + page_size - 1) // page_size
                indices_list.append(data["page_table"][b, :num_pages])
            indices = torch.cat(indices_list)
            last_page_len = (seq_lens % page_size)
            last_page_len = torch.where(last_page_len == 0, page_size, last_page_len)
            
            wrapper.plan(
                indptr=indptr,
                indices=indices,
                last_page_len=last_page_len,
                num_qo_heads=hq,
                num_kv_heads=hkv,
                head_dim=d,
                page_size=page_size,
                data_type=torch.float8_e4m3fn,
            )
            
            attn_out = wrapper.forward(
                q_out.squeeze(1),
                (k_cache, v_cache),
                sm_scale=sm_scale,
                window_left=window - 1 if window else -1,
            )
            print("wrapper.forward SUCCESS! shape:", attn_out.shape)
            gate = qkv[:, (hq + 2 * hkv) * d:]
            o = rms_norm_gate(attn_out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=1e-5, has_norm=True)
            return o.view(batch, length, hq * d)
        except Exception as e:
            print(f"wrapper forward error: {e}")
            import traceback
            traceback.print_exc()

    # Fallback to FA3 for verify
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    attn_out = torch.empty((tokens, hq, d), dtype=torch.bfloat16, device=device)
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
        pack_gqa=True,
        scheduler_metadata=data["scheduler_metadata"],
        out=attn_out,
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d), gate, None, head_dim=d, eps=1e-5, has_norm=True)
    return o.view(batch, length, hq * d)
