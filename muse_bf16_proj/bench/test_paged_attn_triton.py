import math
import torch
import triton
import triton.language as tl
from attn_reference import generate_input, check_implementation, ref_kernel, relative_l2

FP8 = torch.float8_e4m3fn
HEAD_DIM = 64

@triton.jit
def paged_sliding_attn_decode_kernel(
    q_ptr,           # [batch, hq, 64] float32 or FP8
    k_cache_ptr,     # [pages, page_size, hkv, 64] FP8
    v_cache_ptr,     # [pages, page_size, hkv, 64] FP8
    page_table_ptr,  # [batch, max_pages] int32
    seq_lens_ptr,    # [batch] int32
    gate_ptr,        # [batch, hq * 64] bfloat16
    out_ptr,         # [batch, hq * 64] bfloat16
    k_scale_val,
    v_scale_val,
    sm_scale: tl.constexpr,
    window: tl.constexpr,
    stride_pt: tl.constexpr,
    stride_k_page: tl.constexpr,
    stride_k_ps: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_v_page: tl.constexpr,
    stride_v_ps: tl.constexpr,
    stride_v_head: tl.constexpr,
    page_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    group: tl.constexpr = num_q_heads // num_kv_heads
    kv_head_idx = head_idx // group

    # Load Q [64]
    offs_d = tl.arange(0, HEAD_DIM)
    q_offset = batch_idx * num_q_heads * HEAD_DIM + head_idx * HEAD_DIM + offs_d
    q = tl.load(q_ptr + q_offset).to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    total_tokens = seq_len + 1
    first_token = tl.maximum(0, total_tokens - window)
    num_tokens = total_tokens - first_token

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, num_tokens, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < num_tokens

        pos = first_token + offs_n
        logical_page = pos // page_size
        offset_in_page = pos % page_size

        pt_offset = batch_idx * stride_pt + logical_page
        phys_page = tl.load(page_table_ptr + pt_offset, mask=mask_n, other=0)

        # Load K [BLOCK_N, HEAD_DIM]
        k_ptrs = (k_cache_ptr +
                  phys_page[:, None] * stride_k_page +
                  offset_in_page[:, None] * stride_k_ps +
                  kv_head_idx * stride_k_head +
                  offs_d[None, :])
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32) * k_scale_val

        # Compute dot product
        s = tl.sum(q[None, :] * k, axis=1) * sm_scale
        s = tl.where(mask_n, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)

        l_new = l_i * alpha + tl.sum(p)

        # Load V [BLOCK_N, HEAD_DIM]
        v_ptrs = (v_cache_ptr +
                  phys_page[:, None] * stride_v_page +
                  offset_in_page[:, None] * stride_v_ps +
                  kv_head_idx * stride_v_head +
                  offs_d[None, :])
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32) * v_scale_val

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    # Final attention output for this head: [HEAD_DIM]
    out_h = acc / l_i

    # RMSNorm on out_h
    rms = tl.sqrt(tl.sum(out_h * out_h) / HEAD_DIM + 1e-5)
    out_norm = out_h / rms

    # Sigmoid(gate)
    gate_offset = batch_idx * num_q_heads * HEAD_DIM + head_idx * HEAD_DIM + offs_d
    gate = tl.load(gate_ptr + gate_offset).to(tl.float32)
    sig_gate = tl.sigmoid(gate)

    res = out_norm * sig_gate

    # Store directly to out_ptr
    tl.store(out_ptr + gate_offset, res.to(tl.bfloat16))


def test():
    device = "cuda"
    data = generate_input(batch=1, ctx=80000, heads="tp8", layer="sliding", seed=42, page=1, device=device)
    cfg = data["config"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    batch = cfg["batch"]

    from test_fused_prepare import fused_prepare_kv_kernel, get_cos_sin
    qkv = data["qkv"].view(batch, -1)
    q_out = torch.empty((batch, hq * d), dtype=torch.bfloat16, device=device)
    cos_sin = get_cos_sin(cfg["rope_theta"], d, device)

    fused_prepare_kv_kernel[(batch, hq + hkv)](
        qkv,
        data["positions"],
        cos_sin,
        data["cache_loc"],
        q_out,
        data["k_cache"],
        data["v_cache"],
        float(data["k_scale"].item()),
        float(data["v_scale"].item()),
        has_rope=True,
        stride_qkv_tok=qkv.stride(0),
        stride_q_tok=q_out.stride(0),
        stride_k_cache_page=data["k_cache"].stride(0),
        stride_k_cache_ps=data["k_cache"].stride(1),
        stride_k_cache_head=data["k_cache"].stride(2),
        stride_v_cache_page=data["v_cache"].stride(0),
        stride_v_cache_ps=data["v_cache"].stride(1),
        stride_v_cache_head=data["v_cache"].stride(2),
        page_size=cfg["page_size"],
        HEAD_DIM=d,
        num_q_heads=hq,
        num_kv_heads=hkv,
        qk_scale_factor=cfg["qk_scale_factor"],
        num_warps=2,
    )

    out = torch.empty((batch, hq * d), dtype=torch.bfloat16, device=device)
    gate = qkv[:, (hq + 2 * hkv) * d:]

    grid = (batch, hq)
    paged_sliding_attn_decode_kernel[grid](
        q_out,
        data["k_cache"],
        data["v_cache"],
        data["page_table"],
        data["seq_lens"],
        gate,
        out,
        float(data["k_scale"].item()),
        float(data["v_scale"].item()),
        sm_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        window=cfg["window"],
        stride_pt=data["page_table"].stride(0),
        stride_k_page=data["k_cache"].stride(0),
        stride_k_ps=data["k_cache"].stride(1),
        stride_k_head=data["k_cache"].stride(2),
        stride_v_page=data["v_cache"].stride(0),
        stride_v_ps=data["v_cache"].stride(1),
        stride_v_head=data["v_cache"].stride(2),
        page_size=cfg["page_size"],
        HEAD_DIM=d,
        num_q_heads=hq,
        num_kv_heads=hkv,
        BLOCK_N=64,
        num_warps=4,
    )

    from attn_reference import collect, check_implementation
    raw = collect(data, out.view(batch, 1, hq * d))
    passed, msg = check_implementation(data, raw)
    print("Verification result:", passed, msg)

    # CUDA graph timing
    def run_fn():
        paged_sliding_attn_decode_kernel[grid](
            q_out,
            data["k_cache"],
            data["v_cache"],
            data["page_table"],
            data["seq_lens"],
            gate,
            out,
            float(cfg["kv_scale"]),
            float(cfg["kv_scale"]),
            sm_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
            window=cfg["window"],
            stride_pt=data["page_table"].stride(0),
            stride_k_page=data["k_cache"].stride(0),
            stride_k_ps=data["k_cache"].stride(1),
            stride_k_head=data["k_cache"].stride(2),
            stride_v_page=data["v_cache"].stride(0),
            stride_v_ps=data["v_cache"].stride(1),
            stride_v_head=data["v_cache"].stride(2),
            page_size=cfg["page_size"],
            HEAD_DIM=d,
            num_q_heads=hq,
            num_kv_heads=hkv,
            BLOCK_N=64,
            num_warps=4,
        )

    # Warmup
    for _ in range(5):
        run_fn()
    torch.cuda.synchronize()

    # Capture CUDA graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_fn()

    # Time 100 replays
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()

    us = start_event.elapsed_time(end_event) * 1000 / 100
    print(f"CUDA Graph replay time for single-pass paged sliding attn (batch=1): {us:.2f} µs (SGLang was 13.4 µs)!")

if __name__ == "__main__":
    test()
