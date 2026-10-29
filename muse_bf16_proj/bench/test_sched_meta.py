import torch
from sgl_kernel.flash_attn import get_scheduler_metadata, flash_attn_with_kvcache
from attn_reference import generate_input, check_implementation, ref_kernel, collect

device = "cuda"
data = generate_input(batch=1, ctx=80000, heads="tp8", layer="sliding", seed=42, page=1, device=device)
cfg = data["config"]
hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
batch, length = cfg["batch"], cfg["q_len"]
window = cfg["window"]

from test_fused_prepare import fused_prepare_kv_kernel, get_cos_sin
qkv = data["qkv"].view(batch, -1)
q_out = torch.empty((batch, hq, d), dtype=torch.float8_e4m3fn, device=device)
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

# Test get_scheduler_metadata for sliding window
for num_splits in [0, 1, 2]:
    print(f"\n--- Testing num_splits = {num_splits} ---")
    sched = get_scheduler_metadata(
        batch_size=batch,
        max_seqlen_q=length,
        max_seqlen_k=int(data["cache_seqlens"].max()),
        num_heads=hq,
        num_heads_k=hkv,
        headdim=d,
        cache_seqlens=data["cache_seqlens"],
        qkv_dtype=torch.float8_e4m3fn,
        cu_seqlens_q=data["cu_seqlens_q"],
        page_size=cfg["page_size"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        num_splits=num_splits,
    )
    print("sched metadata shape:", sched.shape if sched is not None else None)

    out = flash_attn_with_kvcache(
        q=q_out,
        k_cache=data["k_cache"],
        v_cache=data["v_cache"],
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=sched,
    )
    if isinstance(out, tuple):
        out = out[0]

    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(batch, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=1e-5, has_norm=True)
    raw = collect(data, o.view(batch, 1, hq * d))
    passed, msg = check_implementation(data, raw)
    print(f"num_splits={num_splits}: correct = {passed}")
