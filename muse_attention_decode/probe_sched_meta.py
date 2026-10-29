# okc: problem=muse_attention_decode
"""Probe get_scheduler_metadata behavior and signature."""

import torch
from sgl_kernel.flash_attn import get_scheduler_metadata
import inspect

def kernel_fn(data):
    cfg = data["config"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    sig = inspect.signature(get_scheduler_metadata)
    print("get_scheduler_metadata signature:", sig)
    
    # Test for verify 1 full (batch=1, q_len=8, ctx=80000)
    seqlens = data["cache_seqlens"]
    print("seqlens:", seqlens)
    print("cu_seqlens_q:", data["cu_seqlens_q"])
    print("scheduler_metadata passed in data:", data["scheduler_metadata"] is not None)
    
    for splits in [0, 2, 4, 8, 16]:
        try:
            m = get_scheduler_metadata(
                batch_size=cfg["batch"],
                max_seqlen_q=cfg["q_len"],
                max_seqlen_k=80008,
                num_heads=hq,
                num_heads_k=hkv,
                headdim=d,
                cache_seqlens=seqlens,
                qkv_dtype=torch.float8_e4m3fn,
                cu_seqlens_q=data["cu_seqlens_q"],
                page_size=cfg["page_size"],
                causal=True,
                num_splits=splits,
            )
            print(f"splits={splits}: meta shape={m.shape}, dtype={m.dtype}, num_splits={m[0].item() if m.numel() > 0 else 'N/A'}")
        except Exception as e:
            print(f"splits={splits}: ERROR {e}")
            
    return torch.zeros((cfg["batch"], cfg["q_len"], hq * d), dtype=torch.bfloat16, device="cuda")
