# okc: problem=muse_attention_decode
"""Test FlashInfer CUDAGraphBatchDecodeWithPagedKVCacheWrapper with FP8 and page_size=1."""

import torch
import flashinfer

def kernel_fn(data):
    cfg = data["config"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    device = data["qkv"].device
    batch = cfg["batch"]
    page_size = cfg["page_size"]
    
    # Check if BatchDecodeWithPagedKVCacheWrapper supports FP8 and page_size=1
    try:
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            use_tensor_cores=True,
        )
        print("wrapper instantiated!")
        
        # Build indptr, indices, last_page_len for page_size=1
        seq_lens = data["cache_seqlens"]
        indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
        indptr[1:] = torch.cumsum(seq_lens, 0)
        
        # Flatten page table
        indices = torch.zeros(int(indptr[-1].item()), dtype=torch.int32, device=device)
        last_page_len = torch.ones(batch, dtype=torch.int32, device=device)
        
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
        print(f"wrapper.plan success with page_size={page_size}!")
    except Exception as e:
        print(f"wrapper error: {e}")
        
    return torch.zeros((cfg["batch"], cfg["q_len"], hq * d), dtype=torch.bfloat16, device="cuda")
