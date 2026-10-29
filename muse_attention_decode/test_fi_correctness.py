# okc: problem=muse_attention_decode
import torch
import flashinfer

_WORKSPACE = None

def get_workspace(device):
    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return _WORKSPACE

def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    device = data["qkv"].device
    page_size = cfg["page_size"]

    print(f"Testing flashinfer on batch={batch}, length={length}, window={cfg['window']}, page_size={page_size}")
    
    # Check if FlashInfer works for batch decode (length == 1)
    if length == 1:
        workspace = get_workspace(device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            use_tensor_cores=True,
        )
        
        # Build indptr, indices, last_page_len
        # seq_lens + 1 is total tokens per request
        cache_seqlens = data["cache_seqlens"]
        # Number of pages per sequence: (cache_seqlens + page_size - 1) // page_size
        num_pages_per_seq = (cache_seqlens + page_size - 1) // page_size
        indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
        indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)
        
        # Gather physical page indices from page_table
        total_pages = int(indptr[-1].item()) if not torch.cuda.is_current_stream_capturing() else data["page_table"].numel()
        # Flat indices
        indices_list = []
        for b in range(batch):
            n = int(num_pages_per_seq[b].item())
            indices_list.append(data["page_table"][b, :n])
        indices = torch.cat(indices_list, dim=0).contiguous()
        last_page_len = (cache_seqlens - 1) % page_size + 1
        
        print(f"Planning FlashInfer with total_pages={indices.shape[0]}...")
        wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=hq,
            num_kv_heads=hkv,
            head_dim=d,
            page_size=page_size,
        )
        print("Plan succeeded!")
        
    return data["qkv"][:, :, :16*64]
