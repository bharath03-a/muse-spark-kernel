import torch
from sgl_kernel.flash_attn import get_scheduler_metadata

device = "cuda"
cache_seqlens = torch.full((1,), 80000, dtype=torch.int32, device=device)
cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)

for k in [2048, 80000, 100000, 1048576]:
    s = get_scheduler_metadata(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=k,
        num_heads=16,
        num_heads_k=2,
        headdim=64,
        cache_seqlens=cache_seqlens,
        qkv_dtype=torch.float8_e4m3fn,
        cu_seqlens_q=cu_seqlens_q,
        page_size=1,
        causal=True,
        window_size=(2047, 0),
        num_splits=1,
    )
    print(f"max_seqlen_k={k}: shape={s.shape}")
