# okc: problem=muse_attention_decode
import inspect
import torch
import flashinfer

_WS = None

def get_ws(device):
    global _WS
    if _WS is None:
        _WS = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return _WS

def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    device = data["qkv"].device
    print(f"=== Testing trtllm_batch_decode: batch={batch}, q_len={length}, window={cfg['window']}, page_size={cfg['page_size']} ===")

    ws = get_ws(device)
    k_cache = data["k_cache"]
    v_cache = data["v_cache"]
    page_table = data["page_table"]
    seq_lens = data["cache_seqlens"] # or seq_lens

    print("k_cache shape:", k_cache.shape, k_cache.dtype)
    print("page_table shape:", page_table.shape, page_table.dtype)
    print("seq_lens shape:", seq_lens.shape, seq_lens.dtype)

    # dummy query
    q = torch.zeros((batch, hq, d), dtype=torch.bfloat16, device=device)

    # Try calling trtllm_batch_decode_with_kv_cache with kv_layout='NHD'
    try:
        res = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=ws,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=int(seq_lens.max().item()),
            bmm1_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
            bmm2_scale=1.0,
            window_left=(cfg["window"] - 1) if cfg["window"] > 0 else -1,
            kv_layout="NHD",
        )
        print("trtllm NHD call succeeded! res shape:", res.shape if hasattr(res, 'shape') else type(res))
    except Exception as e:
        print("trtllm NHD error:", e)

    # Try calling with kv_layout='HND'
    try:
        res = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=ws,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=int(seq_lens.max().item()),
            bmm1_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
            bmm2_scale=1.0,
            window_left=(cfg["window"] - 1) if cfg["window"] > 0 else -1,
            kv_layout="HND",
        )
        print("trtllm HND call succeeded! res shape:", res.shape if hasattr(res, 'shape') else type(res))
    except Exception as e:
        print("trtllm HND error:", e)

    return data["qkv"][:, :, :16*64]
