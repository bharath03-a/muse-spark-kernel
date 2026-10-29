# okc: problem=muse_decode_layer
"""Probe XQA kernel execution and B12xRunner details."""

import inspect
import torch
import baseline
from flashinfer.decode import xqa_batch_decode_with_kv_cache

def prepare(weights):
    state = baseline.prepare(weights)
    device = weights["qkv_gate"].device
    
    print("=== XQA SOURCE ===")
    try:
        import flashinfer.xqa as fxqa
        for name in dir(fxqa):
            if not name.startswith("_"):
                print("fxqa member:", name)
        if hasattr(fxqa, "xqa"):
            print("xqa doc/sig:", inspect.signature(fxqa.xqa))
    except Exception as e:
        print("XQA inspect error:", e)

    print("=== B12xRunner DETAILS ===")
    try:
        from flashinfer.fused_moe.runners import B12xW4A16Runner
        print("B12xW4A16Runner init:", inspect.signature(B12xW4A16Runner.__init__))
        print("B12xW4A16Runner methods:", [m for m in dir(B12xW4A16Runner) if not m.startswith("_")])
        # Print run or forward source
        for meth in ["run", "forward", "_run", "execute"]:
            if hasattr(B12xW4A16Runner, meth):
                print(f"B12x {meth} source:")
                print(inspect.getsource(getattr(B12xW4A16Runner, meth)))
    except Exception as e:
        print("B12x error:", e)

    # Preallocate 64MB workspace for XQA
    state["xqa_ws"] = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    state["xqa_tested"] = False
    return state

def kernel_fn(data, state):
    cfg = data["config"]
    # Test XQA once
    if not state["xqa_tested"]:
        state["xqa_tested"] = True
        try:
            tokens = cfg["tokens"]
            hq = cfg["num_q_heads"]
            d = cfg["head_dim"]
            hkv = cfg["num_kv_heads"]
            batch = cfg["batch"]
            length = cfg["q_len"]
            
            # Run baseline attention preparation
            qkv = baseline._gemm(data["attn_input"], state["dense"]["qkv_gate"])
            if cfg["rope_theta"]:
                q_bf16, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
            else:
                q_bf16, k = state["qk_norm"](qkv)
            
            # Test XQA with BF16 query
            window = cfg["window"]
            window_left = (window - 1) if window > 0 else -1
            sm_scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
            
            print(f"Calling XQA: batch={batch}, q_len={length}, window={window}, window_left={window_left}")
            out_xqa_bf16 = xqa_batch_decode_with_kv_cache(
                query=q_bf16.view(tokens, hq, d),
                kv_cache=(data["k_cache"], data["v_cache"]),
                workspace_buffer=state["xqa_ws"],
                block_tables=data["page_table"],
                seq_lens=data["cache_seqlens"],
                max_seq_len=1048576,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                window_left=window_left,
                kv_layout="NHD",
                q_len_per_req=length,
            )
            print("XQA BF16 query SUCCESS! Output shape:", out_xqa_bf16.shape)

            # Compare with FA3
            fa3_out = baseline.flash_attn_with_kvcache(
                q=q_bf16.view(tokens, hq, d).to(baseline.FP8),
                k_cache=data["k_cache"],
                v_cache=data["v_cache"],
                page_table=data["page_table"],
                cache_seqlens=data["cache_seqlens"],
                cu_seqlens_q=data["cu_seqlens_q"],
                max_seqlen_q=length,
                softmax_scale=sm_scale,
                causal=True,
                window_size=(window - 1, 0) if window else (-1, -1),
                scheduler_metadata=data["scheduler_metadata"],
            )
            if isinstance(fa3_out, tuple):
                fa3_out = fa3_out[0]
            
            rel_diff = (out_xqa_bf16.float() - fa3_out.float()).norm() / fa3_out.float().norm()
            print(f"XQA vs FA3 relative diff: {rel_diff.item():.4e}")

        except Exception as e:
            import traceback
            print("XQA test EXCEPTION:", e)
            traceback.print_exc()

    return baseline.kernel_fn(data, state)
